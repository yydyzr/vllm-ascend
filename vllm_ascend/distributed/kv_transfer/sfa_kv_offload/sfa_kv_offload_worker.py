from __future__ import annotations

import os
import threading
import zlib
from typing import Optional
from collections.abc import Generator

import numpy as np
import torch
from torch.utils.cpp_extension import load
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer
from memfabric_hybrid import offload

from vllm_ascend import envs
from vllm_ascend.ascend_config import KV_OFFLOAD_MODE_FUSED_OVERLAP, get_ascend_config
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    SFAKVOffloadConnectorMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    KVTransferThread,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.offload_kv_cache_layout import (
    OFFLOAD_C8_TUPLE_LEN,
    OFFLOAD_INDEXER_S,
    OFFLOAD_MAIN_K,
    OFFLOAD_MAIN_V,
    OFFLOAD_RESIDENT_K,
    OFFLOAD_RESIDENT_V,
    OFFLOAD_TUPLE_LEN,
    is_offload_c8_kv_cache,
)

_SUBSCRIBED_COMPUTE_STREAMS = set()
def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS

def _is_current_stream_capturing() -> bool:
    for npu_runtime in (getattr(torch_npu, "npu", None), getattr(torch, "npu", None)):
        if npu_runtime is None:
            continue
        for attr_name in ("is_current_stream_capturing", "_is_current_stream_capturing"):
            capture_state = getattr(npu_runtime, attr_name, None)
            if not callable(capture_state):
                continue
            try:
                if bool(capture_state()):
                    return True
            except Exception:
                continue
    return False

# cpu sparse attn kernel related
# TODO maybe implement this in vllm custom op framework
os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
# cache_dir = "/root/.cache/torch_extensions/py311_cpu/cpu_sparse_attn"
# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)
#     print(f"已清理缓存目录: {cache_dir}")
ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
npu_include_path = os.path.join(ascend_home, "include")
npu_lib_path = os.path.join(ascend_home, "lib64")
if not os.path.exists(npu_lib_path):
    npu_lib_path = os.path.join(ascend_home, "lib")
torch_npu_path = os.path.dirname(torch_npu.__file__)
torch_npu_include = os.path.join(torch_npu_path, "include")
torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
os.environ['CXX'] = 'clang++'
os.environ['CC'] = 'clang'
abs_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(abs_path, "cpu_sparse_attn.cpp")
logger.info(f'>>>>> load cpu_sparse_attn from src: {src_path}')
cpu_sparse_attn = None
cpu_sparse_attn = load(
    name="cpu_sparse_attn",
    sources=[src_path],
    extra_cflags=[
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=armv8.2-a+sve+fp16+bf16",
        # "-march=native",
        "-fPIC",
        f"-I{npu_include_path}",
        f"-I{torch_npu_include}",
    ],
    extra_ldflags=[
        "-fopenmp",
        f"-L{npu_lib_path}",
        "-lascendcl",
        f"-L{torch_npu_lib_path}",
        "-ltorch_npu",
    ],
    verbose=True,  # 添加 verbose 查看编译过程
)


class SFAKVOffloadWorker:
    # The main class for the cache engine.

    _CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024

    @staticmethod
    def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    @classmethod
    def _empty_aligned_cpu_tensor(
        cls,
        shape: list[int],
        dtype: torch.dtype,
        alignment: int = _CPU_CACHE_ALIGNMENT,
    ) -> torch.Tensor:
        num_elements = int(np.prod(shape))
        extra_elements = cdiv(alignment, torch.empty((), dtype=dtype).element_size())
        tensor = offload.empty([num_elements + extra_elements], dtype=dtype, pin_memory=True)
        return cls._align_memory(tensor, alignment)[:num_elements].view(shape)

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.dp_rank = parallel_config.data_parallel_rank
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")
        self.use_layerwise = use_layerwize
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        ascend_config = get_ascend_config()
        self.use_offload = ascend_config.use_offload
        self.kv_offload_mode = ascend_config.kv_offload_mode
        self.use_fused_overlap_offload = (
            self.use_offload and self.kv_offload_mode == KV_OFFLOAD_MODE_FUSED_OVERLAP
        )

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.group_block_sizes = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        self.block_size = self.group_block_sizes[-1] # only offload kv cache

        self.current_layer_save = 0
        self.current_layer_load = 0
        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.num_offload_layers = self.num_target_layers
        self.num_layers = self.num_offload_layers
        self.offload_layer_names: list[str] = []
        self.layer_name_to_offload_id: dict[str, int] = {}

        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        self.kv_send_thread: KVTransferThread | None = None
        self.layer_save_tasks: list[list[LayerMultiBlockReqMeta]] = []
        self.pending_save_layer_ids: set[int] = set()
        self.submitted_save_layer_ids: set[int] = set()
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        lru_resident_config = ascend_config.lru_resident_cache_config
        self.sfa_sparse_topk = lru_resident_config.topk
        self.lru_resident_capacity = lru_resident_config.buffer_size

        # TODO get from config
        head_num = 1
        head_dim_k = 512
        head_dim_v = 64
        dtype = torch.bfloat16
        self.token_size_bytes_k = head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = head_num * head_dim_v * dtype.itemsize
        self.max_model_len = vllm_config.model_config.max_model_len
        max_block_num = cdiv(self.max_model_len, self.block_size)
        self.cpu_block_table = CpuGpuBuffer(self.max_num_reqs, max_block_num, dtype=torch.int32, device='npu', pin_memory=True)
        self.cpu_block_table_host_buffer = torch.zeros([self.max_num_reqs, max_block_num], dtype=torch.int32, device='cpu', pin_memory=True)
        self.cpu_block_table_req_hashes = torch.empty(
            self.max_num_reqs,
            dtype=torch.int64,
            device='cpu',
            pin_memory=True,
        )
        self.lru_expanded_block_table_cpu = torch.empty(
            [self.max_num_topk_rows, max_block_num],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.actual_seq_len_q = torch.arange(self.max_num_reqs, dtype=torch.int32, device='cpu', pin_memory=True) + 1
        self.req_ids = []

        self.cpu_sparse_attn = cpu_sparse_attn

        self.load_stream = None
        self.load_stream = torch_npu.npu.Stream()
        self.save_stream = None
        self.side_compute_stream = torch_npu.npu.Stream()
        if self.use_fused_overlap_offload:
            self.fused_step_requests: list[ReqMeta] = []
            self._fused_req_meta_by_id: dict[str, ReqMeta] = {}
            self.d2h_save_event = torch_npu.npu.Event()
            self.fused_offload_token_start_cpu = torch.full(
                [self.max_num_reqs],
                -1,
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.fused_offload_num_tokens_cpu = torch.zeros(
                [self.max_num_reqs],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.d2h_slot_mapping_cpu = torch.empty(
                [self.max_num_tokens],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.d2h_token_to_req_cpu = torch.empty(
                [self.max_num_tokens],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.d2h_cum_query_lens_cpu = torch.empty(
                [self.max_num_reqs],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
        self.allocate_dram_size = 64 * 1024 * 1024 * 1024 # TODO get from config
        logger.info(
            f"SFAKVOffloadWoker start init h2d with {self.allocate_dram_size / 1024 / 1024 / 1024} GB dram/rank, "
            "it might be time consuming, please wait."
        )
        offload.initialize(self.tp_rank, self.allocate_dram_size)

    def _infer_group_block_sizes(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig | None,
    ) -> list[int]:
        block_sizes: list[int] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            block_sizes.append(kv_cache_spec.block_size)
        return block_sizes

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.offload_layer_names = [
            layer_name
            for layer_name, cache_or_caches in kv_caches.items()
            if len(self._as_cache_tuple(cache_or_caches))
            in (OFFLOAD_TUPLE_LEN, OFFLOAD_C8_TUPLE_LEN)
        ]
        if not self.offload_layer_names:
            raise ValueError("SFA KV Offload did not find SFA KV cache layers.")

        # Under offload, the attention path (sfa_v1.py / device_op.py) gates the
        # C8 indexer read on the GLOBAL use_sparse_c8_indexer flag, not per-layer.
        # Mixed five/six-tuple layers would therefore route a non-C8 layer through
        # the quant indexer (or vice versa). Forbid it here so the global gate
        # stays sound; C8 must be all-or-nothing across sparse offload layers.
        tuple_lens = {
            len(self._as_cache_tuple(kv_caches[name])) for name in self.offload_layer_names
        }
        if len(tuple_lens) > 1:
            raise ValueError(
                "SFA KV offload does not support mixed LIC8 / non-LIC8 layers: "
                f"found tuple lengths {sorted(tuple_lens)} "
                f"(five-tuple and six-tuple coexist). Under offload, C8 must be "
                f"enabled uniformly across all sparse layers."
            )

        self.num_offload_layers = len(self.offload_layer_names)
        self.num_layers = self.num_offload_layers
        self.layer_name_to_offload_id = {
            layer_name: layer_id
            for layer_id, layer_name in enumerate(self.offload_layer_names)
        }
        self.layer_save_tasks = [[] for _ in range(self.num_layers)]
        self.pending_save_layer_ids.clear()
        self.submitted_save_layer_ids.clear()

        logger.info(
            "SFA KV offload registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("SFA KV offload layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "SFA KV offload layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache_tuple = self._as_cache_tuple(first_kv_cache_tuple)
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = (
            self.kv_cache_config.num_blocks if self.kv_cache_config is not None else first_kv_cache.shape[0]
        )
        logger.info("num_blocks: %s", self.num_blocks)

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        if self.use_sparse and self.use_offload:
            self._register_offload_layers(kv_caches)
            self.k_caches_npu: list[torch.Tensor] = []
            self.v_caches_npu: list[torch.Tensor] = []
            self.topk_buffers_k: list[torch.Tensor] = []
            self.topk_buffers_v: list[torch.Tensor] = []
            for layer_name in self.offload_layer_names:
                cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
                tuple_len = len(cache_or_caches)
                if tuple_len not in (OFFLOAD_TUPLE_LEN, OFFLOAD_C8_TUPLE_LEN):
                    raise ValueError(
                        f"SFA KV offload layer {layer_name}: expected tuple length "
                        f"{OFFLOAD_TUPLE_LEN} or {OFFLOAD_C8_TUPLE_LEN}, got {tuple_len}"
                    )
                self.k_caches_npu.append(cache_or_caches[OFFLOAD_MAIN_K])
                self.v_caches_npu.append(cache_or_caches[OFFLOAD_MAIN_V])
                self.topk_buffers_k.append(cache_or_caches[OFFLOAD_RESIDENT_K])
                self.topk_buffers_v.append(cache_or_caches[OFFLOAD_RESIDENT_V])
                if is_offload_c8_kv_cache(cache_or_caches):
                    # Guard against the LIC8 scale tensor aliasing a resident
                    # top-K buffer. Compare storage identity (data_ptr), NOT
                    # `in`/`==`: those do element-wise comparison and raise a
                    # shape-mismatch RuntimeError on the legitimate difference
                    # (scale dim1 = dsa_block_size, resident dim1 =
                    # resident_capacity).
                    scale_storage_ptr = cache_or_caches[
                        OFFLOAD_INDEXER_S].untyped_storage().data_ptr()
                    if scale_storage_ptr in (
                        cache_or_caches[OFFLOAD_RESIDENT_K]
                        .untyped_storage().data_ptr(),
                        cache_or_caches[OFFLOAD_RESIDENT_V]
                        .untyped_storage().data_ptr(),
                    ):
                        raise ValueError(
                            f"LIC8 scale tensor must not alias resident buffer: {layer_name}"
                        )

            if self.use_layerwise:
                ready_event = threading.Event()
                self.layer_save_finished_events = [threading.Event() for _ in range(self.num_layers)]
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.block_size,
                    self.num_layers,
                    self.tp_rank,
                    ready_event,
                    self.layer_save_finished_events,
                )
                self.kv_send_thread.start()
                ready_event.wait()
            else:
                raise ValueError("SFA KV Offload only support layerwise now.")

            npu_block_num = self.num_blocks
            # we need 4 * npu_blocks of cpu_blocks to fully store all offload blocks (dskv32, 512/128)
            # but you may want to set this to 1 in debug case in case of allocating to much dram
            # TODO remove this and directly compute from model config before merge
            cpu_block_num_multiple = 1
            cpu_block_num = npu_block_num * cpu_block_num_multiple
            cpu_cache_size_single_card = cpu_block_num * self.block_size * (512 + 64) * torch.bfloat16.itemsize * self.num_layers
            logger.info(f'KV offload allocate {cpu_block_num} cpu blocks, size = {cpu_cache_size_single_card / 1024 / 1024 / 1024} GB per rank')
            if cpu_cache_size_single_card > self.allocate_dram_size:
                raise ValueError(
                    f"Needed cpu memory ({cpu_cache_size_single_card / 1024 / 1024 / 1024} GB/rank) is greater than "
                    f"available cpu memory ({self.allocate_dram_size / 1024 / 1024 / 1024} GB/rank), "
                    "try to decrease gpu_memory_utilization or allocate more cpu memory during init."
                )
            self.k_caches_cpu: list[torch.Tensor] = [
                self._empty_aligned_cpu_tensor([cpu_block_num, self.block_size, 1, 512], dtype=torch.bfloat16)
                for _ in range(self.num_layers)
            ]
            self.v_caches_cpu: list[torch.Tensor] = [
                self._empty_aligned_cpu_tensor([cpu_block_num, self.block_size, 1, 64], dtype=torch.bfloat16)
                for _ in range(self.num_layers)
            ]
            if self.use_fused_overlap_offload:
                logger.info(
                    "[fused_overlap_offload][init] layer_count=%s cpu_block_num=%s "
                    "k_cpu_shape=%s rope_cpu_shape=%s k_cpu_device=%s rope_cpu_device=%s "
                    "k_cpu_ptr=%s rope_cpu_ptr=%s cpu_block_table_shape=%s",
                    self.num_layers,
                    cpu_block_num,
                    tuple(self.k_caches_cpu[0].shape),
                    tuple(self.v_caches_cpu[0].shape),
                    self.k_caches_cpu[0].device,
                    self.v_caches_cpu[0].device,
                    self.k_caches_cpu[0].data_ptr(),
                    self.v_caches_cpu[0].data_ptr(),
                    tuple(self.cpu_block_table.gpu.shape),
                )

            # topk cache reuse related
            self.lru_workspace_threads = 8
            self.lru_topk_indices_cpu = torch.empty(
                [self.max_num_topk_rows, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_token_to_req_cpu = torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_slot_to_token_cpu_list = [torch.full(
                [self.max_num_topk_rows, self.lru_resident_capacity],
                -1,
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_slots_cpu_list = [torch.arange(
                self.lru_resident_capacity,
                dtype=torch.int32,
                device='cpu',
            ).view(1, -1).repeat(self.max_num_topk_rows, 1).pin_memory() for _ in range(self.num_layers)]
            self.lru_current_slots_cpu = torch.empty(
                [self.max_num_topk_rows, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_miss_count_cpu_list = [torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_miss_tokens_cpu_list = [torch.empty(
                [self.max_num_topk_rows, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_miss_slots_cpu_list = [torch.empty(
                [self.max_num_topk_rows, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device='cpu', pin_memory=True)
            self.lru_last_req_ids_cpu_list = [torch.full(
                [self.max_num_topk_rows],
                -1,
                dtype=torch.int64,
                device='cpu',
                pin_memory=True,
            ) for _  in range(self.num_layers)]
            self.lru_token_mark_workspace = torch.zeros(
                [self.lru_workspace_threads, self.max_model_len],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_token_pos_workspace = torch.full(
                [self.lru_workspace_threads, self.max_model_len],
                -1,
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_slot_workspace = torch.empty(
                [self.lru_workspace_threads, self.lru_resident_capacity * 3],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_miss_position_workspace = torch.empty(
                [self.lru_workspace_threads, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_epochs = torch.zeros(
                [self.lru_workspace_threads],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )

            self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
            self.lru_last_req_ids_ptrs = [lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list]
            self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
            self.lru_token_to_req_ptr = self.lru_token_to_req_cpu.data_ptr()
            self.lru_slot_to_token_ptrs = [lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list]
            self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
            self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
            self.lru_miss_count_ptrs = [lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list]
            self.lru_miss_tokens_ptrs = [lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list]
            self.lru_miss_slots_ptrs = [lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list]
            self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
            self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
            self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
            self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
            self.lru_epochs_ptr = self.lru_epochs.data_ptr()

            # sparse h2d (sparse_copy related)
            self.addr_k_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_k]
            self.addr_v_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_v]
            self.gvas_k_bases: list[int] = [t.data_ptr() for t in self.k_caches_cpu]
            self.gvas_v_bases: list[int] = [t.data_ptr() for t in self.v_caches_cpu]
            self.npu_k_bases: list[int] = [t.data_ptr() for t in self.k_caches_npu]
            self.npu_v_bases: list[int] = [t.data_ptr() for t in self.v_caches_npu]

            gvas_buffer_offset = 0
            gvas_buffer_size_bytes = self.max_num_topk_rows * self.sfa_sparse_topk * 2 * 8 # 2: k+v, 8: int64
            addr_buffer_offset = gvas_buffer_offset + gvas_buffer_size_bytes
            addr_buffer_size_bytes = self.max_num_topk_rows * self.sfa_sparse_topk * 2 * 8
            size_buffer_offset = addr_buffer_offset + addr_buffer_size_bytes
            size_buffer_size_bytes = self.max_num_topk_rows * self.sfa_sparse_topk * 2 * 4 # 2: k+v, 4: int32
            num_tokens_buffer_offset = size_buffer_offset + size_buffer_size_bytes
            num_tokens_buffer_size_bytes = 4
            sparse_copy_args_buffer_size_bytes = gvas_buffer_size_bytes + addr_buffer_size_bytes + size_buffer_size_bytes + num_tokens_buffer_size_bytes
            self.sparse_copy_args_buffer_cpu = torch.zeros([sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device='cpu', pin_memory=True)
            self.sparse_copy_args_buffer_npu = torch.zeros([sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device='npu')

            self.gvas_buffer_cpu = self.sparse_copy_args_buffer_cpu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
            self.addr_buffer_cpu = self.sparse_copy_args_buffer_cpu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
            self.size_buffer_cpu = self.sparse_copy_args_buffer_cpu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
            self.num_tokens_buffer_cpu = \
                self.sparse_copy_args_buffer_cpu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
            assert self.gvas_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.addr_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.size_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.num_tokens_buffer_cpu.shape == torch.Size([1])

            self.gvas_buffer_npu = self.sparse_copy_args_buffer_npu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
            self.addr_buffer_npu = self.sparse_copy_args_buffer_npu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
            self.size_buffer_npu = self.sparse_copy_args_buffer_npu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
            self.num_tokens_buffer_npu = \
                self.sparse_copy_args_buffer_npu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
            assert self.gvas_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.addr_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.size_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.num_tokens_buffer_npu.shape == torch.Size([1])

            if self.use_fused_overlap_offload:
                # fused_overlap d2h: gvas/addr buffers follow H2D naming (CPU/NPU bases).
                # sparse_copy copies gvas->addr, so pass addr(NPU src) first for D2H.
                d2h_max_copies = self.max_num_tokens * 2
                d2h_gvas_offset = 0
                d2h_gvas_size_bytes = d2h_max_copies * 8
                d2h_addr_offset = d2h_gvas_offset + d2h_gvas_size_bytes
                d2h_addr_size_bytes = d2h_max_copies * 8
                d2h_size_offset = d2h_addr_offset + d2h_addr_size_bytes
                d2h_size_size_bytes = d2h_max_copies * 4
                d2h_num_tokens_offset = d2h_size_offset + d2h_size_size_bytes
                d2h_num_tokens_size_bytes = 4
                d2h_batch_copy_args_size_bytes = (
                    d2h_gvas_size_bytes + d2h_addr_size_bytes + d2h_size_size_bytes + d2h_num_tokens_size_bytes
                )
                self.d2h_batch_copy_args_buffer_cpu = torch.zeros(
                    [d2h_batch_copy_args_size_bytes], dtype=torch.int8, device='cpu', pin_memory=True
                )
                self.d2h_batch_copy_args_buffer_npu = torch.zeros(
                    [d2h_batch_copy_args_size_bytes], dtype=torch.int8, device='npu'
                )
                self.d2h_gvas_buffer_cpu = self.d2h_batch_copy_args_buffer_cpu[
                    d2h_gvas_offset:d2h_gvas_offset + d2h_gvas_size_bytes
                ].view(torch.int64)
                self.d2h_addr_buffer_cpu = self.d2h_batch_copy_args_buffer_cpu[
                    d2h_addr_offset:d2h_addr_offset + d2h_addr_size_bytes
                ].view(torch.int64)
                self.d2h_size_buffer_cpu = self.d2h_batch_copy_args_buffer_cpu[
                    d2h_size_offset:d2h_size_offset + d2h_size_size_bytes
                ].view(torch.int32)
                self.d2h_num_tokens_buffer_cpu = self.d2h_batch_copy_args_buffer_cpu[
                    d2h_num_tokens_offset:d2h_num_tokens_offset + d2h_num_tokens_size_bytes
                ].view(torch.int32)
                self.d2h_gvas_buffer_npu = self.d2h_batch_copy_args_buffer_npu[
                    d2h_gvas_offset:d2h_gvas_offset + d2h_gvas_size_bytes
                ].view(torch.int64)
                self.d2h_addr_buffer_npu = self.d2h_batch_copy_args_buffer_npu[
                    d2h_addr_offset:d2h_addr_offset + d2h_addr_size_bytes
                ].view(torch.int64)
                self.d2h_size_buffer_npu = self.d2h_batch_copy_args_buffer_npu[
                    d2h_size_offset:d2h_size_offset + d2h_size_size_bytes
                ].view(torch.int32)
                self.d2h_num_tokens_buffer_npu = self.d2h_batch_copy_args_buffer_npu[
                    d2h_num_tokens_offset:d2h_num_tokens_offset + d2h_num_tokens_size_bytes
                ].view(torch.int32)

    def start_load_kv(self, metadata: SFAKVOffloadConnectorMetadata):
        # return
        self.current_layer_save = 0
        self.current_layer_load = 0
        req_id_to_block_ids: dict[str, list[int]] = {}
        if self.use_fused_overlap_offload:
            self.fused_step_requests = []
            self._fused_req_meta_by_id = {}
        for layer_save_task in self.layer_save_tasks:
            layer_save_task.clear()
        self.pending_save_layer_ids.clear()
        self.submitted_save_layer_ids.clear()
        for event in getattr(self, "layer_save_finished_events", []):
            event.clear()
        for request in metadata.requests:
            req_id_to_block_ids[request.req_id] = request.block_ids_cpu
            if self.use_fused_overlap_offload:
                if request.offload_num_tokens > 0:
                    self.fused_step_requests.append(request)
                    self._fused_req_meta_by_id[request.req_id] = request
                continue
            if request.num_new_offload_blocks <= 0:
                continue # no new blocks to save
            self.process_layer_data(request)
        num_save_layers = sum(1 for layer_save_task in self.layer_save_tasks if layer_save_task)
        self.num_save_tasks = sum(len(layer_save_task) for layer_save_task in self.layer_save_tasks)
        if self.tp_rank == 0:
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(
                    f'>>>>> start load kv, reqs num: {len(metadata.requests)}, '
                    f'save layer num = {num_save_layers}, save task num = {self.num_save_tasks}'
                )

        # generate block_table for load
        # NOTE reqs in self.req_ids and metadata.requests may not be in same order,
        # use reqs from self.req_ids (order of actual batch) to compute block_table.
        num_reqs = len(self.req_ids)
        cpu_block_table_np = self.cpu_block_table.np[:num_reqs]
        cpu_block_table_np.fill(0)
        for i, req_id in enumerate(self.req_ids[:num_reqs]):
            cpu_block_ids = req_id_to_block_ids[req_id]
            cpu_block_table_np[i][:len(cpu_block_ids)] = np.array([cpu_block_ids], dtype=np.int32)
            if envs.VLLM_ASCEND_SFA_DEBUG:
                self.cpu_block_table_req_hashes[i] = zlib.adler32(
                    req_id.encode('utf-8')
                )
        self.cpu_block_table.copy_to_gpu(num_reqs)

        if self.use_fused_overlap_offload:
            self.fused_offload_token_start_cpu.fill_(-1)
            self.fused_offload_num_tokens_cpu.zero_()
            for req_idx, req_id in enumerate(self.req_ids[:num_reqs]):
                req_meta = self._fused_req_meta_by_id.get(req_id)
                if req_meta is None:
                    continue
                self.fused_offload_token_start_cpu[req_idx] = req_meta.offload_token_start
                self.fused_offload_num_tokens_cpu[req_idx] = req_meta.offload_num_tokens

    def _compute_step_offload_addrs_cpu(
        self,
        args: tuple[int, int, int],
    ) -> None:
        num_actual_tokens, num_reqs, layer_id = args
        if num_actual_tokens <= 0:
            self.d2h_num_tokens_buffer_cpu.zero_()
            return

        block_size_bytes_k = self.block_size * self.token_size_bytes_k
        block_size_bytes_v = self.block_size * self.token_size_bytes_v
        npu_k_base = self.npu_k_bases[layer_id]
        npu_v_base = self.npu_v_bases[layer_id]
        cpu_k_base = self.gvas_k_bases[layer_id]
        cpu_v_base = self.gvas_v_bases[layer_id]

        slots = self.d2h_slot_mapping_cpu[:num_actual_tokens]
        token_to_req = self.d2h_token_to_req_cpu[:num_actual_tokens]
        cum_query_lens = self.d2h_cum_query_lens_cpu[:num_reqs]
        cpu_block_table = self.cpu_block_table_host_buffer[:num_reqs]
        offload_token_start = self.fused_offload_token_start_cpu[:num_reqs]
        offload_num_tokens = self.fused_offload_num_tokens_cpu[:num_reqs]

        def _get_step_offload_addrs(batch_idx: int) -> tuple[int, int, int, int] | None:
            req_idx = int(token_to_req[batch_idx].item())
            if req_idx < 0 or req_idx >= num_reqs:
                return None
            if int(offload_num_tokens[req_idx].item()) <= 0:
                return None

            query_start = 0 if req_idx == 0 else int(cum_query_lens[req_idx - 1].item())
            local_offset = batch_idx - query_start
            if local_offset < 0 or local_offset >= int(offload_num_tokens[req_idx].item()):
                return None

            slot = int(slots[batch_idx].item())
            if slot < 0:
                return None

            global_pos = int(offload_token_start[req_idx].item()) + local_offset
            npu_block_id = slot // self.block_size
            npu_offset_in_block = slot % self.block_size
            cpu_block_idx = global_pos // self.block_size
            cpu_offset_in_block = global_pos % self.block_size
            cpu_block_id = int(cpu_block_table[req_idx, cpu_block_idx].item())
            if cpu_block_id <= 0:
                return None

            gva_k = (
                cpu_k_base
                + cpu_block_id * block_size_bytes_k
                + cpu_offset_in_block * self.token_size_bytes_k
            )
            addr_k = (
                npu_k_base
                + npu_block_id * block_size_bytes_k
                + npu_offset_in_block * self.token_size_bytes_k
            )
            gva_v = (
                cpu_v_base
                + cpu_block_id * block_size_bytes_v
                + cpu_offset_in_block * self.token_size_bytes_v
            )
            addr_v = (
                npu_v_base
                + npu_block_id * block_size_bytes_v
                + npu_offset_in_block * self.token_size_bytes_v
            )
            return gva_k, addr_k, gva_v, addr_v

        v_staging_offset = num_actual_tokens
        k_idx = 0
        for batch_idx in range(num_actual_tokens):
            addrs = _get_step_offload_addrs(batch_idx)
            if addrs is None:
                continue
            gva_k, addr_k, gva_v, addr_v = addrs
            self.d2h_gvas_buffer_cpu[k_idx] = gva_k
            self.d2h_addr_buffer_cpu[k_idx] = addr_k
            staging_idx = v_staging_offset + k_idx
            self.d2h_gvas_buffer_cpu[staging_idx] = gva_v
            self.d2h_addr_buffer_cpu[staging_idx] = addr_v
            k_idx += 1

        num_k_copies = k_idx
        copy_idx = num_k_copies * 2
        if num_k_copies > 0:
            if num_k_copies < num_actual_tokens:
                self.d2h_gvas_buffer_cpu[num_k_copies:copy_idx] = (
                    self.d2h_gvas_buffer_cpu[v_staging_offset:v_staging_offset + num_k_copies]
                )
                self.d2h_addr_buffer_cpu[num_k_copies:copy_idx] = (
                    self.d2h_addr_buffer_cpu[v_staging_offset:v_staging_offset + num_k_copies]
                )
            self.d2h_size_buffer_cpu[:num_k_copies].fill_(self.token_size_bytes_k)
            self.d2h_size_buffer_cpu[num_k_copies:copy_idx].fill_(self.token_size_bytes_v)
        self.d2h_num_tokens_buffer_cpu[0] = copy_idx

    def save_cpu(self, layer_id: int | None = None) -> None:
        if layer_id is None:
            layer_id = self.current_layer_save
            self.current_layer_save += 1
            if self.current_layer_save == self.num_layers:
                self.current_layer_save = 0
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(f"SFA KV offload layer id out of range: {layer_id}")
        if not self.layer_save_tasks[layer_id]:
            return
        if layer_id in self.submitted_save_layer_ids:
            return
        assert self.kv_send_thread is not None
        self.pending_save_layer_ids.add(layer_id)
        self.submitted_save_layer_ids.add(layer_id)
        self.kv_send_thread.add_request(list(self.layer_save_tasks[layer_id]))

    def save_kv_layer(self, layer_name: str) -> None:
        if self.use_fused_overlap_offload:
            return
        if _is_current_stream_capturing():
            return
        self.save_cpu(self._get_offload_layer_id(layer_name))

    def get_fused_overlap_cpu_kv_inputs(self, layer_name: str):
        layer_id = self._get_offload_layer_id(layer_name)
        return (
            self.k_caches_cpu[layer_id],
            self.v_caches_cpu[layer_id],
            self.cpu_block_table.gpu,
            self.cpu_block_table_req_hashes,
        )

    def save_current_kv_tokens(
        self,
        layer_name: str,
        slot_mapping: torch.Tensor,
        token_to_req: torch.Tensor,
        cum_query_lens: torch.Tensor,
        num_actual_tokens: int,
        num_reqs: int,
        capturing: bool = False,
    ) -> None:
        """Immediately copy current-step main MLA KV tokens from NPU to CPU."""
        if not self.use_fused_overlap_offload or num_actual_tokens <= 0 or num_reqs <= 0:
            return
        if not self.fused_step_requests:
            return

        layer_id = self._get_offload_layer_id(layer_name)
        self.d2h_slot_mapping_cpu[:num_actual_tokens].copy_(
            slot_mapping[:num_actual_tokens], non_blocking=capturing
        )
        self.d2h_token_to_req_cpu[:num_actual_tokens].copy_(
            token_to_req[:num_actual_tokens], non_blocking=capturing
        )
        self.d2h_cum_query_lens_cpu[:num_reqs].copy_(
            cum_query_lens[:num_reqs], non_blocking=capturing
        )
        self.cpu_block_table_host_buffer[:num_reqs].copy_(
            self.cpu_block_table.gpu[:num_reqs], non_blocking=capturing
        )

        args = (num_actual_tokens, num_reqs, layer_id)
        current_compute_stream = torch_npu.npu.current_stream()
        if capturing:
            subscribed_compute_streams = get_subscribed_compute_streams()
            if current_compute_stream not in subscribed_compute_streams:
                torch_npu.npu._subscribe_report(current_compute_stream)
                subscribed_compute_streams.add(current_compute_stream)
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self._compute_step_offload_addrs_cpu,
                args,
            )
        else:
            self._compute_step_offload_addrs_cpu(args)

        if not capturing:
            copy_count = int(self.d2h_num_tokens_buffer_cpu[0].item())
            if not getattr(self, "_fused_overlap_d2h_logged", False):
                logger.info(
                    "[fused_overlap_offload][d2h] first submit layer_id=%s "
                    "num_actual_tokens=%s num_reqs=%s copy_count=%s "
                    "slot_shape=%s token_to_req_shape=%s cum_query_lens_shape=%s",
                    layer_id,
                    num_actual_tokens,
                    num_reqs,
                    copy_count,
                    tuple(slot_mapping[:num_actual_tokens].shape),
                    tuple(token_to_req[:num_actual_tokens].shape),
                    tuple(cum_query_lens[:num_reqs].shape),
                )
                self._fused_overlap_d2h_logged = True
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(
                    "[fused_overlap_offload][d2h][debug] layer_id=%s "
                    "num_actual_tokens=%s num_reqs=%s copy_count=%s",
                    layer_id,
                    num_actual_tokens,
                    num_reqs,
                    copy_count,
                )

        self.d2h_batch_copy_args_buffer_npu.copy_(
            self.d2h_batch_copy_args_buffer_cpu, non_blocking=capturing
        )
        # D2H: gvas/addr buffers use H2D naming (CPU/NPU); swap sparse_copy args for NPU->CPU.
        offload.sparse_copy(
            self.d2h_addr_buffer_npu,
            self.d2h_gvas_buffer_npu,
            self.d2h_size_buffer_npu,
            self.d2h_num_tokens_buffer_npu,
            self.k_caches_npu[layer_id].device,
        )
        self.d2h_save_event.record(current_compute_stream)
        if not capturing:
            self.d2h_save_event.synchronize()

    def wait_for_save(self):
        assert self.use_layerwise
        if not self.pending_save_layer_ids:
            # no save tasks, no need to wait
            return
        for layer_id in sorted(self.pending_save_layer_ids):
            event = self.layer_save_finished_events[layer_id]
            is_finish = event.wait(timeout=1)
            if not is_finish:
                logger.info(f'>>>>> layer {layer_id} wait for save timeout')
            event.clear()
        self.pending_save_layer_ids.clear()
        self.submitted_save_layer_ids.clear()
 
    def set_req_ids(self, req_ids: list):
        self.req_ids = req_ids

    def prepare_lru_resident_and_load_cpu(self, args):
        (
            num_reqs,
            miss_count,
            miss_tokens,
            miss_slots,
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
            layer_id,
            do_offload,
        ) = args
        cpu_sparse_attn.lru_resident_compact(
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            num_reqs,
            self.sfa_sparse_topk,
            self.lru_resident_capacity,
            self.max_model_len,
            self.lru_workspace_threads,
            self.lru_workspace_threads,
        )
        num_tokens_to_load = cpu_sparse_attn.compute_lru_resident_addrs(
            miss_count,
            miss_tokens,
            miss_slots,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            self.lru_resident_capacity,
            self.lru_workspace_threads,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
        )

        if not do_offload and layer_id == 0 and self.tp_rank == 0:
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(f'>>>>> load_kv_token_wise, num_tokens_to_load={num_tokens_to_load}')

        if do_offload:
            # in graph mode, we don't want to interrupt graph twice (since it's time consuming),
            # so we start offload here instead of original maybe_save_kv.
            self.save_cpu(layer_id)

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        topk_indices_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        token_to_req_npu: torch.Tensor | None = None,
        capturing: bool = False,
    ) -> bool:
        capturing = capturing or _is_current_stream_capturing()
        layer_id = self._get_offload_layer_id(layer_name)
        topk = self.sfa_sparse_topk
        capacity = self.lru_resident_capacity
        if topk > self.sfa_sparse_topk or capacity > self.lru_resident_capacity:
            raise ValueError(
                "LRU resident tensors exceed configured workspace, "
                f"topk={topk}, capacity={capacity}, "
                f"configured_topk={self.sfa_sparse_topk}, "
                f"configured_capacity={self.lru_resident_capacity}"
            )
        if num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "SFA offload topk rows exceed configured workspace, "
                f"num_tokens={num_tokens}, max_num_topk_rows={self.max_num_topk_rows}"
            )
        cpu_block_table_reqs = self.cpu_block_table_host_buffer[:num_reqs]
        cpu_block_table_reqs.copy_(self.cpu_block_table.gpu[:num_reqs], non_blocking=capturing)
        if token_to_req_npu is not None:
            token_to_req_cpu = self.lru_token_to_req_cpu[:num_tokens]
            token_to_req_cpu.copy_(token_to_req_npu[:num_tokens], non_blocking=capturing)
            cpu_block_table_expanded = torch.index_select(
                self.cpu_block_table.gpu[:num_reqs], 0, token_to_req_npu[:num_tokens].to(torch.int64))
            cpu_block_table = self.lru_expanded_block_table_cpu[:num_tokens]
            cpu_block_table.copy_(cpu_block_table_expanded, non_blocking=capturing)
        else:
            cpu_block_table = cpu_block_table_reqs
        topk_indices_cpu = self.lru_topk_indices_cpu[:num_tokens]
        topk_indices_cpu.copy_(topk_indices_npu[:num_tokens], non_blocking=capturing)
        req_ids_cpu = self.lru_req_ids_cpu[:num_tokens]
        req_ids_cpu.copy_(req_ids_npu[:num_tokens], non_blocking=capturing)

        args = (
            num_tokens,
            self.lru_miss_count_cpu_list[layer_id][:num_tokens],
            self.lru_miss_tokens_cpu_list[layer_id][:num_tokens],
            self.lru_miss_slots_cpu_list[layer_id][:num_tokens],
            self.lru_req_ids_ptr,
            self.lru_last_req_ids_ptrs[layer_id],
            self.lru_topk_indices_ptr,
            self.lru_slot_to_token_ptrs[layer_id],
            self.lru_slots_ptrs[layer_id],
            self.lru_current_slots_ptr,
            self.lru_miss_count_ptrs[layer_id],
            self.lru_miss_tokens_ptrs[layer_id],
            self.lru_miss_slots_ptrs[layer_id],
            cpu_block_table,
            self.block_size,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
            self.gvas_k_bases[layer_id],
            self.gvas_v_bases[layer_id],
            self.addr_k_bases[layer_id],
            self.addr_v_bases[layer_id],
            self.lru_token_mark_workspace_ptr,
            self.lru_token_pos_workspace_ptr,
            self.lru_slot_workspace_ptr,
            self.lru_miss_position_workspace_ptr,
            self.lru_epochs_ptr,
            self.gvas_buffer_cpu,
            self.addr_buffer_cpu,
            self.size_buffer_cpu,
            self.num_tokens_buffer_cpu,
            layer_id,
            capturing,
        )

        if capturing:
            if self.layer_save_tasks[layer_id]:
                self.pending_save_layer_ids.add(layer_id)
            current_compute_stream = torch_npu.npu.current_stream()
            subscribed_compute_streams = get_subscribed_compute_streams()
            if current_compute_stream not in subscribed_compute_streams:
                torch_npu.npu._subscribe_report(current_compute_stream)
                subscribed_compute_streams.add(current_compute_stream)
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self.prepare_lru_resident_and_load_cpu,
                args,
            )
        else:
            self.prepare_lru_resident_and_load_cpu(args)

        self.sparse_copy_args_buffer_npu.copy_(self.sparse_copy_args_buffer_cpu, non_blocking=capturing)
        offload.sparse_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )

        current_slots_cpu = self.lru_current_slots_cpu[:num_tokens]
        current_slots_npu[:num_tokens].copy_(current_slots_cpu, non_blocking=capturing)
        return True

    def process_layer_data(self, request: ReqMeta) -> Generator[
        Optional[torch.Tensor], None, None]:
        """
        Generate kv offload related metadata.
        """
        num_new_offload_blocks = request.num_new_offload_blocks
        block_ids_npu = request.block_ids_npu
        block_ids_cpu = request.block_ids_cpu
        if len(block_ids_npu) > len(block_ids_cpu):
            # in most cases block_ids_npu has one more unfull block, remove it
            block_ids_npu = block_ids_npu[:-1]
        assert len(block_ids_npu) == len(block_ids_cpu)
        block_ids_npu = block_ids_npu[-num_new_offload_blocks:]
        block_ids_cpu = block_ids_cpu[-num_new_offload_blocks:]

        for layer_id in range(self.num_layers):
            req_meta_save = LayerMultiBlockReqMeta(
                request.req_id,
                layer_id,
                block_ids_npu=block_ids_npu,
                block_ids_cpu=block_ids_cpu,
                cache_npu=(self.k_caches_npu[layer_id], self.v_caches_npu[layer_id]),
                cache_cpu=(self.k_caches_cpu[layer_id], self.v_caches_cpu[layer_id]),
            )
            self.layer_save_tasks[layer_id].append(req_meta_save)
