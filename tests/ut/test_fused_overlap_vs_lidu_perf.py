"""Perf comparison: vllm-ascend fused_overlap vs nano LI_MANAGE+scatter_sfa.

Two OPP packages cannot coexist in one process, so run one approach per
process (writes JSON), then --compare reads both.

A: LI + host-side LRU planner + fused_overlap(copy+sfa in one kernel,
   consuming the host-encoded external plan).
B: LightningIndexer baseline + LI_MANAGE(li+lru fused) +
   scatter_copy_sfa(copy+sfa fused).

GLM-5.2 parameters (kv_lora_rank=512, qk_rope_head_dim=64, index_n_heads=32,
index_head_dim=128, index_topk=2048, num_attention_heads=64).
Both sides use the projected MLA formulation: query_nope dim = kv_lora_rank
= 512, rope dim = 64. The fused_overlap adapter splits the concatenated
query (576 = 512 + 64) internally; the tiling's qk_head_dim check (==512)
is on kv_lora_rank, so GLM-5.2 is fully supported.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
from typing import Callable

import numpy as np
import torch
import torch_npu  # noqa: F401

BLOCK_SIZE = 128
# GLM-5.2 MLA parameters (from config.json):
#   kv_lora_rank=512, qk_nope_head_dim=192, qk_rope_head_dim=64,
#   qk_head_dim=256, v_head_dim=256, num_attention_heads=64,
#   index_n_heads=32, index_head_dim=128, index_topk=2048.
# The fused_overlap adapter splits the query into query_nope (first
# kv_lora_rank dims) and query_rope (last qk_rope_head_dim dims), so the
# tiling's qk_head_dim check (==512) is on kv_lora_rank, not the total
# query dim. Both sides use kv_lora_rank=512, rope=64, matching GLM-5.2.
KVD = 512          # kv_lora_rank — compressed KV cache dim (also projected nope dim)
KRD = 64           # qk_rope_head_dim — rope dim
INDEX_DIM = 128    # index_head_dim
TOPK = 2048        # index_topk
NUM_ATTENTION_HEADS = 64  # GLM-5.2 num_attention_heads (total across TP)
_GIB = 1024 * 1024 * 1024
_ALIGN = 2 * 1024 * 1024
TOPK_LAYOUTS = ("contiguous", "random")


def _default_nano_path() -> str:
    """Resolve the nano torch_extension directory.

    Priority: $NANO_PATH env > sibling-relative to this UT > hardcoded fallback.
    """
    env = os.environ.get("NANO_PATH")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.normpath(
        os.path.join(here, "..", "..", "..",
                     "nanovllm-DSA-offload", "torch_extension"))
    if os.path.isdir(os.path.join(sibling, "ops_overlap")):
        return sibling
    return "/Users/zywszr/Desktop/codes/kv_offload/nanovllm-DSA-offload/torch_extension"


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def build_index_scores_and_topk(
    seq_len: int,
    topk_layout: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU scores and the corresponding sorted physical top-k tokens."""
    if topk_layout == "contiguous":
        score_values = torch.arange(seq_len, dtype=torch.int32)
    else:
        score_values = torch.randperm(
            seq_len,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.int32,
        )
    topk_tokens = torch.argsort(score_values, descending=True)[:TOPK]
    return score_values, torch.sort(topk_tokens).values.to(torch.int32)


def bench_events(runner, warmup, iters, reset=None):
    for _ in range(max(0, warmup)):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        runner()
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        runner()
        e.record()
        e.synchronize()
        times.append(float(s.elapsed_time(e)))
    return statistics.mean(times)


def profile_pipeline(steps, warmup, iters, profile_dir=None):
    """Run the full pipeline under torch_npu.profiler (msprof backend).

    ``steps`` is a list of ``(name, callable)`` executed in order each iter.
    Exports an msprof trace to ``profile_dir`` for offline analysis.
    ``torch_npu.profiler`` (msprof) does not support ``key_averages()``,
    so a trace directory is required.
    """
    import tempfile
    import torch_npu.profiler as prof

    if profile_dir is None:
        profile_dir = tempfile.mkdtemp(prefix="fused_overlap_prof_")
        print(f"[profile] no --profile-dir given, using temp dir: {profile_dir}",
              flush=True)

    for _ in range(max(0, warmup)):
        for _, fn in steps:
            fn()
        torch.npu.synchronize()

    exp = prof._ExperimentalConfig(
        export_type=prof.ExportType.Text,
        profiler_level=prof.ProfilerLevel.Level1,
        aic_metrics=prof.AiCMetrics.PipeUtilization,
        data_simplification=True,
    )
    activities = [prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU]
    trace_cb = prof.tensorboard_trace_handler(profile_dir)
    with prof.profile(activities=activities, experimental_config=exp,
                      on_trace_ready=trace_cb) as p:
        for _ in range(iters):
            for _, fn in steps:
                fn()
            torch.npu.synchronize()

    print(f"PROFILE_TRACE_DIR={profile_dir}", flush=True)
    print("Use msprof or tensorboard to open the trace for kernel-level analysis.",
          flush=True)


# ===================== Approach A: vllm-ascend =====================
def run_a(args) -> dict:
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from memfabric_hybrid import offload
    from vllm_ascend.utils import enable_custom_op
    from vllm_ascend.distributed.kv_transfer.kv_offload_decode import (
        kv_offload_decode_manager,
    )

    assert enable_custom_op()
    torch_npu.npu.set_device(0)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    FUSED_OVERLAP = torch.ops._C_ascend.npu_fused_sparse_attention_overlap

    def _align(t, a):
        p = t.data_ptr()
        ap = (p + a - 1) // a * a
        return t[(ap - p) // t.element_size():]

    def _empty_cpu(shape, dtype):
        n = int(np.prod(shape))
        nb = n * torch.empty((), dtype=dtype).element_size()
        raw = offload.empty([nb + _ALIGN], dtype=torch.int8, pin_memory=True)
        return _align(raw, _ALIGN)[:nb].view(dtype).view(shape)

    cfg = offload.OffloadConfig()
    cfg.device_id = torch_npu.npu.current_device()
    cfg.size = int(args.dram_size_gb * _GIB)
    cfg.world_size = 1
    cfg.rank_id = 0
    offload.initialize(cfg)

    dt = torch.bfloat16
    batch = args.batch_size
    q_heads = args.q_heads
    idx_heads = args.indexer_heads
    seq_len = args.seq_len
    topk_layout = args.topk_layout
    fmbn = cdiv(seq_len, BLOCK_SIZE)
    smbn = cdiv(TOPK, BLOCK_SIZE)
    scale = 1.0 / math.sqrt(KVD)
    torch.manual_seed(910000 + batch + q_heads * 17 + seq_len)

    q = torch.randn(batch, q_heads, KVD, dtype=dt, device="npu")
    qr = torch.randn(batch, q_heads, KRD, dtype=dt, device="npu")
    q_fused = torch.cat([q, qr], dim=-1).contiguous()
    full_kv = _empty_cpu([fmbn, BLOCK_SIZE, KVD], dt)
    full_kv.zero_()
    full_rope = _empty_cpu([fmbn, BLOCK_SIZE, KRD], dt)
    full_rope.zero_()
    full_bt = torch.arange(fmbn, dtype=torch.int32, device="npu").unsqueeze(0).expand(batch, -1).contiguous()
    aq = torch.tensor([1] * batch, dtype=torch.int32, device="npu")
    ak = torch.tensor([seq_len] * batch, dtype=torch.int32, device="npu")

    index_seed = 910000 + batch + q_heads * 17 + seq_len
    index_scores_cpu, topk_tokens_cpu = build_index_scores_and_topk(
        seq_len, topk_layout, index_seed)
    topk = topk_tokens_cpu.to("npu").view(1, 1, 1, TOPK).expand(
        batch, 1, 1, TOPK).contiguous()

    total_sel = smbn * batch
    sel_kv = torch.zeros(total_sel, BLOCK_SIZE, KVD, dtype=dt, device="npu")
    sel_rope = torch.zeros(total_sel, BLOCK_SIZE, KRD, dtype=dt, device="npu")
    sel_bt = torch.arange(total_sel, dtype=torch.int32, device="npu").reshape(batch, smbn)
    sel_status = torch.full((batch, 1, 1, TOPK + 1), -1, dtype=torch.int32, device="npu")

    li_q = torch.zeros((batch, idx_heads, INDEX_DIM), dtype=dt, device="npu")
    li_q[:, 0, 0] = 1
    li_q[:, 0, 1] = 64
    li_q[:, 0, 2] = 4096
    li_w = torch.zeros((batch, idx_heads), dtype=dt, device="npu")
    li_w[:, 0] = 1
    bpr = seq_len // BLOCK_SIZE
    li_key = torch.zeros((batch * bpr, BLOCK_SIZE, 1, INDEX_DIM), dtype=dt, device="npu")
    index_scores = index_scores_cpu.to("npu").view(1, bpr, BLOCK_SIZE)
    kr = li_key.view(batch, bpr, BLOCK_SIZE, 1, INDEX_DIM)
    kr[:, :, :, 0, 0] = (index_scores % 64).to(dt)
    kr[:, :, :, 0, 1] = ((index_scores // 64) % 64).to(dt)
    kr[:, :, :, 0, 2] = (index_scores // 4096).to(dt)
    li_bt = torch.arange(batch * bpr, dtype=torch.int32, device="npu").view(batch, bpr)
    li_ql = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
    li_cl = torch.full((batch,), seq_len, dtype=torch.int32, device="npu")

    # ---- host-side LRU planner (production path) -----------------------
    # The production A path runs the LRU/replacement planner on host
    # (kv_offload_decode.cpp::lru_resident_compact_with_plan_stable_rows)
    # BEFORE the fused kernel. The kernel then consumes the encoded plan
    # written into selection_membership_map (EXTERNAL_PLAN_READY_MARKER).
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    npu_include = os.path.join(ascend_home, "include")
    npu_lib = os.path.join(ascend_home, "lib64")
    if not os.path.exists(npu_lib):
        npu_lib = os.path.join(ascend_home, "lib")
    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib = os.path.join(torch_npu_path, "lib")
    os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
    os.environ["CXX"] = "clang++"
    os.environ["CC"] = "clang"
    cpp_src = os.path.join(
        os.path.dirname(os.path.abspath(kv_offload_decode_manager.__file__)),
        "kv_offload_decode.cpp",
    )
    lru_cpp = torch.utils.cpp_extension.load(
        name="kv_offload_decode",
        sources=[cpp_src],
        extra_cflags=[
            "-O3", "-std=c++20", "-fopenmp",
            "-march=armv8.2-a+sve+fp16+bf16", "-fPIC",
            f"-I{npu_include}", f"-I{torch_npu_include}",
        ],
        extra_ldflags=[
            "-fopenmp", f"-L{npu_lib}", "-lascendcl",
            f"-L{torch_npu_lib}", "-ltorch_npu",
        ],
        verbose=False,
    )

    EXTERNAL_PLAN_READY_MARKER = 0x5A45
    PAIRED_SELECTION_COPY_MARKER = 0x5A56
    MEMBERSHIP_MAP_INT16 = 16376
    MEMBERSHIP_ALIGN_INT16 = 16
    MEMBERSHIP_CONTROL_INT16 = 8
    CONTROL_OFFSET_INT16 = (
        MEMBERSHIP_MAP_INT16 + MEMBERSHIP_ALIGN_INT16
    ) // MEMBERSHIP_ALIGN_INT16 * MEMBERSHIP_ALIGN_INT16  # 16384
    STORAGE_INT16 = (
        CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16 + MEMBERSHIP_ALIGN_INT16
    ) // MEMBERSHIP_ALIGN_INT16 * MEMBERSHIP_ALIGN_INT16  # 16392

    topk_buffer_size = TOPK * 2
    max_rows = batch
    threads = 8
    plan_start = CONTROL_OFFSET_INT16 - TOPK
    required_columns = CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16

    membership_map = offload.empty(
        [max_rows * STORAGE_INT16], dtype=torch.int16, pin_memory=True,
    ).view([max_rows, STORAGE_INT16])
    membership_map.fill_(-1)
    control = membership_map[
        :, CONTROL_OFFSET_INT16:CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16]
    control[:, 1] = EXTERNAL_PLAN_READY_MARKER
    control[:, 2] = TOPK
    control[:, 3] = CONTROL_OFFSET_INT16 - TOPK
    control[:, 7] = PAIRED_SELECTION_COPY_MARKER
    plan_storage = membership_map[:max_rows, plan_start:required_columns]
    encoded_plan_stride = membership_map.stride(0)

    lru_req_ids = torch.empty([max_rows], dtype=torch.int64, device="cpu", pin_memory=True)
    lru_last_req_ids = torch.full([max_rows], -1, dtype=torch.int64, device="cpu", pin_memory=True)
    lru_topk_indices = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_stable_prefix_lens = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_visible_seq_lens = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slot_to_token = torch.full([max_rows, topk_buffer_size], -1, dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slots = torch.arange(topk_buffer_size, dtype=torch.int32, device="cpu").view(1, -1).repeat(max_rows, 1).pin_memory()
    lru_current_slots = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_count = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_tokens = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_slots = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_token_mark = torch.zeros([threads, seq_len], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_token_pos = torch.full([threads, seq_len], -1, dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slot_ws = torch.empty([threads, topk_buffer_size * 3], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_pos_ws = torch.empty([threads, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_epochs = torch.zeros([threads], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_physical_row_ws = torch.empty([max_rows * 3], dtype=torch.int32, device="cpu", pin_memory=True)

    lru_cpp.warmup_lru_resident_threads(threads)

    topk_cpu = torch.empty([batch, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    req_ids_cpu = torch.arange(batch, dtype=torch.int64, device="cpu")

    def _run_planner_host(_args):
        lru_cpp.lru_resident_compact_with_plan_stable_rows(
            lru_req_ids.data_ptr(),
            lru_last_req_ids.data_ptr(),
            lru_topk_indices.data_ptr(),
            lru_stable_prefix_lens.data_ptr(),
            lru_slot_to_token.data_ptr(),
            lru_slots.data_ptr(),
            lru_current_slots.data_ptr(),
            lru_miss_count.data_ptr(),
            lru_miss_tokens.data_ptr(),
            lru_miss_slots.data_ptr(),
            lru_token_mark.data_ptr(),
            lru_token_pos.data_ptr(),
            lru_slot_ws.data_ptr(),
            lru_miss_pos_ws.data_ptr(),
            lru_epochs.data_ptr(),
            lru_physical_row_ws.data_ptr(),
            max_rows,
            plan_storage.data_ptr(),
            encoded_plan_stride,
            batch,
            TOPK,
            topk_buffer_size,
            seq_len,
            threads,
            threads,
            lru_visible_seq_lens.data_ptr(),
        )

    def _a_planner_sync():
        """Synchronous planner — used as reset for isolated fused timing."""
        topk_cpu.copy_(topk.reshape(batch, TOPK).to(torch.int32))
        lru_req_ids.copy_(req_ids_cpu)
        lru_topk_indices.copy_(topk_cpu)
        lru_stable_prefix_lens.fill_(0)
        lru_visible_seq_lens.fill_(seq_len)
        _run_planner_host(None)

    subscribed_compute_streams = set()

    def _a_planner():
        """Launch planner via launch_host on the NPU stream (production path).

        D2H copies execute on the NPU stream, then the planner runs as a
        host callback (aclrtLaunchHostFunc).  The fused op on the same
        stream will wait for the callback to finish before executing.
        """
        topk_cpu.copy_(topk.reshape(batch, TOPK).to(torch.int32), non_blocking=True)
        lru_req_ids.copy_(req_ids_cpu, non_blocking=True)
        lru_topk_indices.copy_(topk_cpu, non_blocking=True)
        lru_stable_prefix_lens.fill_(0)
        lru_visible_seq_lens.fill_(seq_len)
        current_compute_stream = torch_npu.npu.current_stream()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)
        torch_npu.npu._launch_host_func(
            current_compute_stream,
            _run_planner_host,
            None,
        )

    def _a_li():
        o = torch_npu.npu_lightning_indexer(
            query=li_q, key=li_key, weights=li_w,
            actual_seq_lengths_query=li_ql, actual_seq_lengths_key=li_cl,
            block_table=li_bt, layout_query="TND", layout_key="PA_BSND",
            sparse_count=TOPK, sparse_mode=3)
        return o[0] if isinstance(o, (tuple, list)) else o

    def _a_fused():
        return FUSED_OVERLAP(
            query=q_fused,
            selection_k_rope=sel_rope,
            selection_kv_cache=sel_kv,
            selection_kv_block_table=sel_bt,
            selection_kv_block_status=sel_status,
            selection_membership_map=membership_map,
            selection_topk_indices=topk,
            full_k_rope=full_rope,
            full_kv_cache=full_kv,
            full_kv_block_table=full_bt,
            full_kv_actual_seq=ak,
            full_q_actual_seq=aq,
            scale_value=scale,
            sparse_block_size=1,
            selection_topk_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3)

    # warmup: planner then fused (production order)
    _a_planner_sync()
    _a_fused()
    torch.npu.synchronize()
    if getattr(args, "profile", False):
        profile_pipeline(
            [("li", _a_li), ("planner", _a_planner), ("fused_overlap", _a_fused)],
            args.warmup, args.profile_iters, args.profile_dir)
        try:
            offload.uninitialize()
        except Exception as exc:
            print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)
        return {
            "approach": "A", "profile": True,
            "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
            "seq_len": seq_len, "topk_layout": topk_layout,
            "hit_ratio": args.hit_ratio,
            "warmup": args.warmup, "iters": args.profile_iters,
        }
    li_ms = bench_events(_a_li, args.warmup, args.iters)
    # pipeline: launch_host planner + fused (same stream, fused waits for planner)
    def _a_pipeline():
        _a_planner()
        _a_fused()
    pipeline_ms = bench_events(_a_pipeline, args.warmup, args.iters)
    # isolated fused (reset with sync planner to ensure membership map is ready)
    fused_ms = bench_events(_a_fused, args.warmup, args.iters,
                           reset=_a_planner_sync)
    # planner cost = pipeline - fused (launch_host overhead included)
    planner_ms = max(0.0, pipeline_ms - fused_ms)
    try:
        offload.uninitialize()
    except Exception as exc:
        print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)

    return {
        "approach": "A",
        "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
        "seq_len": seq_len, "topk_layout": topk_layout,
        "hit_ratio": args.hit_ratio,
        "li_ms": li_ms, "planner_ms": planner_ms, "fused_overlap_ms": fused_ms,
        "pipeline_ms": pipeline_ms,
        "total_ms": li_ms + pipeline_ms,
        "warmup": args.warmup, "iters": args.iters,
    }


# ===================== Approach B: nano =====================
def run_b(args) -> dict:
    if args.nano_path not in sys.path:
        sys.path.insert(0, args.nano_path)

    np_dir = args.nano_path
    pkg_dir = os.path.join(np_dir, "ops_overlap")
    opapi = os.path.join(pkg_dir, "..", "..", "_custom_opp", "vendors",
                         "ops-overlap", "op_api", "lib", "libcust_opapi.so")
    so_files = [f for f in os.listdir(pkg_dir)
                if f.startswith("_C") and f.endswith(".so")] if os.path.isdir(pkg_dir) else []
    print(f"[B-diag] nano_path={np_dir!r}", flush=True)
    print(f"[B-diag] ops_overlap/ exists: {os.path.isdir(pkg_dir)}", flush=True)
    print(f"[B-diag] ops_overlap/__init__.py exists: "
          f"{os.path.isfile(os.path.join(pkg_dir, '__init__.py'))}", flush=True)
    print(f"[B-diag] _C*.so files: {so_files}", flush=True)
    print(f"[B-diag] libcust_opapi.so exists: "
          f"{os.path.isfile(opapi)} ({opapi})", flush=True)
    print(f"[B-diag] sys.path[0:3]={sys.path[:3]}", flush=True)

    try:
        import ops_overlap  # noqa: F401  sets ASCEND_CUSTOM_OPP_PATH
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise ImportError(
            f"Cannot import ops_overlap from --nano-path={np_dir!r}. "
            f"Ensure nano is built (bash build.sh in nanovllm-DSA-offload) and "
            f"the path contains ops_overlap/. Original error: {exc}"
        ) from exc

    # verify the required ops are actually registered (stale _C.so check)
    _required_ops = [
        "li_manage_out",
        "sparse_and_tail_attention_and_scatter_copy",
    ]
    _missing = [op for op in _required_ops
                if not hasattr(torch.ops.ops_overlap, op)]
    if _missing:
        raise RuntimeError(
            f"ops_overlap._C.so is stale — missing ops: {_missing}. "
            f"Rebuild nano: cd <nano-root> && rm -rf _custom_opp "
            f"torch_extension/build torch_extension/ops_overlap/_C*.so && bash build.sh"
        )

    torch_npu.npu.set_device(0)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})

    def _swap(cpu, dev):
        t = torch_npu.empty_with_swapped_memory(cpu.shape, dtype=cpu.dtype, device=dev)
        t.fill_(0)
        t.add_(cpu.to(dev))
        return t

    dev = torch.device("npu:0")
    dt = torch.bfloat16
    batch = args.batch_size
    q_heads = args.q_heads
    idx_heads = args.indexer_heads
    seq_len = args.seq_len
    cache_tokens = args.cache_tokens
    miss = args.miss
    topk_layout = args.topk_layout
    if idx_heads not in (32, 64):
        raise RuntimeError(
            f"li_manage_out requires indexer_heads in (32, 64), got {idx_heads}. "
            f"Use --indexer-heads 32 or --indexer-heads 64.")
    print(f"[B-cfg] batch={batch} q_heads={q_heads} idx_heads={idx_heads} "
          f"seq_len={seq_len} topk_layout={topk_layout} "
          f"cache_tokens={cache_tokens} miss={miss}",
          flush=True)
    torch.manual_seed(910000 + batch + q_heads * 17 + seq_len + 1)
    cb = math.ceil(cache_tokens / BLOCK_SIZE)
    sb = seq_len // BLOCK_SIZE

    hbm_bt = torch.empty((batch, cb), dtype=torch.int32)
    for r in range(batch):
        ph = torch.arange(r * cb, (r + 1) * cb, dtype=torch.int32)
        hbm_bt[r] = ph[torch.randperm(cb)]
    dram_bt = torch.stack([torch.randperm(sb, dtype=torch.int64).to(torch.int32) for _ in range(batch)])

    thb = batch * cb
    hbm_kpe = torch.randn((thb, BLOCK_SIZE, 1, KRD), dtype=dt, device=dev)
    hbm_ckv = torch.randn((thb, BLOCK_SIZE, 1, KVD), dtype=dt, device=dev)
    query = torch.randn((batch, q_heads, KVD), dtype=dt, device=dev)
    qrope = torch.randn((batch, q_heads, KRD), dtype=dt, device=dev)

    # ---- deterministic index key and selected top-k token set ----
    # A base-64 encoding preserves the score in the matching query below.
    tib = batch * sb
    index_seed = 910000 + batch + q_heads * 17 + seq_len
    index_scores_cpu, topk_tokens = build_index_scores_and_topk(
        seq_len, topk_layout, index_seed)
    hit_count = TOPK - miss
    hit_tokens = topk_tokens[:hit_count]
    idx_cache_cpu = torch.zeros((tib, BLOCK_SIZE, 1, INDEX_DIM), dtype=dt)
    index_scores = index_scores_cpu.view(1, sb, BLOCK_SIZE)
    idx_cache_rows = idx_cache_cpu.view(batch, sb, BLOCK_SIZE, 1, INDEX_DIM)
    idx_cache_rows[:, :, :, 0, 0] = (index_scores % 64).to(dt)
    idx_cache_rows[:, :, :, 0, 1] = ((index_scores // 64) % 64).to(dt)
    idx_cache_rows[:, :, :, 0, 2] = (index_scores // 4096).to(dt)
    idx_cache = idx_cache_cpu.to(dev)
    idx_bt = torch.arange(tib, dtype=torch.int32, device=dev).view(batch, sb)

    li_query = torch.zeros((batch, idx_heads, INDEX_DIM), dtype=dt, device=dev)
    li_query[:, 0, 0] = 1
    li_query[:, 0, 1] = 64
    li_query[:, 0, 2] = 4096

    g = torch.Generator().manual_seed(42)
    dk = torch.randn((sb, BLOCK_SIZE, KRD), generator=g, dtype=torch.float32).to(dt)
    dc = torch.randn((sb, BLOCK_SIZE, KVD), generator=g, dtype=torch.float32).to(dt)
    dram_kpe = _swap(dk, dev)
    dram_ckv = _swap(dc, dev)
    torch.npu.synchronize()

    # ---- pre-fill cache_slots: hits + other tokens (match nano test) ----
    pool = batch + 7
    pg = torch.Generator().manual_seed(99)
    req_e = torch.randperm(pool, generator=pg)[:batch].to(torch.int32)
    cs_cpu = torch.full((pool, seq_len), -1, dtype=torch.int32)
    non_topk_mask = torch.ones(seq_len, dtype=torch.bool)
    non_topk_mask[topk_tokens] = False
    non_topk = torch.arange(seq_len, dtype=torch.int32)[non_topk_mask]
    for r in range(batch):
        pr = int(req_e[r])
        gen = torch.Generator().manual_seed(r)
        other_count = cache_tokens - hit_count
        others = non_topk[
            torch.randperm(seq_len - TOPK, generator=gen)[:other_count]
        ]
        cached = torch.cat((hit_tokens, others))
        slots = torch.randperm(cache_tokens, generator=gen, dtype=torch.int32)
        cs_cpu[pr, cached] = slots
    cache_slots = cs_cpu.to(dev)
    cache_slots_init = cache_slots.clone()

    weights = torch.zeros((batch, idx_heads), dtype=dt, device=dev)
    weights[:, 0] = 1
    cache_tokens_t = torch.full((batch,), cache_tokens, dtype=torch.int32, device=dev)
    candidate_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)
    # src_ids and miss_counts are OUTPUTS of li_manage_out, pre-allocate with -1
    src_ids_t = torch.full((batch, TOPK), -1, dtype=torch.int32, device=dev)
    li_src_ids = src_ids_t.unsqueeze(1)
    li_dst_slots = torch.empty_like(li_src_ids)
    miss_counts = torch.empty((batch,), dtype=torch.int32, device=dev)
    # sparse_slots: physical slot positions for top-k tokens, extracted
    # from cache_slots AFTER li_manage updates it with miss destinations
    topk_tokens_dev = topk_tokens.to(dev)
    req_e_dev = req_e.to(dev)
    sparse_slots = torch.empty((batch, 1, TOPK), dtype=torch.int32, device=dev)
    actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    actual_kv = torch.full((batch,), cache_tokens, dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(KVD + KRD)

    def _b_reset():
        cache_slots.copy_(cache_slots_init)
        src_ids_t.fill_(-1)

    def _b_extract_slots():
        for r in range(batch):
            pr = int(req_e[r])
            sparse_slots[r, 0] = cache_slots[pr, topk_tokens_dev]

    def _b_li_manage():
        return torch.ops.ops_overlap.li_manage_out.default(
            li_query, idx_cache, weights, req_e_dev, cache_slots,
            cache_tokens_t, candidate_lens, idx_bt,
            li_src_ids, li_dst_slots, miss_counts)

    def _b_lightning_indexer():
        return torch_npu.npu_lightning_indexer(
            query=li_query,
            key=idx_cache,
            weights=weights,
            actual_seq_lengths_query=actual_q,
            actual_seq_lengths_key=candidate_lens,
            block_table=idx_bt,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
        )

    def _b_scatter_sfa():
        _b_extract_slots()
        return torch.ops.ops_overlap.sparse_and_tail_attention_and_scatter_copy.default(
            query, hbm_ckv, sparse_slots, cache_tokens_t, hbm_bt.to(dev),
            actual_q, actual_kv, qrope, hbm_kpe, dram_kpe, dram_ckv,
            dram_bt.to(dev), src_ids_t, miss_counts, scale)

    _b_lightning_indexer()
    _b_li_manage()
    torch.npu.synchronize()
    actual_miss = miss_counts.cpu().tolist()
    expected_miss = [miss] * batch
    if actual_miss != expected_miss:
        print(f"[B-WARN] miss_counts={actual_miss} expected={expected_miss}, "
              f"check idx_cache/cache_slots construction", flush=True)
    _b_scatter_sfa()
    torch.npu.synchronize()
    if getattr(args, "profile", False):
        profile_pipeline(
            [
                ("lightning_indexer", _b_lightning_indexer),
                ("li_manage", _b_li_manage),
                ("scatter_sfa", _b_scatter_sfa),
            ],
            args.warmup, args.profile_iters, args.profile_dir)
        return {
            "approach": "B", "profile": True,
            "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
            "seq_len": seq_len, "topk_layout": topk_layout,
            "cache_tokens": cache_tokens, "miss": miss,
            "warmup": args.warmup, "iters": args.profile_iters,
        }
    lightning_indexer_ms = bench_events(
        _b_lightning_indexer, args.warmup, args.iters)
    li_manage_ms = bench_events(
        _b_li_manage, args.warmup, args.iters, reset=_b_reset)

    def reset_b():
        _b_reset()
        _b_li_manage()
        torch.npu.synchronize()

    sfa_ms = bench_events(_b_scatter_sfa, args.warmup, args.iters, reset=reset_b)

    return {
        "approach": "B",
        "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
        "seq_len": seq_len, "topk_layout": topk_layout,
        "cache_tokens": cache_tokens, "miss": miss,
        "lightning_indexer_ms": lightning_indexer_ms,
        "li_manage_ms": li_manage_ms, "scatter_sfa_ms": sfa_ms,
        "total_ms": li_manage_ms + sfa_ms, "warmup": args.warmup, "iters": args.iters,
    }


# ===================== Compare =====================
def compare(args) -> None:
    with open(args.out_a) as f:
        ra = json.load(f)
    with open(args.out_b) as f:
        rb = json.load(f)
    print("----- TIMING SUMMARY (ms, mean over iters) -----", flush=True)
    if ra["topk_layout"] != rb["topk_layout"]:
        raise ValueError(
            "Cannot compare runs with different top-k layouts: "
            f"A={ra['topk_layout']}, B={rb['topk_layout']}.")
    print(f"Top-k layout: {ra['topk_layout']}", flush=True)
    print(f"Approach A (vllm-ascend):  LI={ra['li_ms']:.4f}  "
          f"planner(launch_host)={ra.get('planner_ms', 0.0):.4f}  "
          f"fused_overlap={ra['fused_overlap_ms']:.4f}  "
          f"pipeline={ra.get('pipeline_ms', 0.0):.4f}  "
          f"total={ra['total_ms']:.4f}",
          flush=True)
    print(f"Approach B (nano):        LI={rb['lightning_indexer_ms']:.4f}  "
          f"LI_MANAGE={rb['li_manage_ms']:.4f}  "
          f"scatter_sfa={rb['scatter_sfa_ms']:.4f}  total={rb['total_ms']:.4f}",
          flush=True)
    diff = ra["total_ms"] - rb["total_ms"]
    who = "A faster" if diff < 0 else "B faster"
    print(f"Delta (A-B): {diff:+.4f} ms  ({who} by {abs(diff):.4f} ms)", flush=True)
    print("UT_OK", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--approach", choices=("A", "B"), default=None,
                   help="Run one approach in this process (writes --out).")
    p.add_argument("--compare", action="store_true",
                   help="Compare --out-a and --out-b JSON results.")
    p.add_argument("--out", default=None, help="Output JSON path for --approach.")
    p.add_argument("--out-a", default="result_a.json")
    p.add_argument("--out-b", default="result_b.json")
    p.add_argument("--nano-path", default=_default_nano_path(),
                   help="Path to nano torch_extension dir (contains ops_overlap/). "
                        "Override via $NANO_PATH env or --nano-path.")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--tp", type=int, default=16,
                   help="Tensor parallel size. q_heads = NUM_ATTENTION_HEADS / tp "
                        "(GLM-5.2: 64 / 16 = 4).")
    p.add_argument("--q-heads", type=int, default=None,
                   help="Attention heads per rank. Overrides --tp derivation. "
                        "Default: 64 / tp.")
    p.add_argument("--indexer-heads", type=int, default=32,
                   help="Indexer heads (GLM-5.2 index_n_heads=32).")
    p.add_argument("--seq-len", type=int, default=65536)
    p.add_argument("--cache-tokens", type=int, default=8192)
    p.add_argument(
        "--topk-layout",
        choices=TOPK_LAYOUTS,
        default="contiguous",
        help=(
            "Physical top-k distribution: contiguous selects the final TOPK "
            "tokens; random scatters them with a deterministic permutation."
        ),
    )
    p.add_argument("--miss", type=int, default=300, help="Miss count per row (approach B).")
    p.add_argument("--hit-ratio", type=float, default=0.85, help="Approach A hit ratio.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dram-size-gb", type=float, default=2.0)
    p.add_argument("--profile", action="store_true",
                   help="Profile the full pipeline with torch_npu.profiler (msprof) "
                        "and print a kernel-level op table instead of Event timing.")
    p.add_argument("--profile-iters", type=int, default=5,
                   help="Iterations to capture under --profile.")
    p.add_argument("--profile-dir", default=None,
                   help="Directory to export the full msprof trace to (optional).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.q_heads is None:
        if NUM_ATTENTION_HEADS % args.tp != 0:
            raise SystemExit(
                f"NUM_ATTENTION_HEADS={NUM_ATTENTION_HEADS} not divisible by "
                f"--tp={args.tp}. Use --q-heads to override.")
        args.q_heads = NUM_ATTENTION_HEADS // args.tp
        print(f"[cfg] tp={args.tp}  q_heads={args.q_heads} "
              f"(={NUM_ATTENTION_HEADS}/{args.tp})", flush=True)
    if args.compare:
        compare(args)
        return
    if args.approach is None:
        raise SystemExit("Use --approach A|B or --compare. See --help.")
    if args.approach == "A":
        result = run_a(args)
    else:
        result = run_b(args)
    out = args.out or f"result_{args.approach.lower()}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    if result.get("profile"):
        print(f"RESULT approach={args.approach} out={out} profile=True",
              flush=True)
    else:
        print(f"RESULT approach={args.approach} out={out} total_ms={result['total_ms']:.4f}",
              flush=True)
    print("UT_OK", flush=True)


if __name__ == "__main__":
    main()
