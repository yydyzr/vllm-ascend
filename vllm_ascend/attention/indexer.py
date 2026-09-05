from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class AscendSFAIndexerMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class AscendSFAIndexerBackend(AttentionBackend):
    """Placeholder backend for split SFA indexer cache layers.

    The SFA indexer cache is represented as its own AttentionLayerBase so the
    KV-cache planner can assign an independent physical tensor while sharing
    block ids with the main MLA cache group. The current SFA forward path still
    recomposes the legacy cache tuple before calling the kernel. This backend
    exposes the indexer's own block table and slot mapping for sparse offload,
    where main host and indexer device blocks may have different physical IDs.

    Do not reuse AscendSFAMetadataBuilder here. It inherits vLLM's
    MLACommonMetadataBuilder, whose initializer assumes layer_names[0] points to
    a real MLAAttention object with ``prefill_backend`` in static_forward_context.
    The indexer cache layer points to DeepseekV32IndexerCache instead, which has
    no ``prefill_backend``. Keeping a cache-only builder avoids that false
    attention-layer assumption and builds only the indexer's two mapping views.
    """

    accept_output_buffer: bool = True

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA_INDEXER"

    @staticmethod
    def get_builder_cls():
        return AscendSFAIndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendSFAIndexerMetadataBuilder(AttentionMetadataBuilder[Any]):
    """Cache-only metadata builder for split SFA indexer cache layers."""

    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendSFAIndexerMetadata:
        # Main host blocks and indexer device blocks need not share physical
        # IDs. Sparse-offload attention consumes these explicit indexer views.
        return AscendSFAIndexerMetadata(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.slot_mapping,
        )
