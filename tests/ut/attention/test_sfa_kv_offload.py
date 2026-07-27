"""Regression tests for SFA KV-offload attention metadata."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.attention.attention_v1 import AscendAttentionState  # noqa: E402
from vllm_ascend.attention.sfa_kv_offload import (  # noqa: E402
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder  # noqa: E402


def _make_boundary_decode_metadata():
    return SimpleNamespace(
        prefill_context_parallel_metadata=None,
        max_query_len=1,
        num_reqs=1,
        num_actual_tokens=1,
        query_start_loc_cpu=torch.tensor([0, 1]),
        is_prefilling=torch.tensor([True]),
        req_ids_tensor=torch.tensor([7]),
        token_to_req=torch.tensor([0]),
    )


@pytest.mark.parametrize(
    ("kv_transfer_config", "expected"),
    [
        (None, False),
        (SimpleNamespace(is_kv_consumer=False, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=False), True),
    ],
)
def test_pd_decode_consumer_is_derived_from_kv_role(kv_transfer_config, expected):
    vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    with patch.object(AscendSFAMetadataBuilder, "__init__", return_value=None):
        builder = AscendSFAKVOffloadMetadataBuilder(
            kv_cache_spec=None,
            layer_names=[],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

    assert builder.is_pd_decode_consumer is expected


@pytest.mark.parametrize(
    ("is_pd_decode_consumer", "expected_decodes", "expected_prefills"),
    [
        (True, 1, 0),
        (False, 0, 1),
    ],
)
def test_boundary_token_classification_depends_on_pd_decode_role(
    is_pd_decode_consumer,
    expected_decodes,
    expected_prefills,
):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = is_pd_decode_consumer
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, _make_boundary_decode_metadata())

    assert metadata.num_decodes == expected_decodes
    assert metadata.num_prefills == expected_prefills
    assert metadata.num_decode_tokens == expected_decodes
    assert metadata.req_ids_tensor.tolist() == [7]
    assert metadata.token_to_req.tolist() == [0]
    assert AscendSFAKVOffloadImpl._is_decode_only(metadata) is is_pd_decode_consumer


def test_pd_decode_consumer_still_rejects_long_prefill_classification():
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = True
    metadata = SimpleNamespace()
    common_metadata = _make_boundary_decode_metadata()
    common_metadata.max_query_len = 2
    common_metadata.num_actual_tokens = 2
    common_metadata.query_start_loc_cpu = torch.tensor([0, 2])

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, common_metadata)

    assert metadata.num_decodes == 0
    assert metadata.num_prefills == 1
    assert metadata.num_decode_tokens == 0


def _make_hit_ratio_impl() -> AscendSFAKVOffloadImpl:
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl._fused_overlap_hit_total_hits = 0
    impl._fused_overlap_hit_total_valid = 0
    impl._fused_overlap_hit_num_steps = 0
    return impl


def test_fused_overlap_selection_hit_stats_accumulate_across_layers():
    """Global hit totals must grow across layer impls in one forward."""
    AscendSFAKVOffloadImpl._reset_fused_overlap_selection_hit_stats()
    layer0 = _make_hit_ratio_impl()
    layer1 = _make_hit_ratio_impl()

    # status slots already hold token ids 0..3; topk asks for 0,1,2,9 -> 3 hits.
    status0 = torch.tensor([[[-1, 0, 1, 2, 3]]], dtype=torch.int32)
    topk0 = torch.tensor([[[0, 1, 2, 9]]], dtype=torch.int32)
    # second layer: all misses.
    status1 = torch.tensor([[[-1, 10, 11, 12, 13]]], dtype=torch.int32)
    topk1 = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32)

    with (
        patch("vllm_ascend.attention.sfa_kv_offload.envs_ascend.VLLM_ASCEND_SFA_DEBUG", True),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
            return_value=SimpleNamespace(capturing=False),
        ),
    ):
        layer0._maybe_log_fused_overlap_selection_hit_ratio(
            layer_name="model.layers.0.self_attn.attn",
            selection_kv_block_status=status0,
            selection_topk_indices=topk0,
        )
        assert AscendSFAKVOffloadImpl._fused_overlap_hit_global_calls == 1
        assert AscendSFAKVOffloadImpl._fused_overlap_hit_global_valid == 4
        assert AscendSFAKVOffloadImpl._fused_overlap_hit_global_hits == 3
        assert layer0._fused_overlap_hit_num_steps == 1
        assert layer0._fused_overlap_hit_total_hits == 3

        layer1._maybe_log_fused_overlap_selection_hit_ratio(
            layer_name="model.layers.1.self_attn.attn",
            selection_kv_block_status=status1,
            selection_topk_indices=topk1,
        )

    assert AscendSFAKVOffloadImpl._fused_overlap_hit_global_calls == 2
    assert AscendSFAKVOffloadImpl._fused_overlap_hit_global_valid == 8
    assert AscendSFAKVOffloadImpl._fused_overlap_hit_global_hits == 3
    assert layer1._fused_overlap_hit_num_steps == 1
    assert layer1._fused_overlap_hit_total_hits == 0
    assert layer1._fused_overlap_hit_total_valid == 4
