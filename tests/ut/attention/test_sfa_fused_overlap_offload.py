from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention import utils as attention_utils
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


def _make_impl() -> AscendSFAImpl:
    impl = object.__new__(AscendSFAImpl)
    impl.local_num_heads = 2
    impl.sfa_sparse_topk = 4
    impl.block_size = 4
    impl.lru_resident_capacity = 8
    impl.max_num_topk_rows = 4
    impl.scale = 0.5
    impl.selection_kv_block_table = None
    impl.selection_kv_block_status = None
    impl.fused_overlap_last_req_ids = None
    impl._fused_overlap_selection_capacity = None
    impl._fused_overlap_decode_logged = True
    return impl


def test_fused_overlap_decode_uses_cpu_full_kv_and_reused_selection_buffers():
    impl = _make_impl()
    ql_nope = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    q_pe = torch.ones((2, 2, 1), dtype=torch.float32)
    topk = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int64)
    selection_kv = torch.zeros((2, 8, 1, 3), dtype=torch.float32)
    selection_rope = torch.zeros((2, 8, 1, 1), dtype=torch.float32)
    kv_cache = (None, None, None, selection_kv, selection_rope)
    full_kv_cpu = torch.zeros((3, 4, 1, 3), dtype=torch.float32)
    full_rope_cpu = torch.zeros((3, 4, 1, 1), dtype=torch.float32)
    full_block_table = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=2,
        token_to_req=torch.tensor([0, 1], dtype=torch.int32),
        req_ids_tensor=torch.tensor([101, 202], dtype=torch.int64),
    )
    captured = {}

    def fake_fused_op(**kwargs):
        captured.update(kwargs)
        return kwargs["query"] + 10

    with (
        patch.object(impl, "_require_custom_op", return_value=fake_fused_op),
        patch(
            "vllm_ascend.attention.sfa_v1.get_fused_overlap_cpu_kv_inputs",
            return_value=(full_kv_cpu, full_rope_cpu, full_block_table),
        ),
        patch(
            "vllm_ascend.attention.sfa_v1.get_forward_context",
            return_value=SimpleNamespace(capturing=False),
        ),
    ):
        out = impl._execute_fused_overlap_offload_decode(
            ql_nope,
            q_pe,
            kv_cache,
            topk,
            metadata,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
            "model.layers.0.self_attn",
        )

    torch.testing.assert_close(out, ql_nope + 10)
    assert captured["query"].shape == (2, 2, 4)
    assert captured["selection_topk_indices"].shape == (2, 1, 4)
    assert captured["selection_topk_indices"].dtype == torch.int32
    assert captured["selection_kv_cache"].shape == (4, 4, 3)
    assert captured["selection_k_rope"].shape == (4, 4, 1)
    assert captured["selection_kv_block_table"].shape == (2, 1)
    assert captured["selection_kv_block_status"].shape == (2, 1, 5)
    assert captured["full_kv_cache"].shape == (3, 4, 3)
    assert captured["full_k_rope"].shape == (3, 4, 1)
    assert captured["full_kv_block_table"].shape == (2, 2)
    assert captured["selection_topk_block_size"] == 1
    torch.testing.assert_close(
        impl.fused_overlap_last_req_ids[:2],
        torch.tensor([101, 202], dtype=torch.int64),
    )


def test_fused_overlap_selection_invalidation_uses_req_ids_not_row_position():
    impl = _make_impl()
    status = torch.zeros((3, 1, 5), dtype=torch.int32)
    last_req_ids = torch.tensor([10, 20, 30], dtype=torch.int64)
    metadata = SimpleNamespace(
        token_to_req=torch.tensor([0, 1, 0], dtype=torch.int32),
        req_ids_tensor=torch.tensor([10, 99], dtype=torch.int64),
    )

    with patch(
        "vllm_ascend.attention.sfa_v1.get_forward_context",
        return_value=SimpleNamespace(capturing=False),
    ):
        impl._invalidate_fused_overlap_selection_rows(
            status,
            last_req_ids,
            metadata,
            num_tokens=3,
            num_reqs=2,
        )

    torch.testing.assert_close(status[0], torch.zeros((1, 5), dtype=torch.int32))
    torch.testing.assert_close(status[1], torch.full((1, 5), -1, dtype=torch.int32))
    torch.testing.assert_close(status[2], torch.full((1, 5), -1, dtype=torch.int32))
    torch.testing.assert_close(last_req_ids, torch.tensor([10, 99, 10], dtype=torch.int64))


def test_fused_overlap_selection_invalidation_rejects_out_of_range_token_to_req():
    impl = _make_impl()
    metadata = SimpleNamespace(
        token_to_req=torch.tensor([0, 2], dtype=torch.int32),
        req_ids_tensor=torch.tensor([10, 20], dtype=torch.int64),
    )

    with (
        patch(
            "vllm_ascend.attention.sfa_v1.get_forward_context",
            return_value=SimpleNamespace(capturing=False),
        ),
        pytest.raises(RuntimeError, match="token_to_req contains request indices outside"),
    ):
        impl._invalidate_fused_overlap_selection_rows(
            torch.zeros((2, 1, 5), dtype=torch.int32),
            torch.full((2,), -1, dtype=torch.int64),
            metadata,
            num_tokens=2,
            num_reqs=2,
        )


def test_get_fused_overlap_cpu_kv_inputs_delegates_to_v1_connector():
    expected = (
        torch.zeros((1, 4, 1, 3)),
        torch.zeros((1, 4, 1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
    )
    connector = SimpleNamespace(get_fused_overlap_cpu_kv_inputs=lambda layer_name: expected)

    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        result = attention_utils.get_fused_overlap_cpu_kv_inputs("layer.0")

    assert result is expected


def test_get_fused_overlap_cpu_kv_inputs_rejects_connector_without_accessor():
    connector = SimpleNamespace()

    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
        pytest.raises(RuntimeError, match="get_fused_overlap_cpu_kv_inputs connector method"),
    ):
        attention_utils.get_fused_overlap_cpu_kv_inputs("layer.0")
