"""Unit tests for the ``npu_fused_li_manage_c8`` operator.

Two layers:
  1. CPU-runnable meta contract tests, mirroring the C++ schema in
     ``csrc/torch_binding.cpp`` and the void meta in
     ``csrc/torch_binding_meta.cpp``. The in-place operator returns ``None``
     and writes four caller-supplied output buffers.
  2. NPU functional test (skipped without an NPU): builds a deterministic
     bf16 reference case, quantizes query/key with the Hadamard + per-token
     symmetric quantization pattern used by the C8 model path, then checks
     output validity and top-k overlap against the bf16 ``npu_fused_li_manage``
     reference.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.library import Library

def _load_cpp_ext() -> bool:
    """Load the compiled _C_ascend extension, falling back to a direct
    ``torch.ops.load_library`` when the full vllm_ascend stack is unavailable
    (e.g. vllm version mismatch in a minimal single-op test environment)."""
    try:
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
        return bool(torch.ops._C_ascend.npu_fused_li_manage_c8)
    except Exception:
        pass
    try:
        import glob
        import os

        import vllm_ascend

        pkg_dir = os.path.dirname(vllm_ascend.__file__)
        vendor_path = os.path.join(
            pkg_dir, "_cann_ops_custom", "vendors", "custom_transformer"
        )
        if os.path.exists(vendor_path):
            opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
                vendor_path + (":" + opp if opp else "")
            )
            vendor_lib = os.path.join(vendor_path, "op_api", "lib")
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (
                vendor_lib + (":" + ld if ld else "")
            )
        so_candidates = glob.glob(
            os.path.join(pkg_dir, "vllm_ascend_C*.so")
        )
        if not so_candidates:
            return False
        torch.ops.load_library(so_candidates[0])
        return bool(torch.ops._C_ascend.npu_fused_li_manage_c8)
    except Exception:
        return False


try:
    _has_cpp_ext = _load_cpp_ext()
except Exception:
    _has_cpp_ext = False

try:
    import torch_npu  # noqa: F401

    _has_npu = torch.npu.is_available()
except Exception:
    _has_npu = False

BATCH = 1
HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 2048
CACHE_TOKENS = 6144
SEQ_LEN = 262272
BLOCKS_PER_REQ = SEQ_LEN // BLOCK_SIZE

_LIB = Library("_ascend_flimc8_ut", "DEF")

_C8_SCHEMA = (
    "npu_fused_li_manage_c8("
    "Tensor query, Tensor query_scale, Tensor index_weights, "
    "Tensor index_key_cache, Tensor index_key_scale_cache, "
    "Tensor index_block_table, Tensor num_candidate_tokens, "
    "Tensor num_cache_tokens, Tensor req_pool_entries, "
    "Tensor(a!) cache_slots_pool, Tensor(b!) topk_src_ids, "
    "Tensor(c!) topk_dst_slots, Tensor(d!) miss_counts) -> ()"
)


def _register_local_schema() -> None:
    _LIB.define(_C8_SCHEMA)

    def _meta(query, query_scale, index_weights, index_key_cache,
              index_key_scale_cache, index_block_table, num_candidate_tokens,
              num_cache_tokens, req_pool_entries, cache_slots_pool,
              topk_src_ids, topk_dst_slots, miss_counts):
        return None

    _LIB.impl("npu_fused_li_manage_c8", _meta, "Meta")


if not _has_cpp_ext:
    _register_local_schema()


def _op():
    if _has_cpp_ext:
        return torch.ops._C_ascend.npu_fused_li_manage_c8
    return getattr(torch.ops, "_ascend_flimc8_ut").npu_fused_li_manage_c8


@pytest.fixture
def meta_case():
    meta = torch.device("meta")
    pool_size = BATCH + 1
    return {
        "query": torch.empty(BATCH, HEADS, HEAD_DIM, device=meta, dtype=torch.int8),
        "query_scale": torch.empty(BATCH, HEADS, device=meta, dtype=torch.float16),
        "index_weights": torch.empty(BATCH, HEADS, device=meta, dtype=torch.bfloat16),
        "index_key_cache": torch.empty(BATCH * BLOCKS_PER_REQ, BLOCK_SIZE, 1, HEAD_DIM, device=meta, dtype=torch.int8),
        "index_key_scale_cache": torch.empty(BATCH * BLOCKS_PER_REQ, BLOCK_SIZE, 1, 1, device=meta, dtype=torch.float16),
        "index_block_table": torch.empty(BATCH, BLOCKS_PER_REQ, device=meta, dtype=torch.int32),
        "num_candidate_tokens": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "num_cache_tokens": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "req_pool_entries": torch.empty(BATCH, device=meta, dtype=torch.int32),
        "cache_slots_pool": torch.empty(pool_size, SEQ_LEN, device=meta, dtype=torch.int32),
        "topk_src_ids": torch.empty(BATCH, 1, TOPK, device=meta, dtype=torch.int32),
        "topk_dst_slots": torch.empty(BATCH, 1, TOPK, device=meta, dtype=torch.int32),
        "miss_counts": torch.empty(BATCH, device=meta, dtype=torch.int32),
    }


def _call_op(mc):
    return _op().default(
        mc["query"], mc["query_scale"], mc["index_weights"],
        mc["index_key_cache"], mc["index_key_scale_cache"],
        mc["index_block_table"], mc["num_candidate_tokens"],
        mc["num_cache_tokens"], mc["req_pool_entries"],
        mc["cache_slots_pool"], mc["topk_src_ids"],
        mc["topk_dst_slots"], mc["miss_counts"],
    )


def test_meta_returns_none(meta_case):
    result = _call_op(meta_case)
    assert result is None, "FusedLiManageC8 must return None"


def test_meta_preserves_buffer_shapes(meta_case):
    _call_op(meta_case)
    expected = {
        "topk_src_ids": (BATCH, 1, TOPK),
        "topk_dst_slots": (BATCH, 1, TOPK),
        "miss_counts": (BATCH,),
    }
    for name, shape in expected.items():
        assert tuple(meta_case[name].shape) == shape, (
            f"{name} shape changed: {tuple(meta_case[name].shape)} != {shape}"
        )


def test_meta_preserves_buffer_dtypes(meta_case):
    _call_op(meta_case)
    for name in ("topk_src_ids", "topk_dst_slots", "miss_counts"):
        assert meta_case[name].dtype == torch.int32, (
            f"{name} dtype changed: {meta_case[name].dtype} != int32"
        )


# ---------------------------------------------------------------------------
# NPU functional test
# ---------------------------------------------------------------------------


def _apply_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Apply 128x128 Hadamard transform along the last dim."""
    from scipy.linalg import hadamard

    H = torch.tensor(
        hadamard(HEAD_DIM), dtype=x.dtype, device=x.device
    ) / math.sqrt(HEAD_DIM)
    return torch.matmul(x, H)


def _build_case(
    *,
    device: torch.device,
    heads: int,
    batch_size: int,
    seq_len: int,
    cache_tokens_value: int,
    miss_count: int,
    seed: int = 7,
):
    blocks_per_request = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    capacity = blocks_per_request * BLOCK_SIZE
    total_blocks = batch_size * blocks_per_request

    # Random query/key/weights, mirroring the lightning_indexer_quant
    # acceptance cases. Synthetic id-encoded keys are degenerate under int8
    # (quantization error swamps the token-discriminating signal), so use
    # random data where int8 recovers ~0.997 top-k overlap.
    torch.manual_seed(seed)
    query = torch.randn((batch_size, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    weights = torch.randn((batch_size, heads), dtype=torch.bfloat16, device=device).abs() + 0.1
    key = torch.randn(
        (total_blocks, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    # True bf16 scores -> true top-k, used to prefill the cache pool so that
    # the bf16 reference misses exactly miss_count entries by construction.
    key_req0 = key.view(batch_size, capacity, HEAD_DIM)[0]
    true_scores = torch.matmul(query[0].float(), key_req0.float().t())
    true_score = (weights[0].float().unsqueeze(1) * true_scores).sum(0)
    true_topk_desc = true_score.argsort(descending=True)[:TOPK]

    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(
        batch_size, blocks_per_request
    )
    candidate_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    cache_tokens = torch.full(
        (batch_size,), cache_tokens_value, dtype=torch.int32, device=device
    )

    pool_size = batch_size + 1
    req_entries_cpu = torch.arange(batch_size, 0, -1, dtype=torch.int32)
    req_entries = req_entries_cpu.to(device)
    initial_cache_cpu = torch.full((pool_size, capacity), -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(seed)
    hit_tokens = true_topk_desc[miss_count:].cpu()
    other_count = cache_tokens_value - hit_tokens.numel()
    non_topk_mask = torch.ones(seq_len, dtype=torch.bool)
    non_topk_mask[true_topk_desc.cpu()] = False
    non_topk_ids = non_topk_mask.nonzero(as_tuple=True)[0]
    stride = max(non_topk_ids.numel() // other_count, 1) if other_count else 1
    other_tokens = (
        non_topk_ids[::stride][:other_count] if other_count else torch.empty(0, dtype=torch.int64)
    )
    cached_tokens = torch.cat((hit_tokens, other_tokens))
    for row in range(batch_size):
        slots = torch.randperm(cache_tokens_value, generator=generator, dtype=torch.int32)
        initial_cache_cpu[int(req_entries_cpu[row]), cached_tokens] = slots

    initial_cache = initial_cache_cpu.to(device)
    return {
        "capacity": capacity,
        "query": query,
        "key": key,
        "weights": weights,
        "req_entries": req_entries,
        "req_entries_cpu": req_entries_cpu,
        "initial_cache": initial_cache,
        "initial_cache_cpu": initial_cache_cpu,
        "cache_tokens": cache_tokens,
        "candidate_lens": candidate_lens,
        "block_table": block_table,
        "batch_size": batch_size,
    }


def _fresh_outputs(case: dict) -> dict:
    batch_size = case["batch_size"]
    device = case["query"].device
    return {
        "cache_slots": case["initial_cache"].clone(),
        "source_ids": torch.full((batch_size, 1, TOPK), -1, dtype=torch.int32, device=device),
        "destination_slots": torch.full((batch_size, 1, TOPK), -1, dtype=torch.int32, device=device),
        "miss_counts": torch.full((batch_size,), -1, dtype=torch.int32, device=device),
    }


def _run_bf16_reference(case: dict) -> dict:
    out = _fresh_outputs(case)
    torch.ops._C_ascend.npu_fused_li_manage(
        case["query"],
        case["weights"],
        case["key"],
        case["block_table"],
        case["candidate_lens"],
        case["cache_tokens"],
        case["req_entries"],
        out["cache_slots"],
        out["source_ids"],
        out["destination_slots"],
        out["miss_counts"],
    )
    return out


def _run_c8(case: dict) -> dict:
    query_h = _apply_hadamard(case["query"])
    key_h = _apply_hadamard(case["key"])

    query_i8_flat, query_scale_f32 = torch_npu.npu_dynamic_quant(query_h.view(-1, HEAD_DIM))
    query_i8 = query_i8_flat.view_as(query_h)
    key_i8_flat, key_scale_f32 = torch_npu.npu_dynamic_quant(key_h.view(-1, HEAD_DIM))
    key_i8 = key_i8_flat.view_as(key_h)
    key_scale = key_scale_f32.view(
        case["key"].size(0), case["key"].size(1), 1, 1
    ).to(torch.float16)
    query_scale = query_scale_f32.view(
        case["query"].size(0), case["query"].size(1)
    ).to(torch.float16)

    out = _fresh_outputs(case)
    torch.ops._C_ascend.npu_fused_li_manage_c8(
        query_i8,
        query_scale,
        case["weights"],
        key_i8,
        key_scale,
        case["block_table"],
        case["candidate_lens"],
        case["cache_tokens"],
        case["req_entries"],
        out["cache_slots"],
        out["source_ids"],
        out["destination_slots"],
        out["miss_counts"],
    )
    return out


def _check_valid(outputs: dict, case: dict, *, seq_len: int, cache_tokens_value: int) -> None:
    batch_size = case["batch_size"]
    sources = outputs["source_ids"].view(batch_size, TOPK).cpu()
    slots = outputs["destination_slots"].view(batch_size, TOPK).cpu()
    counts = outputs["miss_counts"].cpu()
    pool = outputs["cache_slots"].cpu()

    assert not bool((sources < 0).any()) and not bool((sources >= seq_len).any()), (
        "topk_index contains out-of-range token"
    )
    for row in range(batch_size):
        assert torch.unique(sources[row]).numel() == TOPK, f"row={row} topk_index contains duplicates"
        assert torch.unique(slots[row]).numel() == TOPK, f"row={row} destination slots contain duplicates"
    assert not bool((slots < 0).any()) and not bool((slots >= cache_tokens_value).any()), (
        "destination slot out of range"
    )
    assert not bool((counts < 0).any()) and not bool((counts > TOPK).any()), "miss_counts out of range"

    req_entries_cpu = case["req_entries_cpu"].to(torch.int64)
    old_pool = case["initial_cache_cpu"]
    mapped_rows = set(int(v) for v in req_entries_cpu.tolist())
    for pool_row in range(pool.shape[0]):
        if pool_row not in mapped_rows:
            assert torch.equal(pool[pool_row], old_pool[pool_row]), (
                f"Unmapped request-pool row {pool_row} was modified."
            )

    for row in range(batch_size):
        pool_row = int(req_entries_cpu[row])
        old_state = old_pool[pool_row]
        old_topk_slots = old_state[sources[row]].to(torch.int64)
        actual_miss_mask = old_topk_slots == -1
        actual_miss_count = int(actual_miss_mask.sum())
        assert actual_miss_count == int(counts[row]), (
            f"row={row} recomputed miss count {actual_miss_count} != output {counts[row]}"
        )
        if actual_miss_count:
            assert bool(actual_miss_mask[:actual_miss_count].all()), f"row={row} miss prefix contains a hit"
        assert not bool(actual_miss_mask[actual_miss_count:].any()), f"row={row} hit suffix contains a miss"
        assert torch.equal(slots[row][actual_miss_count:], old_topk_slots[actual_miss_count:]), (
            f"row={row} hit suffix slots mismatch cache pool"
        )


@pytest.mark.skipif(not (_has_cpp_ext and _has_npu), reason="requires NPU and the compiled _C_ascend extension")
def test_c8_topk_overlap_with_bf16_reference():
    device = torch.device("npu:0")
    seq_len = 4096
    cache_tokens_value = 2048
    miss_count = 100

    case = _build_case(
        device=device,
        heads=HEADS,
        batch_size=1,
        seq_len=seq_len,
        cache_tokens_value=cache_tokens_value,
        miss_count=miss_count,
    )

    ref = _run_bf16_reference(case)
    c8 = _run_c8(case)

    _check_valid(ref, case, seq_len=seq_len, cache_tokens_value=cache_tokens_value)
    _check_valid(c8, case, seq_len=seq_len, cache_tokens_value=cache_tokens_value)

    ref_set = set(ref["source_ids"].view(-1).tolist())
    c8_set = set(c8["source_ids"].view(-1).tolist())
    overlap = len(ref_set & c8_set)
    hit_rate = overlap / TOPK
    print(
        f"C8_TOPK_OVERLAP {overlap}/{TOPK} hit_rate={hit_rate:.4f} "
        f"ref_miss_counts={ref['miss_counts'].tolist()} "
        f"c8_miss_counts={c8['miss_counts'].tolist()}"
    )
    assert hit_rate >= 0.95, f"C8 top-k hit rate {hit_rate:.4f} is below 0.95"
