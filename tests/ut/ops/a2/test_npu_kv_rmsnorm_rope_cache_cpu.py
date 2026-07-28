# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Probe whether ``torch_npu.npu_kv_rmsnorm_rope_cache`` can scatter the
fused RMSNorm + RoPE result directly into a CPU-resident paged KV cache.

Background: the sparse KV offload path currently computes K/V on NPU and then
performs an explicit D2H ``sparse_copy`` into the CPU pool. If the fused op
accepted CPU output tensors for ``k_cache`` / ``ckv_cache``, we could skip the
extra copy and let the op write the cache rows straight to host memory.

Run directly on an NPU box:

    python tests/ut/ops/a2/test_npu_kv_rmsnorm_rope_cache_cpu.py
"""

import sys
import traceback

import torch
import torch_npu
from memfabric_hybrid import offload


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# DeepSeek-V2 style MLA dims used by the SFA offload path.
#
# Op argument mapping (see sfa_v1.py::exec_kv):
#   npu_kv_rmsnorm_rope_cache(..., k_cache, ckv_cache, ...)
#     - k_cache  stores the rope part  (k_pe)  -> qk_rope_head_dim
#     - ckv_cache stores the rms part  (k_nope) -> kv_lora_rank
# In the offload manager (sparse_kv_offload_manager.py) the CPU pools are
# named the other way round: k_caches_cpu holds k_nope (kv_lora_rank) and
# v_caches_cpu holds k_pe (qk_rope_head_dim). We keep both names below to
# stay faithful to the production allocation.
KV_LORA_RANK = 512        # rms_size  -> ckv_cache (k_nope) / manager.k_caches_cpu
QK_ROPE_HEAD_DIM = 64     # rope_size -> k_cache   (k_pe)   / manager.v_caches_cpu
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
NUM_BLOCKS = 16
DTYPE = torch.bfloat16
EPSILON = 1e-5
BATCH_TOKENS = 8

# Mirrors _CPU_CACHE_ALIGNMENT in sparse_kv_offload_manager.py: the offload
# CPU pools are 2 MiB aligned so that per-layer delta-addrs are stable for the
# sparse_copy descriptor path.
_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024


def _real_npu_available() -> bool:
    # Under the vllm-ascend UT harness torch_npu is mocked and
    # torch.version.cann is set to None when no NPU is present, so a non-None
    # CANN build id is a reliable real-hardware signal.
    return torch.version.cann is not None and torch.npu.is_available()


def _make_inputs(batch_tokens: int, device: torch.device):
    """Build kv / gamma / cos / sin / index on *device* for *batch_tokens* tokens."""
    kv = torch.randn(
        batch_tokens, NUM_KV_HEADS, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=DTYPE, device=device,
    )
    gamma = torch.ones(KV_LORA_RANK, dtype=DTYPE, device=device)
    cos = torch.randn(batch_tokens, 1, 1, QK_ROPE_HEAD_DIM, dtype=DTYPE, device=device)
    sin = torch.randn(batch_tokens, 1, 1, QK_ROPE_HEAD_DIM, dtype=DTYPE, device=device)
    # PA cache_mode expects a flat index of shape [batch_tokens].
    index = torch.arange(batch_tokens, dtype=torch.int64, device=device)
    return kv, gamma, cos, sin, index


def _run_op(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, is_output_kv):
    return torch_npu.npu_kv_rmsnorm_rope_cache(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        ckv_cache,
        epsilon=EPSILON,
        cache_mode="PA",
        is_output_kv=is_output_kv,
    )


def _reference_npu(batch_tokens: int):
    """Run the op with NPU-resident caches; return (k_rows, ckv_rows) on host."""
    npu = torch.device("npu")
    kv, gamma, cos, sin, index = _make_inputs(batch_tokens, npu)
    k_cache_ref = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, QK_ROPE_HEAD_DIM, dtype=DTYPE, device=npu,
    )
    ckv_cache_ref = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, KV_LORA_RANK, dtype=DTYPE, device=npu,
    )
    _run_op(kv, gamma, cos, sin, index, k_cache_ref, ckv_cache_ref, is_output_kv=False)
    torch.npu.synchronize()
    rows_k = k_cache_ref.reshape(-1, QK_ROPE_HEAD_DIM)[:batch_tokens].cpu()
    rows_ckv = ckv_cache_ref.reshape(-1, KV_LORA_RANK)[:batch_tokens].cpu()
    return rows_k.clone(), rows_ckv.clone()


def _empty_aligned_int8_cpu_tensors(sizes: list[int]) -> list[torch.Tensor]:
    """Replica of ``empty_aligned_int8_cpu_tensors`` in the offload manager.

    Allocates the CPU pools through ``memfabric_hybrid.offload.empty`` (pinned,
    host-registered memory used by the real D2H path) with 2 MiB alignment, so
    the tensors fed to the op are byte-identical to what the production
    ``k_caches_cpu`` / ``v_caches_cpu`` pools look like.
    """
    chunk_nums = [cdiv(size, _CPU_CACHE_ALIGNMENT) for size in sizes]
    total_chunk_num = 1 + sum(chunk_nums)
    raw_tensor = offload.empty(
        [total_chunk_num * _CPU_CACHE_ALIGNMENT], dtype=torch.int8, pin_memory=True,
    )
    base_addr = raw_tensor.data_ptr()
    if base_addr % _CPU_CACHE_ALIGNMENT:
        base_addr = (base_addr // _CPU_CACHE_ALIGNMENT + 1) * _CPU_CACHE_ALIGNMENT
    base_offset = base_addr - raw_tensor.data_ptr()
    allocate_tensors = []
    for size, chunk_num in zip(sizes, chunk_nums):
        allocate_tensors.append(raw_tensor[base_offset:base_offset + size])
        base_offset += chunk_num * _CPU_CACHE_ALIGNMENT
    return allocate_tensors


def _paged_view(int8_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Reinterpret an int8 offload pool as a paged BF16 cache view.

    Mirrors ``reshape_kv_cache_tensors_for_sparse_kv_offload``:
    ``raw.view(dtype).view(num_blocks, block_size, num_kv_heads, dim)``.
    """
    return int8_tensor.view(DTYPE).view(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, dim,
    )


def _offload_cpu_cache_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the CPU KV pools exactly like the offload manager does.

    Returns (k_cache_for_op, ckv_cache_for_op) where:
      - k_cache_for_op  holds k_pe  (qk_rope_head_dim)  -> op's ``k_cache``
      - ckv_cache_for_op holds k_nope (kv_lora_rank)    -> op's ``ckv_cache``
    """
    nope_bytes = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * KV_LORA_RANK * DTYPE.itemsize
    pe_bytes = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * QK_ROPE_HEAD_DIM * DTYPE.itemsize
    # Allocate k+nope and k_pe together so the per-layer delta-addr invariant
    # from empty_aligned_int8_cpu_tensors is preserved.
    nope_pool, pe_pool = _empty_aligned_int8_cpu_tensors([nope_bytes, pe_bytes])
    k_cache_for_op = _paged_view(pe_pool, QK_ROPE_HEAD_DIM)     # rope part
    ckv_cache_for_op = _paged_view(nope_pool, KV_LORA_RANK)     # rms  part
    return k_cache_for_op, ckv_cache_for_op


def _pageable_cpu_cache_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback: plain pageable torch.zeros CPU caches, for comparison only."""
    cpu = torch.device("cpu")
    k_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, QK_ROPE_HEAD_DIM, dtype=DTYPE, device=cpu,
    )
    ckv_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, KV_LORA_RANK, dtype=DTYPE, device=cpu,
    )
    return k_cache, ckv_cache


def _check_written(rows_k: torch.Tensor, rows_ckv: torch.Tensor) -> None:
    assert not torch.all(rows_k == 0), "k_cache (CPU) was not written by the op"
    assert not torch.all(rows_ckv == 0), "ckv_cache (CPU) was not written by the op"


def _check_close(rows_k: torch.Tensor, rows_ckv: torch.Tensor,
                 ref_k: torch.Tensor, ref_ckv: torch.Tensor) -> None:
    torch.testing.assert_close(rows_k, ref_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(rows_ckv, ref_ckv, rtol=1e-2, atol=1e-2)


def _resolve_cpu_cache_pair(allocator: str) -> tuple[torch.Tensor, torch.Tensor]:
    if allocator == "offload":
        return _offload_cpu_cache_pair()
    if allocator == "pageable":
        return _pageable_cpu_cache_pair()
    raise ValueError(f"unknown allocator: {allocator}")


def probe_writes_to_cpu(allocator: str, ref_k, ref_ckv) -> bool:
    print(f"\n[case] writes_to_cpu / {allocator}")
    npu = torch.device("npu")
    kv, gamma, cos, sin, index = _make_inputs(BATCH_TOKENS, npu)
    k_cache_cpu, ckv_cache_cpu = _resolve_cpu_cache_pair(allocator)

    try:
        _run_op(kv, gamma, cos, sin, index, k_cache_cpu, ckv_cache_cpu, is_output_kv=False)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED: {type(exc).__name__}: {exc}")
        return False

    rows_k = k_cache_cpu.reshape(-1, QK_ROPE_HEAD_DIM)[:BATCH_TOKENS]
    rows_ckv = ckv_cache_cpu.reshape(-1, KV_LORA_RANK)[:BATCH_TOKENS]

    try:
        _check_written(rows_k, rows_ckv)
    except AssertionError as exc:
        print(f"  NO-OP: {exc}")
        return False

    try:
        _check_close(rows_k, rows_ckv, ref_k, ref_ckv)
    except AssertionError as exc:
        print(f"  WRITTEN but mismatch vs NPU reference: {exc}")
        return False

    print(f"  PASS: op wrote {BATCH_TOKENS} rows into CPU {allocator} cache, "
          f"matches NPU reference within tol.")
    return True


def probe_is_output_kv_with_cpu(allocator: str) -> bool:
    print(f"\n[case] is_output_kv_with_cpu / {allocator}")
    npu = torch.device("npu")
    kv, gamma, cos, sin, index = _make_inputs(BATCH_TOKENS, npu)
    k_cache_cpu, ckv_cache_cpu = _resolve_cpu_cache_pair(allocator)

    try:
        k_out, ckv_out, k_rope, c_kv = _run_op(
            kv, gamma, cos, sin, index, k_cache_cpu, ckv_cache_cpu, is_output_kv=True,
        )
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED: {type(exc).__name__}: {exc}")
        return False

    ok = True
    if k_rope is None:
        print("  k_rope is None")
        ok = False
    else:
        print(f"  k_rope.shape={tuple(k_rope.shape)} "
              f"device={k_rope.device} dtype={k_rope.dtype}")
        if k_rope.shape[-1] != QK_ROPE_HEAD_DIM:
            print(f"  k_rope trailing dim != {QK_ROPE_HEAD_DIM}")
            ok = False
    if c_kv is None:
        print("  c_kv is None")
        ok = False
    else:
        print(f"  c_kv.shape={tuple(c_kv.shape)} "
              f"device={c_kv.device} dtype={c_kv.dtype}")
        if c_kv.shape[-1] != KV_LORA_RANK:
            print(f"  c_kv trailing dim != {KV_LORA_RANK}")
            ok = False

    rows_k = k_cache_cpu.reshape(-1, QK_ROPE_HEAD_DIM)[:BATCH_TOKENS]
    rows_ckv = ckv_cache_cpu.reshape(-1, KV_LORA_RANK)[:BATCH_TOKENS]
    try:
        _check_written(rows_k, rows_ckv)
        print(f"  CPU {allocator} caches were written by the op.")
    except AssertionError as exc:
        print(f"  NO-OP on caches: {exc}")
        ok = False
    return ok


def probe_diagnostic() -> None:
    """Just report whether the op raises or accepts CPU output, no assertions."""
    print("\n[case] diagnostic (raise vs accept) / offload")
    npu = torch.device("npu")
    kv, gamma, cos, sin, index = _make_inputs(BATCH_TOKENS, npu)
    k_cache_cpu, ckv_cache_cpu = _offload_cpu_cache_pair()

    raised: Exception | None = None
    try:
        _run_op(kv, gamma, cos, sin, index, k_cache_cpu, ckv_cache_cpu, is_output_kv=False)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        raised = exc

    if raised is not None:
        print(f"  -> npu_kv_rmsnorm_rope_cache RAISED on offload CPU cache output: "
              f"{type(raised).__name__}: {raised}")
    else:
        rows_k = k_cache_cpu.reshape(-1, QK_ROPE_HEAD_DIM)[:BATCH_TOKENS]
        rows_ckv = ckv_cache_cpu.reshape(-1, KV_LORA_RANK)[:BATCH_TOKENS]
        print(f"  -> accepted offload CPU cache output; "
              f"k_written={not torch.all(rows_k == 0)}, "
              f"ckv_written={not torch.all(rows_ckv == 0)}")


def main() -> int:
    if not _real_npu_available():
        print("[skip] npu_kv_rmsnorm_rope_cache CPU-output probe needs real "
              "NPU hardware (torch.version.cann is None or npu unavailable).")
        return 0

    print(f"torch.version.cann = {torch.version.cann}")
    print(f"dtype={DTYPE}, kv_lora_rank={KV_LORA_RANK}, "
          f"qk_rope_head_dim={QK_ROPE_HEAD_DIM}, "
          f"block_size={BLOCK_SIZE}, batch_tokens={BATCH_TOKENS}")
    print(f"cpu pool allocator: memfabric_hybrid.offload.empty "
          f"(pinned, {_CPU_CACHE_ALIGNMENT // (1024 * 1024)} MiB aligned)")

    try:
        ref_k, ref_ckv = _reference_npu(BATCH_TOKENS)
        print(f"[ref] NPU reference rows ready: k={tuple(ref_k.shape)} "
              f"ckv={tuple(ref_ckv.shape)}")
    except Exception:  # noqa: BLE001
        print("[ref] failed to build NPU reference:")
        traceback.print_exc()
        return 1

    # Diagnostic first so it always prints even if later cases assert.
    try:
        probe_diagnostic()
    except Exception:  # noqa: BLE001
        print("[diagnostic] unexpected error:")
        traceback.print_exc()

    results: dict[str, bool] = {}

    # Primary target: the offload allocator (what production would use).
    # Also probe a plain pageable torch.zeros CPU cache as a baseline to
    # distinguish "op rejects CPU output" from "op rejects pinned/registered
    # CPU output specifically".
    allocators = ["offload", "pageable"]
    for allocator in allocators:
        name = f"writes_to_cpu/{allocator}"
        try:
            results[name] = probe_writes_to_cpu(allocator, ref_k, ref_ckv)
        except Exception:  # noqa: BLE001
            print(f"[{name}] unexpected error:")
            traceback.print_exc()
            results[name] = False

    for allocator in allocators:
        name = f"is_output_kv_with_cpu/{allocator}"
        try:
            results[name] = probe_is_output_kv_with_cpu(allocator)
        except Exception:  # noqa: BLE001
            print(f"[{name}] unexpected error:")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    failures = [name for name, ok in results.items() if not ok]
    if failures:
        print(f"\n{len(failures)} case(s) failed: {failures}")
        return 1
    print("\nAll cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
