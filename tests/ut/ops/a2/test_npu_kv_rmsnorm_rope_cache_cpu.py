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

"""Probe whether ``torch_npu.npu_kv_rmsnorm_rope_cache`` can scatter into
host-backed KV caches allocated by either:

  - MemFabric ``offload.empty`` (sparse KV offload CPU pool), or
  - ``torch_npu.empty_with_swapped_memory`` (NPU-marked, host-backed).

CPU KV allocation mirrors ``kv_offload_0717``
``tests/ut/test_fuse_sparse_attention_overlap_precisions.py``:
``_empty_aligned_cpu_tensor`` / ``_alloc_cpu_kv_tensor``.

WARNING: on some torch_npu builds, swapped tensors SIGSEGV if you touch
``.shape`` / ``.device`` / ``.data_ptr()`` / ``zero_()`` / ``__repr__``.
Only pass them into the op; do not print or introspect them. Swap probes
run in a child process so a segfault becomes FAIL instead of killing the
parent.

Run on an NPU box:

    python tests/ut/ops/a2/test_npu_kv_rmsnorm_rope_cache_cpu.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import traceback
from typing import Literal

import numpy as np
import torch
import torch_npu
from memfabric_hybrid import offload

empty_swapped = getattr(torch_npu, "empty_with_swapped_memory", None)
CpuKvBackend = Literal["offload", "swap_memory"]

# Match SparseKVOffloadManager / 0717 op UT CPU KV pool alignment.
_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
_GIB = 1024 * 1024 * 1024

# DeepSeek-V2 style MLA dims (PA layout for npu_kv_rmsnorm_rope_cache).
#
# Op args (see sfa_v1.py::exec_kv):
#   k_cache   = rope part (k_pe)  -> qk_rope_head_dim
#   ckv_cache = rms part  (k_nope) -> kv_lora_rank
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
NUM_BLOCKS = 16
DTYPE = torch.bfloat16
EPSILON = 1e-5
BATCH_TOKENS = 8
_SEED = 1234

_OFFLOAD_POOL_BYTES = int(
    os.environ.get("VLLM_ASCEND_PROBE_OFFLOAD_POOL_GB", "1")
) * _GIB

# Planned shapes as Python tuples only — never read swapped_tensor.shape.
K_CACHE_SHAPE = [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, QK_ROPE_HEAD_DIM]
CKV_CACHE_SHAPE = [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, KV_LORA_RANK]


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def _real_npu_available() -> bool:
    return torch.version.cann is not None and torch.npu.is_available()


def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    """Same as 0717 op UT / KVOffloadDecodeManager alignment helper."""
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset):]


def _empty_aligned_cpu_tensor(
    shape: list[int],
    dtype: torch.dtype,
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> torch.Tensor:
    """Same allocation path as 0717 UT / SparseKVOffloadManager CPU KV.

    Allocate raw int8 bytes from MemFabric, align, then view as ``dtype``.
    Avoids ``offload.empty(..., bfloat16)`` which some MemFabric builds reject.
    """
    num_elements = int(np.prod(shape))
    nbytes = num_elements * torch.empty((), dtype=dtype).element_size()
    extra_bytes = alignment
    print(
        "OFFLOAD_EMPTY shape={} dtype={} nbytes={:.3f} MiB (+align {:.3f} MiB)".format(
            shape, dtype, nbytes / (1024 * 1024), extra_bytes / (1024 * 1024),
        ),
        flush=True,
    )
    try:
        raw = offload.empty([nbytes + extra_bytes], dtype=torch.int8, pin_memory=True)
    except Exception as exc:
        raise RuntimeError(
            f"offload.empty failed for {(nbytes + extra_bytes) / _GIB:.3f} GiB "
            f"(shape={shape}, dtype={dtype}): {type(exc).__name__}: {exc}. "
            "Check MemFabric pool size and host free/locked memory."
        ) from exc
    aligned = _align_memory(raw, alignment)[:nbytes]
    return aligned.view(dtype).view(shape)


def _alloc_cpu_kv_tensor(
    backend: CpuKvBackend,
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate host-side KV used by the fused op.

    - offload: MemFabric ``offload.empty`` + 2MiB align
    - swap_memory: ``torch_npu.empty_with_swapped_memory``

    WARNING: on some torch_npu builds, swapped tensors SIGSEGV if you touch
    ``.shape`` / ``.device`` / ``.data_ptr()`` / ``zero_()`` / ``__repr__``.
    Only pass them into the op; do not print or introspect them.
    """
    if backend == "offload":
        return _empty_aligned_cpu_tensor(shape, dtype=dtype)
    if backend == "swap_memory":
        if empty_swapped is None:
            raise RuntimeError(
                "cpu-kv-backend=swap_memory requires "
                "torch_npu.empty_with_swapped_memory"
            )
        # Keep args minimal; do not print/return-repr the result here.
        print(
            "SWAP_EMPTY request_shape={} dtype={} (no tensor introspection)".format(
                shape, dtype,
            ),
            flush=True,
        )
        return empty_swapped(tuple(shape), dtype=dtype, device="npu")
    raise ValueError(f"unknown cpu-kv-backend: {backend}")


def init_offload_pool(dram_size_gb: float | None = None) -> None:
    """Mirror SparseKVOffloadManager MemFabric OffloadConfig initialize."""
    size_bytes = (
        int(dram_size_gb * _GIB) if dram_size_gb is not None else _OFFLOAD_POOL_BYTES
    )
    config = offload.OffloadConfig()
    config.device_id = torch_npu.npu.current_device()
    config.size = size_bytes
    config.world_size = 1
    config.rank_id = 0
    print(
        "OFFLOAD_INIT device_id={} size_gb={:.3f} world_size=1 rank_id=0".format(
            config.device_id, size_bytes / _GIB,
        ),
        flush=True,
    )
    try:
        offload.initialize(config)
    except Exception as exc:
        raise RuntimeError(
            f"offload.initialize failed for {size_bytes / _GIB:.3f} GiB pool: "
            f"{type(exc).__name__}: {exc}."
        ) from exc


def _alloc_host_kv_pair(backend: CpuKvBackend) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate (k_cache, ckv_cache) for the op.

    Returns (rope/k_pe cache, rms/k_nope cache) matching op argument order.
    Prints planned shapes only (Python lists). Never print swapped tensor attrs.
    """
    print(
        "CPU_KV_PLAN backend={} k_shape={} ckv_shape={} dtype={}".format(
            backend, K_CACHE_SHAPE, CKV_CACHE_SHAPE, DTYPE,
        ),
        flush=True,
    )
    k_cache = _alloc_cpu_kv_tensor(backend, K_CACHE_SHAPE, DTYPE)
    ckv_cache = _alloc_cpu_kv_tensor(backend, CKV_CACHE_SHAPE, DTYPE)

    if backend == "offload":
        # Safe for MemFabric host views; do NOT torch.randn (extra host malloc).
        k_cache.zero_()
        ckv_cache.zero_()
        print(
            "FULL_KV_HOST backend=offload shape={} ckv_shape={} dtype={} "
            "device={} ptr_align={}".format(
                tuple(k_cache.shape),
                tuple(ckv_cache.shape),
                DTYPE,
                k_cache.device,
                k_cache.data_ptr() % _CPU_CACHE_ALIGNMENT == 0,
            ),
            flush=True,
        )
    else:
        # swap_memory: skip zero_/shape/device/data_ptr — known SIGSEGV.
        # Leave allocation as-is; values are uninitialized.
        print(
            "FULL_KV_HOST backend=swap_memory requested_k_shape={} "
            "requested_ckv_shape={} dtype={} "
            "(skip tensor introspection to avoid SIGSEGV)".format(
                K_CACHE_SHAPE, CKV_CACHE_SHAPE, DTYPE,
            ),
            flush=True,
        )
    return k_cache, ckv_cache


def _make_inputs(batch_tokens: int, device: torch.device):
    """Deterministic inputs so parent/child probes share a reference."""
    torch.manual_seed(_SEED)
    if device.type == "npu":
        torch.npu.manual_seed(_SEED)
    kv = torch.randn(
        batch_tokens, NUM_KV_HEADS, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=DTYPE, device=device,
    )
    gamma = torch.ones(KV_LORA_RANK, dtype=DTYPE, device=device)
    cos = torch.randn(batch_tokens, 1, 1, QK_ROPE_HEAD_DIM, dtype=DTYPE, device=device)
    sin = torch.randn(batch_tokens, 1, 1, QK_ROPE_HEAD_DIM, dtype=DTYPE, device=device)
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
    """Run the op with normal NPU caches; return host rows for comparison."""
    npu = torch.device("npu")
    kv, gamma, cos, sin, index = _make_inputs(batch_tokens, npu)
    k_cache_ref = torch.zeros(K_CACHE_SHAPE, dtype=DTYPE, device=npu)
    ckv_cache_ref = torch.zeros(CKV_CACHE_SHAPE, dtype=DTYPE, device=npu)
    _run_op(kv, gamma, cos, sin, index, k_cache_ref, ckv_cache_ref, is_output_kv=False)
    torch.npu.synchronize()
    rows_k = k_cache_ref.reshape(-1, QK_ROPE_HEAD_DIM)[:batch_tokens].cpu()
    rows_ckv = ckv_cache_ref.reshape(-1, KV_LORA_RANK)[:batch_tokens].cpu()
    return rows_k.clone(), rows_ckv.clone()


def _materialize_swapped(swapped: torch.Tensor, shape: list[int]) -> torch.Tensor:
    """Copy swapped tensor into a normal NPU tensor via documented-safe mul_.

    Never touch ``swapped.shape`` / reshape / print / .cpu on *swapped*.
    Shape must be the planned Python list from allocate time.
    """
    npu = torch.device("npu")
    normal = torch.empty(shape, dtype=DTYPE, device=npu).fill_(1)
    normal.mul_(swapped)
    torch.npu.synchronize()
    return normal


def _read_rows(cache_tensor: torch.Tensor, dim: int, n: int,
               backend: CpuKvBackend) -> torch.Tensor:
    """Read first *n* token rows to host without introspecting swapped tensors."""
    if backend == "swap_memory":
        shape = K_CACHE_SHAPE if dim == QK_ROPE_HEAD_DIM else CKV_CACHE_SHAPE
        normal = _materialize_swapped(cache_tensor, shape)
        return normal.reshape(-1, dim)[:n].cpu().clone()
    return cache_tensor.reshape(-1, dim)[:n].cpu().clone()


def _check_written(rows_k: torch.Tensor, rows_ckv: torch.Tensor) -> None:
    assert not torch.all(rows_k == 0), "k_cache was not written by the op"
    assert not torch.all(rows_ckv == 0), "ckv_cache was not written by the op"


def _check_close(rows_k, rows_ckv, ref_k, ref_ckv) -> None:
    torch.testing.assert_close(rows_k, ref_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(rows_ckv, ref_ckv, rtol=1e-2, atol=1e-2)


def probe_writes_to_cpu(backend: CpuKvBackend, ref_k, ref_ckv) -> bool:
    print(f"\n[case] writes_to_cpu / {backend}", flush=True)
    npu = torch.device("npu")
    kv, gamma, cos, sin, index = _make_inputs(BATCH_TOKENS, npu)
    k_cache, ckv_cache = _alloc_host_kv_pair(backend)

    try:
        _run_op(kv, gamma, cos, sin, index, k_cache, ckv_cache, is_output_kv=False)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED: {type(exc).__name__}: {exc}", flush=True)
        return False

    try:
        rows_k = _read_rows(k_cache, QK_ROPE_HEAD_DIM, BATCH_TOKENS, backend)
        rows_ckv = _read_rows(ckv_cache, KV_LORA_RANK, BATCH_TOKENS, backend)
    except Exception as exc:  # noqa: BLE001
        print(f"  READ-BACK FAILED: {type(exc).__name__}: {exc}", flush=True)
        return False

    try:
        _check_written(rows_k, rows_ckv)
    except AssertionError as exc:
        print(f"  NO-OP: {exc}", flush=True)
        return False

    try:
        _check_close(rows_k, rows_ckv, ref_k, ref_ckv)
    except AssertionError as exc:
        print(f"  WRITTEN but mismatch vs NPU reference: {exc}", flush=True)
        return False

    print(
        f"  PASS: op wrote {BATCH_TOKENS} rows into {backend} cache, "
        f"matches NPU reference within tol.",
        flush=True,
    )
    return True


def probe_is_output_kv_with_cpu(backend: CpuKvBackend) -> bool:
    print(f"\n[case] is_output_kv_with_cpu / {backend}", flush=True)
    npu = torch.device("npu")
    kv, gamma, cos, sin, index = _make_inputs(BATCH_TOKENS, npu)
    k_cache, ckv_cache = _alloc_host_kv_pair(backend)

    try:
        _k_out, _ckv_out, k_rope, c_kv = _run_op(
            kv, gamma, cos, sin, index, k_cache, ckv_cache, is_output_kv=True,
        )
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED: {type(exc).__name__}: {exc}", flush=True)
        return False

    ok = True
    # Do not introspect returned tensors under swap_memory.
    if backend == "swap_memory":
        print(
            f"  k_rope is None? {k_rope is None}; c_kv is None? {c_kv is None}",
            flush=True,
        )
        if k_rope is None or c_kv is None:
            ok = False
    else:
        if k_rope is None:
            print("  k_rope is None", flush=True)
            ok = False
        else:
            print(
                f"  k_rope.shape={tuple(k_rope.shape)} "
                f"device={k_rope.device} dtype={k_rope.dtype}",
                flush=True,
            )
            if k_rope.shape[-1] != QK_ROPE_HEAD_DIM:
                print(f"  k_rope trailing dim != {QK_ROPE_HEAD_DIM}", flush=True)
                ok = False
        if c_kv is None:
            print("  c_kv is None", flush=True)
            ok = False
        else:
            print(
                f"  c_kv.shape={tuple(c_kv.shape)} "
                f"device={c_kv.device} dtype={c_kv.dtype}",
                flush=True,
            )
            if c_kv.shape[-1] != KV_LORA_RANK:
                print(f"  c_kv trailing dim != {KV_LORA_RANK}", flush=True)
                ok = False

    try:
        rows_k = _read_rows(k_cache, QK_ROPE_HEAD_DIM, BATCH_TOKENS, backend)
        rows_ckv = _read_rows(ckv_cache, KV_LORA_RANK, BATCH_TOKENS, backend)
    except Exception as exc:  # noqa: BLE001
        print(f"  READ-BACK FAILED: {type(exc).__name__}: {exc}", flush=True)
        return False
    try:
        _check_written(rows_k, rows_ckv)
        print(f"  {backend} caches were written by the op.", flush=True)
    except AssertionError as exc:
        print(f"  NO-OP on caches: {exc}", flush=True)
        ok = False
    return ok


def probe_diagnostic(backend: CpuKvBackend) -> None:
    print(f"\n[case] diagnostic (raise vs accept) / {backend}", flush=True)
    npu = torch.device("npu")
    kv, gamma, cos, sin, index = _make_inputs(BATCH_TOKENS, npu)
    k_cache, ckv_cache = _alloc_host_kv_pair(backend)

    raised: Exception | None = None
    try:
        _run_op(kv, gamma, cos, sin, index, k_cache, ckv_cache, is_output_kv=False)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        raised = exc

    if raised is not None:
        print(
            f"  -> RAISED on {backend} cache output: "
            f"{type(raised).__name__}: {raised}",
            flush=True,
        )
        return

    try:
        rows_k = _read_rows(k_cache, QK_ROPE_HEAD_DIM, BATCH_TOKENS, backend)
        rows_ckv = _read_rows(ckv_cache, KV_LORA_RANK, BATCH_TOKENS, backend)
        print(
            f"  -> accepted {backend} cache output; "
            f"k_written={not torch.all(rows_k == 0)}, "
            f"ckv_written={not torch.all(rows_ckv == 0)}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"  -> accepted {backend} but READ-BACK FAILED: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def _run_swap_in_subprocess(case_name: str) -> bool:
    """Isolate swap_memory probes: segfault becomes FAIL, parent keeps running."""
    script = os.path.abspath(__file__)
    print(f"\n[case] {case_name}  (subprocess-isolated)", flush=True)
    proc = subprocess.run(
        [sys.executable, script, "--probe-swap", case_name],
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", flush=True)

    if proc.returncode == 0:
        return True
    if proc.returncode < 0:
        sig = -proc.returncode
        try:
            signame = signal.Signals(sig).name
        except ValueError:
            signame = f"signal {sig}"
        print(f"  [segfault] swap child killed by {signame}", flush=True)
        return False
    print(f"  [exit {proc.returncode}] swap child failed", flush=True)
    return False


def _run_single_swap_probe(case_name: str) -> int:
    """Child entry: run one swap_memory probe and exit 0/1."""
    try:
        ref_k, ref_ckv = _reference_npu(BATCH_TOKENS)
    except Exception:  # noqa: BLE001
        print("[ref] failed in swap child:", flush=True)
        traceback.print_exc()
        return 1

    if case_name == "writes_to_cpu/swap_memory":
        ok = probe_writes_to_cpu("swap_memory", ref_k, ref_ckv)
    elif case_name == "is_output_kv_with_cpu/swap_memory":
        ok = probe_is_output_kv_with_cpu("swap_memory")
    elif case_name == "diagnostic/swap_memory":
        probe_diagnostic("swap_memory")
        ok = True
    else:
        print(f"[swap child] unknown case: {case_name}", flush=True)
        return 1
    return 0 if ok else 1


def main() -> int:
    if not _real_npu_available():
        print(
            "[skip] needs real NPU (torch.version.cann is None or npu unavailable).",
            flush=True,
        )
        return 0

    print(f"torch.version.cann = {torch.version.cann}", flush=True)
    print(
        f"dtype={DTYPE}, kv_lora_rank={KV_LORA_RANK}, "
        f"qk_rope_head_dim={QK_ROPE_HEAD_DIM}, "
        f"block_size={BLOCK_SIZE}, batch_tokens={BATCH_TOKENS}",
        flush=True,
    )
    print(
        "cpu-kv-backend: offload=MemFabric offload.empty; "
        "swap_memory=torch_npu.empty_with_swapped_memory "
        "(alloc path mirrors kv_offload_0717 op UT)",
        flush=True,
    )

    try:
        init_offload_pool()
    except Exception:  # noqa: BLE001
        print("[init] failed to initialize memfabric offload pool:", flush=True)
        traceback.print_exc()
        print("[init] offload cases will fail; swap_memory cases still run.", flush=True)

    try:
        ref_k, ref_ckv = _reference_npu(BATCH_TOKENS)
        print(
            f"[ref] NPU reference rows ready: k={tuple(ref_k.shape)} "
            f"ckv={tuple(ref_ckv.shape)}",
            flush=True,
        )
    except Exception:  # noqa: BLE001
        print("[ref] failed to build NPU reference:", flush=True)
        traceback.print_exc()
        return 1

    results: dict[str, bool] = {}

    # ---- offload (in-process) ----
    try:
        probe_diagnostic("offload")
    except Exception:  # noqa: BLE001
        print("[diagnostic/offload] unexpected error:", flush=True)
        traceback.print_exc()

    for name, fn in (
        ("writes_to_cpu/offload",
         lambda: probe_writes_to_cpu("offload", ref_k, ref_ckv)),
        ("is_output_kv_with_cpu/offload",
         lambda: probe_is_output_kv_with_cpu("offload")),
    ):
        try:
            results[name] = fn()
        except Exception:  # noqa: BLE001
            print(f"[{name}] unexpected error:", flush=True)
            traceback.print_exc()
            results[name] = False

    # ---- swap_memory (subprocess-isolated; segfault -> FAIL) ----
    _run_swap_in_subprocess("diagnostic/swap_memory")
    results["writes_to_cpu/swap_memory"] = _run_swap_in_subprocess(
        "writes_to_cpu/swap_memory"
    )
    results["is_output_kv_with_cpu/swap_memory"] = _run_swap_in_subprocess(
        "is_output_kv_with_cpu/swap_memory"
    )

    print("\n" + "=" * 60, flush=True)
    print("Summary", flush=True)
    print("=" * 60, flush=True)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)

    failures = [name for name, ok in results.items() if not ok]
    if failures:
        print(f"\n{len(failures)} case(s) failed: {failures}", flush=True)
        return 1
    print("\nAll cases passed.", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--probe-swap":
        sys.exit(_run_single_swap_probe(sys.argv[2]))
    sys.exit(main())
