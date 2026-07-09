"""Probe whether mla_preprocess (MLAPO) can write KV directly into
memfabric offload.empty()-allocated host memory.

Background:
  offload.empty() allocates from the memfabric/hybm host pool. That host memory
  is MTE-visible (GVA / DRAM-mapped), so NPU kernels can DMA-write it without a
  separate H2D/D2H staging buffer — if the op passes the host ptr through as a
  valid MTE destination.

Baseline: tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_preprocess.py

Run (on NPU host with memfabric_hybrid installed):
  python tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_preprocess_offload_cpu_cache.py
"""

from __future__ import annotations

import gc
import sys
import traceback

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

_OFFLOAD_DRAM_BYTES = 1 * 1024 * 1024 * 1024
CACHE_MODE = "krope_ctkv"


def _require_offload():
    try:
        from memfabric_hybrid import offload
    except ImportError as e:
        raise RuntimeError(f"memfabric_hybrid not installed: {e}") from e
    return offload


def _init_offload(offload, device_id: int = 0) -> None:
    ret = offload.initialize(device_id, _OFFLOAD_DRAM_BYTES)
    if ret != 0:
        raise RuntimeError(f"offload.initialize failed, ret={ret}")


def _build_npu_inputs(token_num: int, head_num: int, dtype: torch.dtype):
    N_7168 = 7168
    hidden_states = torch.randn((token_num, N_7168), dtype=dtype).npu()
    quant_scale0 = torch.randn((1,), dtype=dtype).npu()
    quant_offset0 = torch.randint(0, 7, (1,), dtype=torch.int8).npu()

    wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8).npu()
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)

    de_scale0 = torch.rand((2112,), dtype=torch.float).npu()
    bias0 = torch.randint(0, 7, (2112,), dtype=torch.int32).npu()
    gamma1 = torch.randn((1536,), dtype=dtype).npu()
    beta1 = torch.randn((1536,), dtype=dtype).npu()
    quant_scale1 = torch.randn((1,), dtype=dtype).npu()
    quant_offset1 = torch.randint(0, 7, (1,), dtype=torch.int8).npu()

    wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8).npu()
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)

    de_scale1 = torch.rand((head_num * 192,), dtype=torch.float).npu()
    bias1 = torch.randint(0, 7, (head_num * 192,), dtype=torch.int32).npu()
    gamma2 = torch.randn((512,), dtype=dtype).npu()
    cos = torch.randn((token_num, 64), dtype=dtype).npu()
    sin = torch.randn((token_num, 64), dtype=dtype).npu()

    wuk = torch.randn((head_num, 128, 512), dtype=dtype).npu()
    wuk = torch_npu.npu_format_cast(wuk, 29)

    slotmapping = torch.zeros((token_num,), dtype=torch.int32).npu()

    ctkv_scale = torch.randn((1,), dtype=dtype).npu()
    qnope_scale = torch.randn((head_num,), dtype=dtype).npu()
    return dict(
        hidden_states=hidden_states,
        wdqkv=wdqkv,
        de_scale0=de_scale0,
        gamma1=gamma1,
        beta1=beta1,
        wuq=wuq,
        de_scale1=de_scale1,
        gamma2=gamma2,
        cos=cos,
        sin=sin,
        wuk=wuk,
        slotmapping=slotmapping,
        quant_scale0=quant_scale0,
        quant_offset0=quant_offset0,
        bias0=bias0,
        quant_scale1=quant_scale1,
        quant_offset1=quant_offset1,
        bias1=bias1,
        ctkv_scale=ctkv_scale,
        q_nope_scale=qnope_scale,
    )


def _run_mla_preprocess(
    *,
    cache_mode: str,
    kv_cache: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    inputs: dict,
):
    hidden_states = inputs["hidden_states"]
    wuk = inputs["wuk"]
    q_nope_out = torch.empty(
        (hidden_states.shape[0], wuk.shape[0], kv_cache.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    q_rope_out = torch.empty(
        (hidden_states.shape[0], wuk.shape[0], kv_cache_rope.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    q_down = torch.empty(
        (hidden_states.shape[0], 1536),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    torch.ops._C_ascend.mla_preprocess(
        hidden_states,
        inputs["wdqkv"],
        inputs["de_scale0"],
        inputs["gamma1"],
        inputs["beta1"],
        inputs["wuq"],
        inputs["de_scale1"],
        inputs["gamma2"],
        inputs["cos"],
        inputs["sin"],
        wuk,
        kv_cache,
        kv_cache_rope,
        inputs["slotmapping"],
        quant_scale0=inputs["quant_scale0"],
        quant_offset0=inputs["quant_offset0"],
        bias0=inputs["bias0"],
        quant_scale1=inputs["quant_scale1"],
        quant_offset1=inputs["quant_offset1"],
        bias1=inputs["bias1"],
        ctkv_scale=inputs["ctkv_scale"],
        q_nope_scale=inputs["q_nope_scale"],
        cache_mode=cache_mode,
        quant_mode="per_tensor_quant_asymm",
        enable_inner_out=False,
        q_out0=q_nope_out,
        kv_cache_out0=kv_cache,
        q_out1=q_rope_out,
        kv_cache_out1=kv_cache_rope,
        inner_out=q_down,
    )
    torch.npu.synchronize()
    return q_nope_out, q_rope_out


@torch.inference_mode()
def case_write_offload_cpu_kv_cache() -> None:
    """Can MLAPO write kv_cache_out into offload.empty host tensors?"""
    print("[case1] write into offload.empty host kv_cache ...")
    offload = _require_offload()
    device_id = int(torch.npu.current_device())
    _init_offload(offload, device_id=device_id)

    token_num = 1
    head_num = 2
    block_num = 1
    block_size = 128
    dtype = torch.bfloat16
    kv_shape = (block_num, head_num * 512 // 32, block_size, 32)
    rope_shape = (block_num, head_num * 64 // 16, block_size, 16)

    kv_cache_cpu = offload.empty(list(kv_shape), dtype=dtype, pin_memory=True)
    kv_cache_rope_cpu = offload.empty(list(rope_shape), dtype=dtype, pin_memory=True)
    assert kv_cache_cpu.device.type == "cpu"
    assert kv_cache_rope_cpu.device.type == "cpu"

    kv_cache_cpu.fill_(0)
    kv_cache_rope_cpu.fill_(0)
    kv_before = kv_cache_cpu.clone()
    rope_before = kv_cache_rope_cpu.clone()

    inputs = _build_npu_inputs(token_num, head_num, dtype)

    try:
        q_nope_out, q_rope_out = _run_mla_preprocess(
            cache_mode=CACHE_MODE,
            kv_cache=kv_cache_cpu,
            kv_cache_rope=kv_cache_rope_cpu,
            inputs=inputs,
        )
    except Exception as e:
        raise RuntimeError(
            "mla_preprocess failed when kv_cache / kv_cache_rope are "
            f"offload.empty host tensors: {type(e).__name__}: {e}"
        ) from e
    finally:
        try:
            offload.uninitialize()
        except Exception:
            pass

    assert q_nope_out.device.type == "npu"
    assert q_rope_out.device.type == "npu"
    assert torch.isfinite(q_nope_out.float()).all()
    assert torch.isfinite(q_rope_out.float()).all()

    kv_changed = not torch.equal(kv_cache_cpu, kv_before)
    rope_changed = not torch.equal(kv_cache_rope_cpu, rope_before)
    print(f"  kv_cache changed={kv_changed}, rope_cache changed={rope_changed}")
    if not (kv_changed or rope_changed):
        raise AssertionError(
            "mla_preprocess returned without error, but offload.empty (MTE-visible) "
            "host kv_cache / kv_cache_rope were not modified. Either the op did not "
            "treat the host ptr as an MTE destination, or the write landed elsewhere."
        )
    print("[case1] PASS")


@torch.inference_mode()
def case_offload_cpu_vs_npu_kv_cache() -> None:
    """Compare host-offload cache write against the normal NPU cache path."""
    print("[case2] compare offload host cache vs NPU cache ...")
    offload = _require_offload()
    device_id = int(torch.npu.current_device())
    _init_offload(offload, device_id=device_id)

    token_num = 1
    head_num = 2
    block_num = 1
    block_size = 128
    dtype = torch.bfloat16
    kv_shape = (block_num, head_num * 512 // 32, block_size, 32)
    rope_shape = (block_num, head_num * 64 // 16, block_size, 16)

    inputs = _build_npu_inputs(token_num, head_num, dtype)
    kv_cache_npu = torch.zeros(kv_shape, dtype=dtype, device="npu")
    kv_cache_rope_npu = torch.zeros(rope_shape, dtype=dtype, device="npu")
    q_nope_npu, q_rope_npu = _run_mla_preprocess(
        cache_mode=CACHE_MODE,
        kv_cache=kv_cache_npu,
        kv_cache_rope=kv_cache_rope_npu,
        inputs=inputs,
    )

    kv_cache_cpu = offload.empty(list(kv_shape), dtype=dtype, pin_memory=True)
    kv_cache_rope_cpu = offload.empty(list(rope_shape), dtype=dtype, pin_memory=True)
    kv_cache_cpu.zero_()
    kv_cache_rope_cpu.zero_()

    try:
        q_nope_cpu_path, q_rope_cpu_path = _run_mla_preprocess(
            cache_mode=CACHE_MODE,
            kv_cache=kv_cache_cpu,
            kv_cache_rope=kv_cache_rope_cpu,
            inputs=inputs,
        )
    except Exception as e:
        raise RuntimeError(
            "host-offload kv_cache path crashed while NPU path succeeded: "
            f"{type(e).__name__}: {e}"
        ) from e
    finally:
        try:
            offload.uninitialize()
        except Exception:
            pass

    torch.testing.assert_close(q_nope_cpu_path, q_nope_npu, atol=0, rtol=0)
    torch.testing.assert_close(q_rope_cpu_path, q_rope_npu, atol=0, rtol=0)
    torch.testing.assert_close(kv_cache_cpu, kv_cache_npu.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(kv_cache_rope_cpu, kv_cache_rope_npu.cpu(), atol=0, rtol=0)
    print("[case2] PASS")

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def main() -> int:
    cases = [
        case_write_offload_cpu_kv_cache,
        case_offload_cpu_vs_npu_kv_cache,
    ]
    failed = 0
    for case in cases:
        try:
            case()
        except Exception:
            failed += 1
            print(f"[{case.__name__}] FAIL")
            traceback.print_exc()
    if failed:
        print(f"\n{failed}/{len(cases)} case(s) failed")
        return 1
    print(f"\nAll {len(cases)} case(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
