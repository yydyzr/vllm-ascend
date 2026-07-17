import os
import platform
import sys
from pathlib import Path

import pytest
import torch
import torch_npu
from torch.utils.cpp_extension import load


def _load_cpu_sparse_attn():
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / (
        "vllm_ascend/distributed/kv_transfer/sfa_kv_offload/cpu_sparse_attn.cpp"
    )
    ascend_home = os.environ.get(
        "ASCEND_HOME_PATH",
        "/usr/local/Ascend/ascend-toolkit/latest",
    )
    npu_lib_path = Path(ascend_home) / "lib64"
    if not npu_lib_path.exists():
        npu_lib_path = Path(ascend_home) / "lib"
    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    return load(
        name="cpu_sparse_attn_current_kv_d2h_test",
        sources=[str(source)],
        extra_cflags=[
            "-O3",
            "-std=c++20",
            "-fopenmp",
            "-march=armv8.2-a+sve+fp16+bf16",
            f"-I{Path(ascend_home) / 'include'}",
            f"-I{torch_npu_root / 'include'}",
        ],
        extra_ldflags=[
            "-fopenmp",
            f"-L{npu_lib_path}",
            "-lascendcl",
            f"-L{torch_npu_root / 'lib'}",
            "-ltorch_npu",
        ],
        verbose=False,
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="Ascend ACL is unavailable on macOS")
@pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64"),
    reason="the production extension uses Ascend aarch64 compiler flags",
)
def test_current_kv_d2h_descriptors_compact_padded_v_entries():
    ext = _load_cpu_sparse_attn()
    slot_mapping = torch.tensor([0, 1, 2, -1], dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cum_query_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    offload_token_start = torch.tensor([0, 0, 0, -1], dtype=torch.int32)
    offload_num_tokens = torch.tensor([1, 1, 1, 0], dtype=torch.int32)
    cpu_block_table = torch.tensor([[1], [2], [3], [0]], dtype=torch.int32)
    gvas = torch.zeros(8, dtype=torch.int64)
    addrs = torch.zeros(8, dtype=torch.int64)
    sizes = torch.zeros(8, dtype=torch.int32)
    copy_count = torch.zeros(1, dtype=torch.int32)

    result = ext.compute_current_kv_d2h_descriptors(
        slot_mapping,
        token_to_req,
        cum_query_lens,
        offload_token_start,
        offload_num_tokens,
        cpu_block_table,
        128,
        1024,
        128,
        30_000_000,
        40_000_000,
        10_000_000,
        20_000_000,
        4,
        4,
        gvas,
        addrs,
        sizes,
        copy_count,
    )

    assert result == 6
    assert copy_count.tolist() == [6]
    assert gvas[:6].tolist() == [
        30_000_000 + 128 * 1024,
        30_000_000 + 2 * 128 * 1024,
        30_000_000 + 3 * 128 * 1024,
        40_000_000 + 128 * 128,
        40_000_000 + 2 * 128 * 128,
        40_000_000 + 3 * 128 * 128,
    ]
    assert addrs[:6].tolist() == [
        10_000_000,
        10_000_000 + 1024,
        10_000_000 + 2 * 1024,
        20_000_000,
        20_000_000 + 128,
        20_000_000 + 2 * 128,
    ]
    assert sizes[:6].tolist() == [1024, 1024, 1024, 128, 128, 128]
