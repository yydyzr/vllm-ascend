# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KDA_COMMON_H = (
    REPO_ROOT / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h"
)
KDA_HOST_CMAKE = REPO_ROOT / "csrc/attention/chunk_kda_fwd/op_host/CMakeLists.txt"
GDN_STRUCT_H = (
    REPO_ROOT
    / "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel"
    / "chunk_gated_delta_rule_fwd_h_struct.h"
)
SOURCE_TREE_INCLUDE = (
    KDA_COMMON_H.parent
    / "../../../moe/chunk_gated_delta_rule_fwd_h/op_kernel"
    / "chunk_gated_delta_rule_fwd_h_struct.h"
)


def test_gdn_fwd_h_struct_header_exists():
    assert GDN_STRUCT_H.is_file()


def test_chunk_kda_fwd_source_tree_include_resolves_gdn_header():
    assert SOURCE_TREE_INCLUDE.resolve() == GDN_STRUCT_H.resolve()
    assert SOURCE_TREE_INCLUDE.resolve().is_file()


def test_chunk_kda_fwd_common_prefers_source_tree_or_include_path():
    common_text = KDA_COMMON_H.read_text(encoding="utf-8")
    cmake_text = KDA_HOST_CMAKE.read_text(encoding="utf-8")

    assert (
        "moe/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
        in common_text
    )
    assert "moe/chunk_gated_delta_rule_fwd_h/op_kernel" in cmake_text
    assert '#include "chunk_gated_delta_rule_fwd_h_struct.h"' in common_text
