# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Guard spawn-time imports against the MegaMoe circular import.

Worker spawn unpickles quantization config, which does:

    modelslim_config -> methods.fp8 -> w4a8_mxfp4 -> ascend_forward_context._EXTRA_CTX

``vllm_ascend.ops.__init__`` side-imports fused_moe, which also imports
``_EXTRA_CTX``. A top-level ``mega_moe_adapter`` import in
``ascend_forward_context`` re-enters that package init while
``_EXTRA_CTX`` is still unbound.
"""

from __future__ import annotations

import inspect

import vllm_ascend.ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, _select_a5_moe_comm_method


def test_forward_context_does_not_import_ops_at_module_level():
    assert _EXTRA_CTX is not None
    assert "get_model_cann_mega_moe_capability" not in vars(afc)
    assert "from vllm_ascend.ops.fused_moe.mega_moe_adapter import" in inspect.getsource(
        _select_a5_moe_comm_method
    )
