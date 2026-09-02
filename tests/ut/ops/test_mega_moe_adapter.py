from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.ut.base import PytestBase
from vllm_ascend.ops.fused_moe.mega_moe_adapter import (
    CannMegaMoeActivation,
    CannMegaMoeApiCapability,
    CannMegaMoeLayerCapability,
    evaluate_cann_mega_moe_layer,
    get_model_cann_mega_moe_capability,
    reset_cann_mega_moe_capability_state,
    resolve_cann_mega_moe_activation,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def _a5_api_capability(*, supports_situ: bool = True) -> CannMegaMoeApiCapability:
    return CannMegaMoeApiCapability(
        available=True,
        supports_situ=supports_situ,
        supports_comm_context_preload=True,
    )


def _moe_config(
    *,
    hidden_dim: int = 2048,
    intermediate_size_per_partition: int = 1024,
    experts_per_token: int = 8,
    ep_size: int = 8,
    num_experts: int = 256,
    in_dtype=torch.bfloat16,
    activation_situ_beta=None,
    activation_situ_linear_beta=None,
):
    return SimpleNamespace(
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_size_per_partition,
        experts_per_token=experts_per_token,
        ep_size=ep_size,
        num_experts=num_experts,
        in_dtype=in_dtype,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
    )


def _quant_method(quant_type: QuantType = QuantType.W4A8MXFP, group_size: int = 32):
    return SimpleNamespace(quant_type=quant_type, group_size=group_size)


class _MoEActivationSitu:
    """Stand-in for vllm.MoEActivation.SITU (enum member, no beta attributes)."""

    name = "SITU"
    value = "situ"

    def __str__(self) -> str:
        return "MoEActivation.SITU"


class TestMegaMoeAdapter(PytestBase):
    @pytest.fixture(autouse=True)
    def _reset_adapter_state(self):
        reset_cann_mega_moe_capability_state()
        yield
        reset_cann_mega_moe_capability_state()

    def test_resolve_swiglu_and_situ_activations(self):
        assert resolve_cann_mega_moe_activation("silu") == CannMegaMoeActivation(name="swiglu")
        situ = SimpleNamespace(linear_beta=1.5, beta=0.5)
        assert resolve_cann_mega_moe_activation(situ) == CannMegaMoeActivation(
            name="situglu",
            alpha=1.5,
            beta=0.5,
        )
        assert resolve_cann_mega_moe_activation("gelu") is None
        assert resolve_cann_mega_moe_activation(
            _MoEActivationSitu(),
            situ_beta=4.0,
            situ_linear_beta=25.0,
        ) == CannMegaMoeActivation(name="situglu", alpha=25.0, beta=4.0)
        assert resolve_cann_mega_moe_activation("situ") == CannMegaMoeActivation(name="situglu")

    def test_a5_w4a8_mxfp_group32_is_supported(self):
        capability = evaluate_cann_mega_moe_layer(
            _moe_config(),
            _quant_method(),
            "silu",
            device_type=AscendDeviceType.A5,
            api_capability=_a5_api_capability(),
        )
        assert capability.supported
        assert capability.quant_type == QuantType.W4A8MXFP
        assert capability.activation == CannMegaMoeActivation(name="swiglu")

    def test_a5_rejects_non_mxfp_quant(self):
        capability = evaluate_cann_mega_moe_layer(
            _moe_config(),
            _quant_method(QuantType.W8A8),
            "silu",
            device_type=AscendDeviceType.A5,
            api_capability=_a5_api_capability(),
        )
        assert not capability.supported
        assert "A5 MegaMoe does not support" in capability.reason

    def test_a5_rejects_non_32_group_size(self):
        capability = evaluate_cann_mega_moe_layer(
            _moe_config(),
            _quant_method(group_size=64),
            "silu",
            device_type=AscendDeviceType.A5,
            api_capability=_a5_api_capability(),
        )
        assert not capability.supported
        assert capability.reason == "A5 MXFP MegaMoe requires group_size=32"

    def test_a5_rejects_missing_comm_context_preload(self):
        capability = evaluate_cann_mega_moe_layer(
            _moe_config(),
            _quant_method(),
            "silu",
            device_type=AscendDeviceType.A5,
            api_capability=CannMegaMoeApiCapability(True, supports_comm_context_preload=False),
        )
        assert not capability.supported
        assert capability.reason == "A5 MegaMoe requires comm-context preloading"

    def test_a5_supports_situ_when_api_exposes_params(self):
        situ = SimpleNamespace(linear_beta=1.0, beta=2.0)
        capability = evaluate_cann_mega_moe_layer(
            _moe_config(),
            _quant_method(),
            situ,
            device_type=AscendDeviceType.A5,
            api_capability=_a5_api_capability(supports_situ=True),
        )
        assert capability.supported
        assert capability.activation == CannMegaMoeActivation(name="situglu", alpha=1.0, beta=2.0)

    def test_a5_supports_moe_activation_situ_enum(self):
        capability = evaluate_cann_mega_moe_layer(
            _moe_config(activation_situ_beta=4.0, activation_situ_linear_beta=25.0),
            _quant_method(),
            _MoEActivationSitu(),
            device_type=AscendDeviceType.A5,
            api_capability=_a5_api_capability(supports_situ=True),
        )
        assert capability.supported
        assert capability.activation == CannMegaMoeActivation(name="situglu", alpha=25.0, beta=4.0)

    def test_a3_supports_w8a8_and_rejects_mxfp(self):
        supported = evaluate_cann_mega_moe_layer(
            _moe_config(),
            _quant_method(QuantType.W8A8, group_size=None),
            "silu",
            device_type=AscendDeviceType.A3,
            api_capability=CannMegaMoeApiCapability(True),
        )
        assert supported.supported

        rejected = evaluate_cann_mega_moe_layer(
            _moe_config(),
            _quant_method(),
            "silu",
            device_type=AscendDeviceType.A3,
            api_capability=CannMegaMoeApiCapability(True),
        )
        assert not rejected.supported
        assert "A2/A3 MegaMoe does not support" in rejected.reason

    def test_get_model_capability_aggregates_unsupported_layer(self):
        class DummyLayer(nn.Module):
            def __init__(self, capability):
                super().__init__()
                self.cann_mega_moe_capability = capability

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = DummyLayer(
                    CannMegaMoeLayerCapability(True, "", QuantType.W4A8MXFP, CannMegaMoeActivation("swiglu"))
                )
                self.layer1 = DummyLayer(CannMegaMoeLayerCapability(False, "bad hidden", QuantType.W4A8MXFP))

        result = get_model_cann_mega_moe_capability(DummyModel())
        assert not result.supported
        assert result.reason == "bad hidden"

    def test_get_model_capability_falls_back_to_registered_layers(self):
        evaluate_cann_mega_moe_layer(
            _moe_config(),
            _quant_method(),
            "silu",
            device_type=AscendDeviceType.A5,
            api_capability=_a5_api_capability(),
        )
        result = get_model_cann_mega_moe_capability(None)
        assert result.supported
        assert result.quant_type == QuantType.W4A8MXFP


def test_cann_mega_moe_quant_settings_for_w4a8_mxfp():
    from vllm_ascend.ops.fused_moe import moe_utils

    mode, dispatch_dtype, weight_type = moe_utils._get_cann_mega_moe_quant_settings(QuantType.W4A8MXFP)
    assert mode == moe_utils._CANN_MEGA_MOE_QUANT_MODE_MX
    assert dispatch_dtype == torch.float8_e4m3fn
    assert weight_type is not None
