from pathlib import Path

import torch

from vllm_ascend.attention import fused_overlap_debug as dump_mod


def test_dump_op_inputs_writes_step_in_name(tmp_path: Path):
    path = dump_mod.dump_op_inputs(
        tmp_path,
        mode="fused_graph",
        layer_name="model.layers.0.self_attn",
        layer_id=0,
        op_name="npu_fused_sparse_attention_overlap",
        inputs={"full_kv_cache": torch.arange(4, dtype=torch.float32), "scale_value": 0.5},
        rank=0,
        pid=123,
        step=2,
    )
    assert path.name == "sfa_fused_graph_inputs_layer0_step2_rank0_pid123.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["mode"] == "fused_graph"
    assert payload["step"] == 2
    assert payload["inputs"]["scale_value"] == 0.5
    torch.testing.assert_close(payload["inputs"]["full_kv_cache"], torch.arange(4, dtype=torch.float32))


def test_host_callback_filters_by_dump_steps(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dump_mod, "_graph_inputs_step_counts", {})
    monkeypatch.setattr(dump_mod, "_is_stream_capturing", lambda: False)
    staged = {
        "full_kv_cache": torch.ones(2, 2),
        "query": torch.zeros(2, 2),
        "scale_value": 1.0,
    }
    dump_steps = frozenset({1, 3})
    for _ in range(4):
        dump_mod._dump_fused_inputs_host_callback(
            (
                str(tmp_path),
                "fused_graph",
                "model.layers.0.self_attn",
                0,
                "npu_fused_sparse_attention_overlap",
                0,
                99,
                dump_steps,
                staged,
            )
        )
    files = sorted(p.name for p in tmp_path.glob("sfa_fused_graph_inputs_*.pt"))
    assert files == [
        "sfa_fused_graph_inputs_layer0_step1_rank0_pid99.pt",
        "sfa_fused_graph_inputs_layer0_step3_rank0_pid99.pt",
    ]


def test_host_callback_uses_per_layer_step_counters(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dump_mod, "_graph_inputs_step_counts", {})
    monkeypatch.setattr(dump_mod, "_is_stream_capturing", lambda: False)
    staged = {"full_kv_cache": torch.ones(2, 2)}
    dump_steps = frozenset({0})
    for layer_id in (0, 58):
        dump_mod._dump_fused_inputs_host_callback(
            (
                str(tmp_path),
                "fused_graph",
                f"model.layers.{layer_id}.self_attn",
                layer_id,
                "npu_fused_sparse_attention_overlap",
                0,
                99,
                dump_steps,
                staged,
            )
        )
    files = sorted(p.name for p in tmp_path.glob("sfa_fused_graph_inputs_*.pt"))
    assert files == [
        "sfa_fused_graph_inputs_layer0_step0_rank0_pid99.pt",
        "sfa_fused_graph_inputs_layer58_step0_rank0_pid99.pt",
    ]


def test_host_callback_skips_during_capture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dump_mod, "_graph_inputs_step_counts", {})
    monkeypatch.setattr(dump_mod, "_is_stream_capturing", lambda: True)
    staged = {"full_kv_cache": torch.ones(2, 2)}
    dump_mod._dump_fused_inputs_host_callback(
        (
            str(tmp_path),
            "fused_graph",
            "model.layers.0.self_attn",
            0,
            "npu_fused_sparse_attention_overlap",
            0,
            99,
            frozenset({0}),
            staged,
        )
    )
    assert dump_mod._graph_inputs_step_counts == {}
    assert list(tmp_path.glob("sfa_fused_graph_inputs_*.pt")) == []
