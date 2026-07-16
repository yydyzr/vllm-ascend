from vllm_ascend.worker.sfa_decode_token_log import _format_token_rows, maybe_log_decode_step_tokens


def test_format_token_rows_includes_sampled_and_spec():
    text = _format_token_rows(
        ["req-a", "req-b"],
        [[11], [22, 23]],
        [[101, 102], []],
    )
    assert "req-a:sampled=[11]" in text
    assert "req-a:spec=[101, 102]" in text
    assert "req-b:sampled=[22, 23]" in text
    assert "req-b:spec" not in text


def test_maybe_log_decode_step_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.sfa_decode_token_log.envs.VLLM_ASCEND_SFA_DECODE_TOKEN_LOG",
        False,
    )
    maybe_log_decode_step_tokens(
        step=0,
        req_ids=["r1"],
        sampled_token_ids=[[1]],
        spec_token_ids=None,
        total_num_scheduled_tokens=1,
        cudagraph_mode="FULL",
        enforce_eager=False,
        cudagraph_stats=None,
        batch_desc="bd",
        tp_rank=0,
    )
