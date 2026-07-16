from unittest.mock import MagicMock

from vllm_ascend.attention import npu_host_func_stream as stream_mod


def test_ensure_host_func_stream_subscribed_only_once(monkeypatch):
    subscribed: list = []

    def fake_subscribe(stream):
        subscribed.append(stream)

    fake_npu = MagicMock()
    fake_npu.npu._subscribe_report = fake_subscribe
    monkeypatch.setattr(stream_mod, "torch_npu", fake_npu)
    monkeypatch.setattr(stream_mod, "_SUBSCRIBED_HOST_FUNC_STREAMS", set())

    stream = object()
    stream_mod.ensure_host_func_stream_subscribed(stream)
    stream_mod.ensure_host_func_stream_subscribed(stream)

    assert subscribed == [stream]
    assert stream in stream_mod.get_subscribed_host_func_streams()
