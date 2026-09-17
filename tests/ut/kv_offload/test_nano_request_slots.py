# SPDX-License-Identifier: Apache-2.0
"""CPU tests for nano request-slot occupancy and missed-D2D restore."""

from types import SimpleNamespace

import numpy as np

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _make_runner() -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.max_num_reqs = 2
    runner._offload_pool_slots = SimpleNamespace(np=np.zeros(4, dtype=np.int32), copy_to_gpu=lambda n: None)
    runner._offload_pool_generations = SimpleNamespace(np=np.zeros(4, dtype=np.int64), copy_to_gpu=lambda n: None)
    runner._offload_request_slots = {}
    runner._offload_slot_generation = 0
    runner._offload_slot_generations = {}
    runner._offload_slot_last_prefix = {}
    runner._nano_need_eager_tail_restore = False
    runner._prebound_nano_slots = lambda: {}
    runner._nano_tails_pending_d2d_restore = lambda: set()
    return runner


def test_missed_tail_d2d_requests_eager_restore():
    runner = _make_runner()
    runner.input_batch = SimpleNamespace(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1})
    runner._nano_tails_pending_d2d_restore = lambda: {"a"}
    runner._prepare_nano_request_slots(2, 3, dummy=False)
    assert runner._nano_need_eager_tail_restore is True


def test_landed_tail_d2d_does_not_request_eager_restore():
    runner = _make_runner()
    runner.input_batch = SimpleNamespace(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1})
    runner._prepare_nano_request_slots(2, 3, dummy=False)
    assert runner._nano_need_eager_tail_restore is False
