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

import unittest
from unittest.mock import MagicMock, patch

from vllm.config import VllmConfig

from vllm_ascend.ascend_config import (
    KV_OFFLOAD_MODE_FUSED_OVERLAP,
    KV_OFFLOAD_MODE_LEGACY,
    clear_ascend_config,
    init_ascend_config,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import RequestTracker
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_scheduler import (
    SFAKVOffloadlScheduler,
)


def _make_kv_cache_config(block_size: int = 16, num_blocks: int = 100):
    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = block_size
    kv_cache_group = MagicMock()
    kv_cache_group.kv_cache_spec = kv_cache_spec
    kv_cache_config = MagicMock()
    kv_cache_config.num_blocks = num_blocks
    kv_cache_config.kv_cache_groups = [kv_cache_group, kv_cache_group]
    return kv_cache_config


def _make_vllm_config(kv_offload_mode: str):
    vllm_config = VllmConfig()
    vllm_config.additional_config = {
        "use_offload": True,
        "kv_offload_mode": kv_offload_mode,
    }
    vllm_config.kv_transfer_config = MagicMock()
    vllm_config.kv_transfer_config.kv_role = "kv_producer"
    vllm_config.kv_transfer_config.kv_connector_extra_config = {}
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.decode_context_parallel_size = 1
    init_ascend_config(vllm_config)
    return vllm_config


def _make_sched_output(
    *,
    new_reqs=None,
    cached_req_ids=None,
    cached_num_computed_tokens=None,
    cached_new_block_ids=None,
    num_scheduled_tokens=None,
):
    sched_output = MagicMock()
    sched_output.finished_req_ids = set()
    sched_output.preempted_req_ids = set()
    sched_output.scheduled_new_reqs = new_reqs or []
    sched_output.num_scheduled_tokens = num_scheduled_tokens or {}
    sched_output.scheduled_spec_decode_tokens = {}
    sched_output.scheduled_cached_reqs = MagicMock()
    sched_output.scheduled_cached_reqs.req_ids = cached_req_ids or []
    sched_output.scheduled_cached_reqs.num_computed_tokens = cached_num_computed_tokens or []
    sched_output.scheduled_cached_reqs.new_block_ids = cached_new_block_ids or []
    return sched_output


@patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
class TestSFAKVOffloadSchedulerMetadata(unittest.TestCase):
    def tearDown(self):
        clear_ascend_config()

    def _make_scheduler(self, kv_offload_mode: str, block_size: int = 16):
        vllm_config = _make_vllm_config(kv_offload_mode)
        return SFAKVOffloadlScheduler(
            vllm_config,
            use_layerwise=True,
            kv_cache_config=_make_kv_cache_config(block_size=block_size),
        )

    def test_legacy_decode_does_not_allocate_partial_cpu_block(self, _mock):
        scheduler = self._make_scheduler(KV_OFFLOAD_MODE_LEGACY)
        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            allocated_block_ids_npu=[10, 11, 12, 13, 14, 15],
            allocated_block_ids_cpu=[1, 2, 3, 4, 5, 6],
        )
        scheduler._unfinished_requests["r1"] = (MagicMock(), [])
        scheduler._unfinished_request_ids.add("r1")

        sched_output = _make_sched_output(
            cached_req_ids=["r1"],
            cached_num_computed_tokens=[96],
            cached_new_block_ids=[[16]],
            num_scheduled_tokens={"r1": 1},
        )
        meta = scheduler.build_connector_meta(sched_output)
        req_meta = meta.requests[0]
        self.assertEqual(req_meta.num_new_offload_blocks, 0)
        self.assertEqual(req_meta.offload_num_tokens, 0)
        self.assertEqual(len(scheduler._request_trackers["r1"].allocated_block_ids_cpu), 6)

    def test_fused_decode_allocates_partial_cpu_block(self, _mock):
        scheduler = self._make_scheduler(KV_OFFLOAD_MODE_FUSED_OVERLAP)
        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            allocated_block_ids_npu=[10, 11, 12, 13, 14, 15],
            allocated_block_ids_cpu=[1, 2, 3, 4, 5, 6],
        )
        scheduler._unfinished_requests["r1"] = (MagicMock(), [])
        scheduler._unfinished_request_ids.add("r1")

        sched_output = _make_sched_output(
            cached_req_ids=["r1"],
            cached_num_computed_tokens=[96],
            cached_new_block_ids=[[16]],
            num_scheduled_tokens={"r1": 1},
        )
        meta = scheduler.build_connector_meta(sched_output)
        req_meta = meta.requests[0]
        self.assertEqual(req_meta.num_new_offload_blocks, 1)
        self.assertEqual(req_meta.offload_token_start, 96)
        self.assertEqual(req_meta.offload_num_tokens, 1)
        self.assertEqual(req_meta.num_tokens_after_step, 97)
        self.assertEqual(len(scheduler._request_trackers["r1"].allocated_block_ids_cpu), 7)

    def test_fused_decode_into_existing_partial_block(self, _mock):
        scheduler = self._make_scheduler(KV_OFFLOAD_MODE_FUSED_OVERLAP)
        scheduler._request_trackers["r1"] = RequestTracker(
            req_id="r1",
            allocated_block_ids_npu=[10, 11, 12, 13, 14, 15, 16],
            allocated_block_ids_cpu=[1, 2, 3, 4, 5, 6, 7],
        )
        scheduler._unfinished_requests["r1"] = (MagicMock(), [])
        scheduler._unfinished_request_ids.add("r1")

        sched_output = _make_sched_output(
            cached_req_ids=["r1"],
            cached_num_computed_tokens=[100],
            cached_new_block_ids=[[]],
            num_scheduled_tokens={"r1": 1},
        )
        meta = scheduler.build_connector_meta(sched_output)
        req_meta = meta.requests[0]
        self.assertEqual(req_meta.num_new_offload_blocks, 0)
        self.assertEqual(req_meta.offload_token_start, 100)
        self.assertEqual(req_meta.offload_num_tokens, 1)
        self.assertEqual(req_meta.num_tokens_after_step, 101)

    def test_fused_prefill_chunk_spans_multiple_blocks(self, _mock):
        scheduler = self._make_scheduler(KV_OFFLOAD_MODE_FUSED_OVERLAP)
        new_req = MagicMock()
        new_req.req_id = "r1"
        new_req.num_computed_tokens = 0
        new_req.block_ids = [[], [0, 1, 2]]

        sched_output = _make_sched_output(
            new_reqs=[new_req],
            num_scheduled_tokens={"r1": 40},
        )
        meta = scheduler.build_connector_meta(sched_output)
        req_meta = meta.requests[0]
        self.assertEqual(req_meta.num_new_offload_blocks, 3)
        self.assertEqual(req_meta.offload_token_start, 0)
        self.assertEqual(req_meta.offload_num_tokens, 40)
        self.assertEqual(req_meta.num_tokens_after_step, 40)
