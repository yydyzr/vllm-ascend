# SPDX-License-Identifier: Apache-2.0
"""Unit tests for nano top-k slot binding and tail geometry."""

import pytest

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.nano_topk_slots import (
    NANO_RING_TOKENS,
    NanoTopkSlotAllocator,
    nano_pool_capacity,
    nano_tail_device_token,
    nano_tail_geometry,
)


def test_nano_pool_capacity_includes_padding_rows():
    assert nano_pool_capacity(8) == 10


def test_nano_tail_geometry_skips_aligned_prefix():
    assert nano_tail_geometry(10240, 128) == (0, 0)
    assert nano_tail_geometry(0, 128) == (0, 0)


def test_nano_tail_geometry_keeps_incomplete_last_block():
    assert nano_tail_geometry(10367) == (127, 80)
    assert nano_tail_geometry(129) == (1, 1)


def test_nano_tail_device_token_matches_restore_dst():
    # Same numbers as sfa_kv_offload metadata: prefix=10240, pool=1, topk=8192.
    tail_tokens, tail_block = nano_tail_geometry(10367)
    assert (tail_tokens, tail_block) == (127, 80)
    hot = 8192
    row = hot + NANO_RING_TOKENS
    # restore: pool * (topk/128+2)*128 + hot + (80%2)*128
    assert nano_tail_device_token(1, tail_block, row, hot) == 1 * row + hot
    assert nano_tail_device_token(0, 81, row, hot) == hot + 128


def test_nano_tail_device_token_rejects_non_ring_row():
    with pytest.raises(ValueError, match="circular suffix"):
        nano_tail_device_token(1, 0, 256, 128)


def test_nano_slot_allocator_reuses_and_releases():
    allocator = NanoTopkSlotAllocator(2)
    first = allocator.bind("req-a")
    second = allocator.bind("req-b")
    assert {first, second} == {0, 1}
    assert allocator.bind("req-a") == first
    allocator.release("req-a")
    assert allocator.get("req-a") is None
    reused = allocator.bind("req-c")
    assert reused == first


def test_nano_slot_allocator_exhausts_capacity():
    allocator = NanoTopkSlotAllocator(1)
    allocator.bind("req-a")
    with pytest.raises(RuntimeError, match="exhausted"):
        allocator.bind("req-b")
