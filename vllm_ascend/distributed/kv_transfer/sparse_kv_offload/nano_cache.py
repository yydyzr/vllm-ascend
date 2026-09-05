# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side copy descriptors for nano's bounded HBM cache.

Descriptor storage belongs to the caller and outlives graph capture/replay.
Only tensor operations run in the descriptor builders; registered host and
device base addresses are fixed for the lifetime of a layer's cache.
"""

import torch

TAIL_BLOCKS = 2
KV_COMPONENTS = 2


class CopyDescriptors:
    def __init__(self, capacity: int, device: torch.device):
        self.sources = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.destinations = torch.zeros_like(self.sources)
        self.lengths = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.count = torch.full((1,), capacity, dtype=torch.int32, device=device)

    def args(self):
        return self.sources, self.destinations, self.lengths, self.count


def prepare_tail_copy(
    descriptors,
    batch,
    *,
    block_size,
    hot_tokens,
    source_bases,
    destination_bases,
    token_bytes,
    source_block_capacity,
    active=None,
):
    """Up to two block-contiguous copies per request for each KV component."""
    requests, columns = batch.source_block_table.shape
    spans = requests * TAIL_BLOCKS
    if descriptors.sources.numel() != spans * KV_COMPONENTS:
        raise ValueError("Tail descriptor capacity must match the captured request capacity")
    if columns == 0 or source_block_capacity <= 0:
        raise ValueError("Tail copy requires a nonempty registered main CPU cache")
    parts = torch.arange(TAIL_BLOCKS, device=batch.seq_lens.device, dtype=torch.int64)
    logical = batch.offload_lens.to(torch.int64)[:, None] // block_size + parts
    # The unused second block may be beyond the logical table at max_model_len.
    physical = batch.source_block_table.gather(1, logical.clamp(0, columns - 1)).to(torch.int64)
    lengths = (batch.seq_lens[:, None] - batch.offload_lens[:, None] - parts * block_size).clamp(0, block_size)
    valid = (logical >= 0) & (logical < columns) & (physical >= 0) & (physical < source_block_capacity)
    if active is not None:
        valid = valid & active[:, None]
    lengths = torch.where(valid, lengths, 0).flatten()
    source_slots = (physical.clamp(0, source_block_capacity - 1) * block_size).flatten()
    destination_slots = (
        batch.pool_entries.to(torch.int64)[:, None] * (hot_tokens + TAIL_BLOCKS * block_size)
        + hot_tokens
        + logical.remainder(TAIL_BLOCKS) * block_size
    ).flatten()
    for component in range(KV_COMPONENTS):
        section = slice(component * spans, (component + 1) * spans)
        descriptors.sources[section].copy_(source_bases[component] + source_slots * token_bytes[component])
        descriptors.destinations[section].copy_(
            destination_bases[component] + destination_slots * token_bytes[component]
        )
        descriptors.lengths[section].copy_(lengths * token_bytes[component])


def prepare_resident_gather(
    descriptors,
    *,
    block_table,
    selected,
    token_to_request,
    visible_lengths,
    block_size,
    hot_tokens,
    source_bases,
    destination_bases,
    token_bytes,
    source_block_capacity,
):
    """Gather one causal TopK set per query into separate bounded hot rows.

    This eager fallback overwrites nano resident rows. Its caller invalidates
    LIM residency before returning to the fused path. It never uses the full
    NPU main cache, including for short requests and heterogeneous query counts.
    """
    rows, topk = selected.shape
    if descriptors.sources.numel() < KV_COMPONENTS * rows * topk:
        raise ValueError("Resident gather exceeds its descriptor capacity")
    if topk > hot_tokens:
        raise ValueError("Resident gather exceeds the hot-cache row")
    device = selected.device
    columns = block_table.shape[1]
    selected = selected.to(torch.int64)
    logical = selected.clamp_min(0) // block_size
    tables = block_table.index_select(0, token_to_request.to(torch.int64))
    physical = tables.gather(1, logical.clamp(max=columns - 1)).to(torch.int64)
    valid = (selected >= 0) & (selected < visible_lengths[:, None]) & (logical < columns)
    valid = valid & (physical >= 0) & (physical < source_block_capacity)
    source_slots = physical.clamp(0, source_block_capacity - 1) * block_size + selected.clamp_min(0) % block_size
    slots = torch.arange(topk, dtype=torch.int64, device=device).expand(rows, -1)
    row_stride = hot_tokens + TAIL_BLOCKS * block_size
    destinations = torch.arange(rows, dtype=torch.int64, device=device)[:, None] * row_stride + slots
    count = rows * topk
    for component in range(KV_COMPONENTS):
        section = slice(component * count, (component + 1) * count)
        descriptors.sources[section].copy_(source_bases[component] + source_slots.flatten() * token_bytes[component])
        descriptors.destinations[section].copy_(
            destination_bases[component] + destinations.flatten() * token_bytes[component]
        )
        descriptors.lengths[section].copy_(valid.flatten().to(torch.int32) * token_bytes[component])
    descriptors.count.fill_(KV_COMPONENTS * count)
    resident_slots = torch.where(valid, slots, -1).to(torch.int32)
    pages = hot_tokens // block_size + TAIL_BLOCKS
    resident_table = torch.arange(rows, dtype=torch.int32, device=device)[:, None] * pages + torch.arange(
        (topk + block_size - 1) // block_size, dtype=torch.int32, device=device
    )
    return resident_slots, resident_table
