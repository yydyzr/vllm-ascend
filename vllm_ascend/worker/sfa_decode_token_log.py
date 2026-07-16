"""Per-decode-step sampled token logging for SFA graph vs eager comparison."""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs


def _format_token_rows(
    req_ids: list[str],
    sampled_token_ids: list[list[int]],
    spec_token_ids: list[list[int]] | None,
) -> str:
    parts: list[str] = []
    for idx, req_id in enumerate(req_ids):
        sampled = sampled_token_ids[idx] if idx < len(sampled_token_ids) else []
        parts.append(f"{req_id}:sampled={sampled}")
        if spec_token_ids is not None and idx < len(spec_token_ids):
            spec = spec_token_ids[idx]
            if spec:
                parts.append(f"{req_id}:spec={spec}")
    return "{" + ", ".join(parts) + "}"


def maybe_log_decode_step_tokens(
    *,
    step: int,
    req_ids: list[str],
    sampled_token_ids: list[list[int]],
    spec_token_ids: list[list[int]] | None,
    total_num_scheduled_tokens: int,
    cudagraph_mode: Any,
    enforce_eager: bool,
    cudagraph_stats: Any,
    batch_desc: Any,
    tp_rank: int,
) -> None:
    if not envs.VLLM_ASCEND_SFA_DECODE_TOKEN_LOG:
        return
    if not req_ids:
        return

    run_mode = "eager" if enforce_eager else "graph"
    stats_repr = repr(cudagraph_stats) if cudagraph_stats is not None else "none"
    logger.warning(
        "[sfa_decode_token] step=%s run=%s cudagraph_mode=%s enforce_eager=%s "
        "pid=%s tp_rank=%s scheduled_tokens=%s batch_desc=%s cudagraph_stats=%s "
        "tokens=%s",
        step,
        run_mode,
        cudagraph_mode,
        enforce_eager,
        os.getpid(),
        tp_rank,
        total_num_scheduled_tokens,
        batch_desc,
        stats_repr,
        _format_token_rows(req_ids, sampled_token_ids, spec_token_ids),
    )
