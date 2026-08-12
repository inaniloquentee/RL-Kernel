# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Two-GPU P2P NCCL reference check for issue #235.

Run with:

    torchrun --standalone --nproc-per-node=2 \
      scripts/ws2_p2p_nccl_attention_reference_check.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionCPMergedState,
    AttentionCPPartialState,
    AttentionParallelSpec,
    P2PNCCLAttentionCPCommunication,
)
from rl_engine.kernels.ops.pytorch.attention.cp_attention import (
    AttentionPartialState,
    DeterministicCPAttentionReferenceOp,
    merge_attention_partial_states,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--atol", type=float, default=2.0e-4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this check requires at least two visible CUDA devices")
    dist.init_process_group("nccl", init_method="env://")
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if world_size != 2:
            raise RuntimeError("this reference check requires exactly two NCCL ranks")
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        result = run_check(args, rank=rank, device=device)
        failures = torch.tensor(
            [0 if result["passed"] else 1],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(failures, op=dist.ReduceOp.SUM)
        result["global_failure_count"] = int(failures.item())
        reports: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(reports, result)
        if rank == 0:
            print(json.dumps({"ranks": reports}, indent=2, sort_keys=True))
        return 0 if int(failures.item()) == 0 else 1
    finally:
        dist.destroy_process_group()


def run_check(
    args: argparse.Namespace,
    *,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    if args.seq_len < 2 or args.seq_len % 2 != 0:
        raise ValueError("seq_len must be positive and divisible by CP=2")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shape_q = (args.batch, args.q_heads, args.seq_len, args.head_dim)
    shape_kv = (args.batch, args.kv_heads, args.seq_len, args.head_dim)
    q = torch.randn(shape_q, generator=generator, dtype=torch.bfloat16).to(device)
    k = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    v = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    owner_ranges = ((0, args.seq_len // 2), (args.seq_len // 2, args.seq_len))
    query_ranges = owner_ranges
    blocks: list[AttentionCPBlockMetadata] = []
    for owner, (owner_start, owner_end) in enumerate(owner_ranges):
        for start in range(owner_start, owner_end, args.chunk_size):
            blocks.append(
                AttentionCPBlockMetadata(
                    global_block_index=len(blocks),
                    kv_block_start=start,
                    kv_block_end=min(start + args.chunk_size, owner_end),
                    owner_cp_rank=owner,
                    owner_tp_rank=0,
                )
            )
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(
            tp_world_size=2,
            tp_rank=0,
            cp_world_size=2,
            cp_rank=rank,
        ),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=tuple(blocks),
        expected_kv_token_range=(0, args.seq_len),
        query_token_ranges=query_ranges,
    )
    reference = DeterministicCPAttentionReferenceOp()
    local_states: list[AttentionCPPartialState] = []
    for block in reversed(blocks):
        if block.owner_cp_rank != rank:
            continue
        state = reference.local_partial_state(
            q,
            k[:, :, block.kv_block_start : block.kv_block_end, :],
            v[:, :, block.kv_block_start : block.kv_block_end, :],
            q_start=0,
            k_start=block.kv_block_start,
            total_kv_len=args.seq_len,
            total_query_len=args.seq_len,
            causal=True,
        )
        local_states.append(
            AttentionCPPartialState(state.out, state.lse, block)
        )

    communication = P2PNCCLAttentionCPCommunication()
    gathered = communication.all_gather_partial_states(tuple(local_states), plan)
    merged = merge_attention_partial_states(
        [
            AttentionPartialState(
                state.out,
                state.lse,
                state.block.kv_block_start,
                state.block.kv_block_end,
            )
            for state in gathered
        ]
    )
    local = communication.reduce_scatter_merged_state(
        AttentionCPMergedState(merged.out, merged.lse),
        plan,
    )
    full_out, full_lse = reference.forward_fp32_with_lse(q, k, v, causal=True)
    start, end = query_ranges[rank]
    out_max_abs = float((local.out - full_out[:, :, start:end, :]).abs().max().item())
    lse_max_abs = float((local.lse - full_lse[:, :, start:end]).abs().max().item())
    expected_indices = list(range(len(blocks)))
    gathered_indices = [state.block.global_block_index for state in gathered]
    passed = (
        gathered_indices == expected_indices
        and out_max_abs <= args.atol
        and lse_max_abs <= args.atol
    )
    return {
        "rank": rank,
        "device": str(device),
        "dtype": "bf16",
        "accum_dtype": "fp32",
        "downcast_at": "final_write",
        "transport": "p2p_nccl_reference",
        "query_range": [start, end],
        "gathered_block_indices": gathered_indices,
        "out_max_abs": out_max_abs,
        "lse_max_abs": lse_max_abs,
        "atol": args.atol,
        "passed": passed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
