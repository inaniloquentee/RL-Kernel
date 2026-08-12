# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""PR7 FlashInfer RoPE-fused paged attention validation entry point.

The default dry-run mode is CI/local friendly: it builds the FlashInfer page plan
and provenance without importing FlashInfer or requiring CUDA.  On a CUDA host
with FlashInfer installed, omit ``--dry-run`` to run the opt-in PR7 candidate and
compare it with the PR6 full logical KV reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.ops.cuda.attention.cp_comm import (  # noqa: E402
    AttentionCPCommunicationPlan,
    AttentionParallelSpec,
)
from rl_engine.kernels.ops.cuda.attention.flashinfer_paged_attention import (  # noqa: E402
    FlashInferPagedAttentionConfig,
    FlashInferQwen3PagedAttentionOp,
    FlashInferRoPEFusionConfig,
    FlashInferSplitKVPolicy,
    build_flashinfer_paged_kv_plan,
)
from rl_engine.testing.attention_comparison import (  # noqa: E402
    DecodeAttentionInputs,
    DecodeKVCacheMetadata,
    run_decode_full_prefill_reference,
)


def main() -> int:
    args = _parse_args()
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    inputs = _make_inputs(args, device)
    config = _make_config(args)
    config.validate(head_dim=args.head_dim, query_len=args.query_len)
    plan = build_flashinfer_paged_kv_plan(
        inputs.metadata,
        batch_size=args.batch_size,
        query_len=args.query_len,
        cache_capacity=inputs.k_cache.size(2),
        device=device,
    )
    report: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "executed",
        "pr": "PR7",
        "target": "Qwen3-8B TP=2 CP=2 BF16 attention candidate",
        "mode": config.mode,
        "device": str(device),
        "shape": {
            "batch_size": args.batch_size,
            "query_len": args.query_len,
            "kv_seq_len": args.kv_seq_len,
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
        },
        "rope": config.rope.provenance(args.head_dim),
        "split_kv": {
            **config.split_kv.to_dict(),
            "provenance_status": "requested_only_dry_run",
            "actual_plan_required_for_strict_pass": config.require_batch_invariant,
            "requested_execution_plans": [
                config.split_kv.resolve(
                    int(seq_len),
                    backend="flashinfer_dry_run_requested_only",
                ).to_dict()
                for seq_len in plan.kv_seq_lens.tolist()
            ],
        },
        "communication": config.cp_comm_plan.provenance()
        | {"cp_comm_required": config.require_cp_comm},
        "paged_kv_plan": plan.provenance(),
        "tests_expected": [
            "FlashInfer ROPE_LLAMA vs NativeRoPEOp + full logical KV reference",
            "split-K disabled/fixed policy drift",
            "batch composition/position invariant sweep",
            "attention-domain LSE export drift",
            "CP=2 TP=2 custom CUDA AG/RS communication interface wiring; real ops are future work",
        ],
    }
    if args.dry_run:
        _emit(report, json_output=args.json)
        return 0

    if device.type != "cuda":
        raise SystemExit("non-dry-run PR7 validation requires --device cuda")
    op = FlashInferQwen3PagedAttentionOp()
    candidate = op(
        inputs.q,
        inputs.k_cache,
        inputs.v_cache,
        inputs.metadata,
        config=config,
    )
    reference_inputs = replace(
        inputs,
        metadata=replace(
            inputs.metadata,
            q_rope_state="pre_rope",
            k_cache_rope_state="pre_rope",
        ),
    )
    reference = run_decode_full_prefill_reference(reference_inputs)
    out_diff = (candidate.out.float() - reference.out.float()).abs()
    lse_diff = (candidate.lse.float() - reference.lse.float()).abs()
    report["candidate_provenance"] = candidate.provenance
    report["drift"] = {
        "out_max_abs": float(out_diff.max().item()),
        "lse_max_abs": float(lse_diff.max().item()),
    }
    if config.require_batch_invariant:
        report["batch_invariant_sweep"] = _run_batch_invariance_sweep(
            op,
            inputs,
            candidate,
            config,
        )
    _emit(report, json_output=args.json)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="decode")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--kv-seq-len", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tp-world-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--cp-world-size", type=int, default=2)
    parser.add_argument("--cp-rank", type=int, default=0)
    parser.add_argument(
        "--cp-comm-backend",
        choices=["cuda_ag_rs", "local_debug"],
        default="cuda_ag_rs",
    )
    parser.add_argument("--require-cp-comm", action="store_true")
    parser.add_argument("--fixed-split-size", type=int, default=None)
    parser.add_argument(
        "--split-kv-policy",
        choices=["disabled", "fixed", "auto"],
        default="disabled",
    )
    parser.add_argument(
        "--require-batch-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _make_config(args: argparse.Namespace) -> FlashInferPagedAttentionConfig:
    if args.split_kv_policy == "disabled":
        split_kv = FlashInferSplitKVPolicy.disabled()
    elif args.split_kv_policy == "fixed":
        if args.fixed_split_size is None:
            raise SystemExit("--split-kv-policy fixed requires --fixed-split-size")
        split_kv = FlashInferSplitKVPolicy.fixed(args.fixed_split_size)
    else:
        split_kv = FlashInferSplitKVPolicy.auto()
    return FlashInferPagedAttentionConfig(
        mode=args.mode,
        workspace_size_bytes=128 * 1024 * 1024,
        require_batch_invariant=args.require_batch_invariant,
        rope=FlashInferRoPEFusionConfig(
            rope_theta=1_000_000.0,
            rope_scale=1.0,
            rotary_dim=args.head_dim,
            q_rope_state="pre_rope",
            k_cache_rope_state="pre_rope",
        ),
        split_kv=split_kv,
        cp_comm_plan=AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(
                tp_world_size=args.tp_world_size,
                tp_rank=args.tp_rank,
                cp_world_size=args.cp_world_size,
                cp_rank=args.cp_rank,
            ),
            backend=args.cp_comm_backend,
            status="interface_only",
        ),
        require_cp_comm=args.require_cp_comm,
    )


def _make_inputs(args: argparse.Namespace, device: torch.device) -> DecodeAttentionInputs:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.kv_seq_len % args.page_size != 0:
        raise SystemExit("--kv-seq-len must be divisible by --page-size for this scaffold")
    if args.mode == "decode" and args.query_len != 1:
        raise SystemExit("--mode decode requires --query-len 1")
    generator = torch.Generator(device=device).manual_seed(2357)
    q = torch.randn(
        args.batch_size,
        args.q_heads,
        args.query_len,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    k_cache = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.kv_seq_len,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.kv_seq_len,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    page_count = args.kv_seq_len // args.page_size
    block_table = torch.arange(page_count, device=device, dtype=torch.long).repeat(
        args.batch_size,
        1,
    )
    positions = torch.arange(args.kv_seq_len, device=device, dtype=torch.long).repeat(
        args.batch_size,
        1,
    )
    query_start = args.kv_seq_len - args.query_len
    query_positions = torch.arange(
        query_start,
        args.kv_seq_len,
        device=device,
        dtype=torch.long,
    ).repeat(args.batch_size, 1)
    metadata = DecodeKVCacheMetadata(
        cache_position=query_positions.clone(),
        kv_seq_lens=torch.full(
            (args.batch_size,),
            args.kv_seq_len,
            device=device,
            dtype=torch.long,
        ),
        block_table=block_table,
        global_token_positions=positions,
        query_position_ids=query_positions.clone(),
        key_position_ids=positions.clone(),
        page_size=args.page_size,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
        cp_world_size=args.cp_world_size,
    )
    return DecodeAttentionInputs(q=q, k_cache=k_cache, v_cache=v_cache, metadata=metadata)


def _run_batch_invariance_sweep(
    op: FlashInferQwen3PagedAttentionOp,
    inputs: DecodeAttentionInputs,
    batch_result: Any,
    config: FlashInferPagedAttentionConfig,
) -> dict[str, Any]:
    rows = []
    max_out = 0.0
    max_lse = 0.0
    for batch_index in range(inputs.q.size(0)):
        single = _select_batch_row(inputs, batch_index)
        single_result = op(
            single.q,
            single.k_cache,
            single.v_cache,
            single.metadata,
            config=config,
        )
        out_diff = (
            single_result.out.float() - batch_result.out[batch_index : batch_index + 1].float()
        ).abs()
        lse_diff = (
            single_result.lse.float() - batch_result.lse[batch_index : batch_index + 1].float()
        ).abs()
        row_out = float(out_diff.max().item())
        row_lse = float(lse_diff.max().item())
        max_out = max(max_out, row_out)
        max_lse = max(max_lse, row_lse)
        rows.append({"batch_index": batch_index, "out_max_abs": row_out, "lse_max_abs": row_lse})
    return {
        "method": "single_row_vs_same_row_inside_batch",
        "row_count": len(rows),
        "out_max_abs": max_out,
        "lse_max_abs": max_lse,
        "rows": rows,
    }


def _select_batch_row(inputs: DecodeAttentionInputs, batch_index: int) -> DecodeAttentionInputs:
    metadata = inputs.metadata
    cp_block_owners = (
        None
        if metadata.cp_block_owners is None
        else metadata.cp_block_owners[batch_index : batch_index + 1]
    )
    selected_metadata = DecodeKVCacheMetadata(
        cache_position=metadata.cache_position[batch_index : batch_index + 1],
        kv_seq_lens=metadata.kv_seq_lens[batch_index : batch_index + 1],
        block_table=metadata.block_table[batch_index : batch_index + 1],
        global_token_positions=metadata.global_token_positions[batch_index : batch_index + 1],
        query_position_ids=metadata.query_position_ids[batch_index : batch_index + 1],
        key_position_ids=metadata.key_position_ids[batch_index : batch_index + 1],
        page_size=metadata.page_size,
        prefix_cache_key=metadata.prefix_cache_key,
        prefix_cache_enabled=metadata.prefix_cache_enabled,
        q_rope_state=metadata.q_rope_state,
        k_cache_rope_state=metadata.k_cache_rope_state,
        cp_block_owners=cp_block_owners,
        cp_world_size=metadata.cp_world_size,
    )
    return replace(
        inputs,
        q=inputs.q[batch_index : batch_index + 1],
        k_cache=inputs.k_cache[batch_index : batch_index + 1],
        v_cache=inputs.v_cache[batch_index : batch_index + 1],
        k_new=(None if inputs.k_new is None else inputs.k_new[batch_index : batch_index + 1]),
        v_new=(None if inputs.v_new is None else inputs.v_new[batch_index : batch_index + 1]),
        metadata=selected_metadata,
    )


def _emit(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"PR7 FlashInfer check: {report['status']}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
