# WS2 PR7 Fused Attention Backend Alignment

Status: PR7 scaffold for #235

## Goal

PR7 is the production-backend alignment layer after PR2/PR3/PR6 have defined
the reference semantics.  It evaluates fused prefill/decode candidates while
keeping RL-Kernel's reference contract as the source of truth.

The full PR7 plan has two backend lanes:

| Lane | Role | Status in this scaffold |
| --- | --- | --- |
| Training full-prefill candidate | Evaluate TE public `DotProductAttention` / fused attention for training-style full prefill | Planned; not implemented here |
| Rollout paged prefill/decode candidate | Evaluate FlashInfer paged attention with Qwen3 RoPE fusion, split-KV policy, LSE export, and batch-invariant sweep | Implemented as opt-in scaffold |

This file therefore documents both the original PR7 requirements and the new
FlashInfer/RoPE/split-KV/batch-invariant additions.  It should not be read as a
claim that FlashInfer or TE replaces the deterministic reference.

## Source Of Truth

The correctness source remains:

```text
AttentionContract
PR2 single-GPU full/chunked/paged attention reference
PR3 deterministic CP reference and global_block_index merge
PR6 decode paged-KV replay vs full logical KV reference
attention-domain lse
dlogp drift report
```

PR7 backends are candidates.  A candidate can be promoted only if it exports or
reconstructs the states required by the contract and passes the drift gates.

## Original PR7 Requirements

| Requirement | PR7 rule |
| --- | --- |
| Backend selection | Record `requested_backend`, `actual_backend`, `fallback`, and `fallback_reason`; no silent fallback. |
| Training path | Training-side forward is full-prefill / teacher-forcing.  TE may be evaluated through its public `DotProductAttention` path, not by redefining RL-Kernel semantics. |
| Rollout path | Rollout-side forward is paged KV prefill/decode.  FlashInfer is a rollout candidate, not a training backward backend. |
| LSE | Candidate must return attention-domain `lse` shaped as `[B, Hq, Sq]`. |
| CP/TP communication | PR7 exposes the `TP=2, CP=2` custom CUDA AG/RS communication interface.  Real communication kernels are future work and fail-closed in this scaffold. |
| CP order | CP correctness remains PR3's `global_block_index` merge.  A black-box backend that only returns final output must be validated against PR3/PR6; it cannot define CP merge order. |
| Paged KV | Page table, `cache_position`, `query_position_ids`, `key_position_ids`, and logical token order must be validated before execution. |
| Precision | Record accumulation/downcast policy.  BF16 output is acceptable only after FP32/reference drift is reported. |
| Backward | Training backward belongs to PR8.  PR7 forward results must not claim `dq/dk/dv` alignment. |

## Current Implementation

Implemented files:

```text
rl_engine/kernels/ops/cuda/attention/cp_comm.py
rl_engine/kernels/ops/cuda/attention/flashinfer_paged_attention.py
tests/test_flashinfer_pr7_attention.py
scripts/ws2_pr7_flashinfer_attention_check.py
```

The FlashInfer adapter is opt-in and lazy-imported.  Importing RL-Kernel does not
require FlashInfer.  Real FlashInfer execution requires CUDA tensors; CPU tests
use a fake wrapper only to validate parameter binding and provenance.

Current code path:

```text
FlashInferQwen3PagedAttentionOp.forward(...)
  validate q / k_cache / v_cache
  validate Qwen3 RoPE fusion config
  validate split-KV policy
  validate CP=2 / TP=2 AG/RS communication interface contract
  build_flashinfer_paged_kv_plan(...)
    validate page bounds
    validate block_table/global_token_positions logical order
    validate key_position_ids
  materialize_flashinfer_paged_kv_cache(...)
  flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper
    or flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper
  wrapper.plan(..., pos_encoding_mode="ROPE_LLAMA", rope_theta=1e6, rope_scale=1.0, ...)
  wrapper.run_return_lse(...)
  restore out -> [B, Hq, Sq, D]
  restore lse -> [B, Hq, Sq]
```

## CP/TP Communication Interface

Issue #235 targets `Qwen3-8B, TP=2, CP=2, BF16`.  PR7 therefore surfaces the
distributed attention boundary even though the real custom CUDA communication
operators are not implemented in this scaffold.

The exposed interface is:

```text
AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
AttentionCPCommunicationPlan(backend="cuda_ag_rs", status="interface_only")
AttentionCPPartialState(out, lse, AttentionCPBlockMetadata(...))
CUDAAGRSAttentionCPCommunication.all_gather_partial_states(...)
CUDAAGRSAttentionCPCommunication.reduce_scatter_merged_state(...)
sort_attention_cp_partial_states(..., plan=...)
```

The future custom CUDA AG/RS operators must move attention-domain partial states
with compute and communication kept decoupled:

```text
per-rank local attention over rank-owned KV blocks
  -> AttentionCPPartialState(
       out: [B, Hq, Sq, D],
       lse: [B, Hq, Sq] fp32,
       global_block_index,
       kv_block_start / kv_block_end,
       owner_cp_rank / owner_tp_rank,
     )
  -> custom CUDA AG communication operator
  -> sort by global_block_index
  -> PR3 FP32 online-softmax merge
  -> custom CUDA RS communication operator
```

In this PR, `CUDAAGRSAttentionCPCommunication.all_gather_partial_states(...)`
and `CUDAAGRSAttentionCPCommunication.reduce_scatter_merged_state(...)` are
fail-closed placeholders and raise `AttentionCPCommunicationUnavailable`.
Likewise, `FlashInferPagedAttentionConfig(require_cp_comm=True)` raises before
execution.  This prevents the FlashInfer local paged-attention candidate from
being mistaken for a complete CP communication implementation.

Ordering is part of the interface:

```text
merge_key = AttentionCPBlockMetadata.global_block_index
accum_dtype = fp32
required_state = (out, lse)
communication_pattern = ag_rs
compute_communication = decoupled
duplicate global_block_index -> error
```

FlashInfer split-KV is backend-local KV reduction inside one rank.  It is not
the CP merge and does not define cross-rank order.  The real CP path must still
return ordered partial states before calling the PR3 merge rule.

## FlashInfer RoPE Fusion

The implemented fused boundary is:

```text
pre-RoPE Q, pre-RoPE K cache, V
  -> FlashInfer paged attention with pos_encoding_mode="ROPE_LLAMA"
  -> out, attention-domain lse
```

The accepted Qwen3 settings are locked to:

```text
pos_encoding_mode = "ROPE_LLAMA"
rope_theta        = 1_000_000.0
rope_scale        = 1.0
rotary_dim        = head_dim
layout            = Qwen3 rotate-half / non-interleaved
```

The adapter rejects post-RoPE Q/K for this path.  That avoids silently rotating
the same tensor twice.  If a later rollout path stores post-RoPE K and rotates
only Q in a fused decode kernel, it must be represented as a separate
materialization/capability.

## Split-KV Policy

PR7 exposes split-KV as a contract/provenance knob instead of inheriting
backend defaults:

| Policy | FlashInfer plan kwargs | Batch-invariant status |
| --- | --- | --- |
| `disabled` | `disable_split_kv=True` | strict candidate |
| `fixed:<N>` | `fixed_split_size=N`, `disable_split_kv=False` | candidate; must pass drift sweep |
| `auto` | `disable_split_kv=False` | rejected when batch invariance is required |

The shared WS2 contract now treats Split-KV as a first-class semantic field. PR7 still
must export the backend's actual token boundaries; a requested FlashInfer knob alone is
not sufficient evidence of train/rollout equivalence.
Once PR1/PR4 add a first-class field, the PR7 provenance can be wired into that
field without changing the backend adapter.

## Batch-Invariant Validation

PR7 distinguishes "configured for batch invariance" from "proven batch
invariant":

```text
configured:
  split-KV disabled or fixed
  no auto backend scheduling in the contract
  page/order metadata validated

proven:
  same sample alone vs same sample inside a batch has zero or tolerated drift
  out and lse both reported
  real CUDA/H-card run, not fake wrapper
```

The validation script reports:

```text
batch_invariant_sweep.method = single_row_vs_same_row_inside_batch
batch_invariant_sweep.out_max_abs
batch_invariant_sweep.lse_max_abs
```

FlashInfer is not declared batch-invariant by default.  It becomes an accepted
candidate only if the real-hardware sweep passes under the selected split-KV
policy.

## TE Relationship

The TE lane is still part of the full PR7 plan, but it is not implemented in
this scaffold.

| TE use | PR7 position |
| --- | --- |
| TE CP correction helpers | Already used by PR2/PR3/PR6 as optional merge oracle. |
| TE full fused attention | Future PR7 training full-prefill candidate through public `DotProductAttention`. |
| TE internal CP black box | Not a source of RL-Kernel `global_block_index` order unless it exposes compatible partial states or passes reference drift gates. |
| TE backward | PR8 only, and only if compatible saved forward/backward state is available. |

FlashInfer and TE can coexist in PR7:

```text
training candidate:
  TE DotProductAttention full prefill

rollout candidate:
  FlashInfer ROPE_LLAMA paged prefill/decode

shared gate:
  compare both against RL-Kernel reference states and selected-logprob drift
```

## Relation To Existing PRs

| PR | Relationship |
| --- | --- |
| PR1 | Carries the strict Split-KV policy and capability requirements; runtime adapters still export actual boundaries and fallback provenance. |
| PR2 | Provides full/chunked/paged single-GPU references and fused-like RoPE attribution. PR7 uses the same semantic boundary, but calls a real backend candidate. |
| PR3 | Owns CP=1/2 deterministic semantics and `global_block_index` merge. PR7 does not replace this. |
| PR4 / #263 | Records actual backend and split-KV provenance. PR7 matches that policy and can later wire into PR4 fields. |
| PR5 | Should run the cross matrix once PR7 backends exist: full/chunked/paged, split-KV disabled/fixed, batch shape, TP/CP, dtype. |
| PR6 / #260 | Provides decode paged-KV replay and full logical KV reference. PR7 consumes PR6-style metadata and validates the same page/position identity before FlashInfer execution. |
| PR8 | Owns training backward alignment. PR7 forward-only results are not backward claims. |

## Conflict Check

| Topic | Possible conflict | PR7 resolution |
| --- | --- | --- |
| RoPE fusion vs PR2/PR6 post-RoPE references | A fused backend might hide post-RoPE Q/K state. | PR7 records `rope_fusion_boundary`, rejects post-RoPE inputs for this path, and compares final `out/lse` against references. |
| FlashInfer split-KV vs PR3 CP merge order | FlashInfer internal split-KV order is not PR3 `global_block_index`. | Treat split-KV as backend-local reduction.  CP merge remains PR3; FlashInfer must pass drift gates and cannot define CP order. |
| CP=2/TP=2 target vs missing communication operator | The issue requires distributed semantics, but the custom CUDA AG/RS communication operators are not ready. | PR7 exposes `AttentionCPCommunicationPlan` / `AttentionCPPartialState` / `CUDAAGRSAttentionCPCommunication` as interface-only, and fails if execution is required. |
| Batch-invariant claim vs backend heuristics | Auto split-KV may depend on batch composition/runtime scheduling. | Reject `auto` when `require_batch_invariant=True`; report disabled/fixed policy explicitly. |
| TE training lane vs FlashInfer rollout lane | Two libraries may have different materialization boundaries. | Both bind to the same RL-Kernel semantic contract and are judged by shared `out/lse/dlogp` reports. |
| Requested policy vs actual plan | A requested fixed/max-splits knob may not equal the runtime token boundaries. | Require actual runtime plan callbacks in strict fixed mode; fail closed when unavailable. |
| Decode vs training full prefill | Training does not own a persistent paged KV cache. | PR7 compares rollout paged KV decode/prefill to training-style full logical KV reference, not to a nonexistent training decode cache. |

No direct conflict was found between the three new points and the original PR7
plan.  The important restriction is that FlashInfer split-KV/RoPE fusion remains
a candidate path until real CUDA/H-card drift reports prove it.

## Validation

CPU-safe structural tests:

```bash
pytest tests/test_flashinfer_pr7_attention.py -q
```

Dry-run plan/provenance checks without CUDA or FlashInfer:

```bash
python scripts/ws2_pr7_flashinfer_attention_check.py --dry-run --json
python scripts/ws2_pr7_flashinfer_attention_check.py --dry-run --mode prefill --query-len 16 --json
```

CUDA/H-card validation once hardware is available:

```bash
python scripts/ws2_pr7_flashinfer_attention_check.py \
  --no-dry-run \
  --device cuda \
  --mode decode \
  --split-kv-policy disabled \
  --json

python scripts/ws2_pr7_flashinfer_attention_check.py \
  --no-dry-run \
  --device cuda \
  --mode prefill \
  --query-len 16 \
  --split-kv-policy fixed \
  --fixed-split-size 4 \
  --json
```

The CUDA report must include:

```text
out_max_abs
lse_max_abs
batch_invariant_sweep.out_max_abs
batch_invariant_sweep.lse_max_abs
rope_fusion_boundary
split_kv_policy
actual_backend
fallback / fallback_reason
```

## Non-Claims

This scaffold does not enable FlashInfer by default, does not implement the TE
training lane, does not implement the custom CUDA AG/RS communication operators,
does not replace PR3 CP merge, does not prove real H-card batch-invariance
locally, and does not implement training backward.
# P2P NCCL reference and strict arithmetic provenance

The existing `CUDAAGRSAttentionCPCommunication` interface is preserved and
remains fail-closed until the self-owned CUDA AG/RS kernels exist.  For GPU
validation, `P2PNCCLAttentionCPCommunication` implements the same partial-state
protocol with `torch.distributed.batch_isend_irecv` on an NCCL process group.
It requires an authoritative manifest containing every logical KV block,
token range, CP owner, TP owner, and query scatter range.  Missing blocks,
gaps, overlaps, wrong owners, incomplete gathers, non-NCCL groups, and CPU
tensors are rejected before merge.

Strict FlashInfer validation also requires runtime callbacks for both:

- the actual Split-KV token boundaries for every request; and
- arithmetic provenance declaring FP32 accumulation, FP32 LSE, and downcast
  only at the final output write.

Requested knobs, maximum split counts, or labels such as
`flashinfer_internal` are not accepted as proof.

Run the two-GPU transport check with:

```bash
torchrun --standalone --nproc-per-node=2 \
  scripts/ws2_p2p_nccl_attention_reference_check.py
```
