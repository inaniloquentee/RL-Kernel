<!--
Draft source: #83 roadmap, WS1 #108, WS2 #111 / #207 / #241, the DeepSeek-V4
Flash model card, the DeepSeek-V4 hash-routing discussion, and vLLM's fused MoE
modular-kernel design. This issue is a model-level specialization of
docs/design/model-level-train-rollout-consistency-plan.md.
-->

# [Roadmap] DeepSeek-V4 Flash MoE Train-Inference Consistency

## Context

DeepSeek-V4 Flash is a Mixture-of-Experts model: 284B total parameters, 13B
activated parameters, 1M context, and FP4 + FP8 mixed precision. The model card
also says post-training uses a two-stage path: independent cultivation of
domain-specific experts through SFT and RL with GRPO, then unified consolidation
via on-policy distillation. That means training and rollout are not guaranteed
to materialize the same graph.

This roadmap is about the MoE subgraph only. Hybrid attention, mHC, and the
rest of the transformer stack are upstream/downstream boundaries unless they
change the MoE inputs or outputs.

The key reason MoE is not a single-op problem is that vLLM's fused MoE modular
kernel already splits the path into separate semantic components:

- `FusedMoEPrepareAndFinalizeModular`
- `FusedMoEExpertsModular`
- `TopKWeightAndReduce`

The input activation format can also be contiguous or batched, depending on the
All2All dispatch. Training may decompose the same semantics into router,
dispatch, expert, and combine ops. The contract must therefore live above the
kernel boundary.

## What Can Drift

| Boundary | Typical drift source | Solution surface |
| --- | --- | --- |
| Router | hash routing on the first 3 MoE layers, top-k, expert bias, router dtype, load-balancing policy | deterministic router contract + provenance |
| Dispatch | contiguous vs batched activation layout, token packing, all2all order, `expert_num_tokens` | explicit materialization map + `ShardingSpec` |
| Expert compute | permute, W1, act + mul, quantization, W2, unpermute | reference expert contract + grouped GEMM determinism |
| Final combine | top-k weight application, reduction order, accumulator dtype, downcast point | `ReductionSpec` with FP32 merge and explicit downcast |
| Distributed layout | EP / TP / maybe CP around the MoE boundary, batch size, padding, pack order | cross-config `dlogp` audit + per-rank drift report |

The final floating-point reduction point is the weighted expert combine. Above
it are batch, pack, padding, EP, TP, CP, and communication provenance.

## Shared Contract

A run is comparable only when it shares:

- the same checkpoint and tokenizer;
- the same input token ids and masks;
- the same hidden-state boundary feeding the MoE block;
- the same routing metadata: `n_hash_layers`, `topk`, `router_score_function`,
  `router_dtype`, `expert_bias`, `load_balancing`, `activation_format`,
  `topk_id_dtype`;
- the same reduction metadata: `accum_dtype`, `order`, `downcast_point`,
  `combine_backend`;
- the same fallback rule: no silent fallback.

If the upstream attention / embedding / LM-head boundary is already dirty, MoE
does not get blamed for it. This roadmap owns the MoE subgraph and its immediate
interfaces.

R3 routing replay is optional. It is useful when rollout can export routed
experts, because it reduces one large source of mismatch and makes audit cleaner.
But this roadmap does not require R3 as a prerequisite, and R3 does not define
the MoE semantic contract. The contract must still stand without routing replay.

## Proposed Issue Ladder

| Phase | Proposed issue title | Focus | Done when |
| --- | --- | --- | --- |
| 0 | `[DSV4-MOE] semantic contract and model profile` | model profile, materialization map, provenance schema | model / route / layout can be reported and unsupported configs fail closed |
| 1 | `[DSV4-MOE] router and hash-routing parity` | first 3 hash-routed layers, top-k, expert bias, router dtype | router outputs match before dispatch |
| 2 | `[DSV4-MOE] fused vs unfused MoE materialization` | training graph vs rollout graph | fused and unfused paths map to one semantic contract |
| 3 | `[DSV4-MOE] expert compute and reduction-order consistency` | permute, grouped GEMM, quantization, top-k reduce | final expert outputs match under the same reduction contract |
| 4 | `[DSV4-MOE] EP/TP dispatch and cross-config dlogp` | all2all, pack/padding, batch shape, per-rank drift | rollout vs training dlogp is aligned on supported layouts |
| 5 | `[DSV4-MOE] vime integration and benchmark gate` | adapter, audit/strict mode, smoke tests, CI | one-command smoke runs with clear fallback and report |

## Phase 0: Semantic Contract and Model Profile

Deliverables:

- `DeepSeekV4FlashMoEProfile`
- `MoEMaterialization`
- `MoERoutingContract`
- `MoEReductionSpec`
- report fields for router mode, dispatch mode, expert precision, and fallback
  reason

Acceptance:

- the model profile can enumerate all MoE layers;
- the first 3 hash-routed layers are explicit in the report;
- contiguous and batched activation layouts are recorded separately;
- unsupported shapes, dtypes, or layouts fail closed.

## Phase 1: Router and Hash-Routing Parity

DeepSeek-V4 Flash explicitly uses hash routing in the first 3 MoE layers. That
means the router itself is part of the consistency contract.

Deliverables:

- deterministic hash-routing reference;
- router provenance for `n_hash_layers`, top-k, expert bias, and router score;
- router-only drift report before expert dispatch;
- a visible failure mode when routing metadata is missing.

Acceptance:

- identical hidden states produce identical `topk_ids` and `topk_weights`
  across train and rollout;
- hash-routed and learned-routed layers are both reported with the same schema;
- drift is visible before dispatch begins;
- no silent router fallback.

## Phase 2: Fused vs Unfused MoE Materialization

Training may implement `router -> dispatch -> experts -> combine`, while rollout
may use fused MoE kernels. The task is to prove that both graphs map to one
semantic contract.

Deliverables:

- a shared MoE semantic boundary for fused and unfused graphs;
- contiguous and batched activation adapters;
- explicit `prepare / finalize / experts / topk-weight-reduce` provenance;
- reference paths for both materialization styles.

Acceptance:

- fused and unfused paths agree on dispatch metadata and final output;
- both contiguous and batched variants pass the same contract;
- `TopKWeightAndReduce` placement is explicit and reproducible.

## Phase 3: Expert Compute and Reduction-Order Consistency

This is where the actual floating-point reduction happens.

Deliverables:

- deterministic permute / unpermute reference;
- grouped GEMM candidate and reference comparison;
- quantization / dequantization boundary recording;
- explicit FP32 accumulation and downcast point.

Acceptance:

- `Permute -> W1 -> Act+Mul -> Quant -> W2 -> Unpermute -> TopKWeightAndReduce`
  matches the reference under the agreed tolerance;
- accumulation dtype and reduction order are visible in the report;
- FP4 expert weights and FP8 non-expert parameters are only accepted with
  explicit provenance;
- all-masked / empty-expert cases do not produce NaNs.

## Phase 4: EP/TP Dispatch and Cross-Config Dlogp

Above the final reduction point, MoE drift is usually batch / pack / padding /
TP / EP / communication drift.

Deliverables:

- `ShardingSpec` for MoE inputs and outputs;
- `ReductionSpec` for dispatch and combine;
- per-rank and per-expert drift reports;
- supported-layout matrix for train and rollout.

Acceptance:

- the same checkpoint and token sequence give aligned `dlogp` across supported
  EP / TP layouts;
- batch size, packing, and padding do not change MoE semantics;
- unsupported all2all / TP / EP combinations are declared and fail closed.

## Phase 5: vime Integration and Benchmark Gate

Deliverables:

- DSV4 MoE model adapter in the existing RL-Kernel style;
- audit / strict mode wiring;
- benchmark and CI smoke tests on a fixed DeepSeek-V4 Flash MoE fixture;
- structured fallback reasons and provenance logging.

Acceptance:

- one command runs the smoke case in audit and strict mode;
- strict supported paths have `fallback=0`;
- benchmark reports include model, checkpoint, routing mode, materialization,
  reduction order, backend, and `dlogp`;
- the issue clearly distinguishes operator-level consistency from WS3 full-engine
  alignment.

## Notes on Scope

- This issue does not try to solve the hybrid-attention stack around MoE, except
  where it changes the MoE input boundary.
- This issue does not replace vLLM or Megatron. It defines the semantic contract
  that those paths must satisfy when they are plugged into RL-Kernel.
- This issue does not claim real-engine alignment. That is a WS3 problem after
  the MoE contract is stable.

## References

- [#83: RL-Kernel Roadmap](https://github.com/RL-Align/RL-Kernel/issues/83)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)
- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)
- [#235: CP-aware deterministic Attention](https://github.com/RL-Align/RL-Kernel/issues/235)
- [#241: TP-aware deterministic logprob](https://github.com/RL-Align/RL-Kernel/issues/241)
- [DeepSeek-V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [MoE hash-routing discussion](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/22)
- [vLLM fused MoE modular kernel](https://docs.vllm.ai/en/latest/design/fused_moe_modular_kernel/)
