# DeepSeek-V4-Flash-0731 Operator-Level Train-Inference Consistency Research

Status: Draft / Hopper baseline

## Scope

This report studies operator-level train-inference consistency for
`deepseek-ai/DeepSeek-V4-Flash-0731`, strictly aligned to the WS1 -> WS2 split
used in [#83](https://github.com/RL-Align/RL-Kernel/issues/83).

Boundaries:

- Training side: Megatron
- Inference side: vLLM + vime
- Hardware: NVIDIA Hopper
- Consistency rule: follow #83 / #108; bitwise preferred, otherwise tight
  numerical tolerance; forward + backward; deterministic reduction order;
  batch-invariance, padding, and parallelism invariance
- Out of scope: full WS3 real-engine end-to-end alignment, time planning, and
  manpower estimation
- Performance optimization: only after consistency is proven

## Quick Verdict

The current public surface is enough to identify the main drift surfaces, but
not enough to serve as the final implementation guide. The missing piece is a
model-level contract that names the exact MoE and attention materializations and
then assigns them to WS1 and WS2 with strict acceptance criteria.

This report provides that contract.

## 1. Model Architecture and Key Operators

Public evidence for this model:

- DeepSeek-V4-Flash-0731 is the official release of DeepSeek-V4-Flash and has
  the same model structure as DeepSeek-V4-Flash-DSpark, with a speculative
  decoding module attached. [HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- The model card shows 304B parameters and BF16 weights, and the docs expose
  vLLM launch settings with `--enable-expert-parallel`, `--moe-backend
  deep_gemm_mega_moe`, `--kv-cache-dtype fp8`, `--attention-config
  '{"use_fp4_indexer_cache": true}'`, `--speculative-config
  '{"method":"dspark"...}'`, `--chunked-prefill-size 4096`, and
  `--swa-full-tokens-ratio 0.1`.
- vLLM's fused MoE modular kernel splits the MoE path into
  `FusedMoEPrepareAndFinalizeModular`, `FusedMoEExpertsModular`, and
  `TopKWeightAndReduce`. [vLLM modular kernel](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/modular_kernel/)
- vLLM's MoE design docs treat all2all backend selection, activation format,
  quantization format, and shared-expert overlap as first-class MoE material.
  [vLLM MoE features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)

### Key operator map

| Semantic block | Single operators / boundaries to track | Training materialization (Megatron) | Inference materialization (vLLM + vime) |
| --- | --- | --- | --- |
| Embedding / hidden-state entry | embedding lookup, token mask, position boundary | embedding -> hidden state | embedding -> hidden state |
| MoE router | router logits, hash routing, top-k, expert bias, load balancing | router -> top-k selection -> route metadata | router logits or top-k path inside fused/modular MoE |
| Expert dispatch | permute, dispatch, all2all, activation packing | dispatch -> expert shards | `FusedMoEPrepareAndFinalizeModular` / all2all backend |
| Shared expert | shared expert gate, shared expert MLP, shared-expert overlap | separate shared branch or fused branch | shared-expert overlap may happen during combine |
| Expert compute | W1, activation, W2, grouped GEMM, quantized expert matmul | expert MLP / grouped GEMM | `FusedMoEExpertsModular` / fused experts kernel |
| Final combine | top-k weight application, reduction order, downcast point | combine -> post-MoE hidden | `TopKWeightAndReduce` |
| Hybrid attention | SWA / CSA / HCA, mHC, cache layout, long-context merge | attention graph may differ from rollout | attention backend may differ from training materialization |
| Logprob / loss | LM head, selected-logprob, log-softmax, backward gradient | teacher-forcing scoring | rollout-side selected-logprob / vime audit |
| Speculative decoding | draft / target accept-reject, speculative module | usually absent | DSpark speculative decoding module |
| Long-context compression | fp8 KV cache, FP4 indexer cache, chunked prefill | training-side long sequence layout | rollout-side cache / chunking / compression layout |

The final floating-point reduction point for MoE is the weighted expert combine.
Everything above it is batch, pack, padding, EP, TP, CP, and communication
provenance.

## 2. Train-Inference Mismatch Risks

### P0

| Risk surface | Why it can diverge |
| --- | --- |
| MoE gating + expert dispatch + shared expert | different router implementation, top-k tie-breaking, expert-bias handling, activation layout, or shared-expert overlap can alter routed tokens before any later reduction |
| Hybrid Attention (SWA/CSA/HCA) | rollout and training may materialize different attention graphs, cache rules, or merge boundaries |
| mHC | compression state, long-context windowing, and hidden-state materialization can change what enters the MoE block |

### P1

| Risk surface | Why it can diverge |
| --- | --- |
| FP4 / FP8 mixed precision | scale generation, accumulation dtype, and downcast point can change expert outputs even if routing is identical |
| Deterministic logprob | reduction order, TP vocab sharding, and batch / padding layout can shift `dlogp` |

### P2

| Risk surface | Why it can diverge |
| --- | --- |
| Speculative decoding | draft-target acceptance changes rollout control flow without changing the training graph |
| Long-context compression details | chunking, cache layout, and indexer semantics may be rollout-only and should not be mistaken for MoE correctness |

## 3. WS1: Single-GPU / Single-Operator Plan

WS1 should extend the existing ground-truth harness, not replace it. Threshold policy remains the same as #83 / #108.

### WS1-0: Ground-truth harness extension

Problem operators:

- router logits / top-k
- hash routing
- expert dispatch
- shared expert
- expert W1 / activation / W2
- final top-k combine
- hybrid-attention boundary operators
- selected-logprob

Solution:

- extend the current WS1 harness with `ModelProfile`, `LayerContract`, and
  MoE-specific reference paths
- add explicit routing metadata, dispatch metadata, and reduction metadata
- keep one numerical contract table; do not create a new tolerance table for
  DeepSeek-V4

Strict acceptance:

- same checkpoint, same tokens, same masks, same seed
- bitwise preferred; otherwise tight tolerance only where mathematically
  unavoidable
- forward + backward are both covered
- unsupported shapes / dtypes / layouts fail closed

### WS1-1: Operator batch sweep

Sweep each MoE operator across:

- batch size
- padding side
- contiguous vs batched activation format
- top-k
- number of experts
- expert parallel layout

Strict acceptance:

- per-operator outputs are batch-invariant within the #83 / #108 policy
- no operator changes meaning when padding or packing changes
- reduction order is fixed and reported

### WS1-2: Full-chain single-GPU consistency

The single-GPU chain should cover:

`embedding -> hybrid attention -> MoE router/dispatch/experts/combine -> LM head -> selected-logprob`

Strict acceptance:

- full-chain forward and backward are stable
- the same input produces the same output under batch sweeps and padding
- any fallback is visible and intentional
- final report includes max / p95 / p99 absolute drift and `dlogp`

## 4. WS2: Multi-GPU / Distributed Plan

WS2 starts once the WS1 operator semantics are stable.

### WS2-0: Distributed contract

Problem operators:

- EP all2all dispatch
- TP vocab-parallel logprob
- cross-rank expert combine
- batch / pack / padding interaction
- distributed hybrid-attention boundary if attention is part of the chain

Solution:

- add `ShardingSpec` for MoE inputs and outputs
- add `ReductionSpec` for dispatch and combine
- record `accum_dtype`, reduction order, and downcast point
- make collectives deterministic where the contract requires it

Strict acceptance:

- same checkpoint, token sequence, and masks produce aligned outputs across
  supported EP / TP / CP layouts
- per-rank drift is reported, not hidden
- unsupported collective or topology combinations fail closed

### WS2-1: Cross-config logprob alignment

Compare:

- Megatron training forward
- vLLM + vime rollout forward

Target metric:

```text
dlogp = training_recomputed_logp - rollout_old_logp
```

Strict acceptance:

- `dlogp` is measured only on active response/action tokens
- batch, padding, and parallelism differences do not change the semantic
  contract
- `ratio0`, `clipfrac0`, `approx_kl0`, and absolute-drift percentiles are
  reported
- #83 / #108 thresholds remain the single source of truth

### WS2-2: Optional R3 routing replay

R3 is useful but optional.

It can reduce one major source of mismatch when rollout exports routed experts,
but it is not the semantic contract itself and should not be required for the
roadmap to hold.

Strict acceptance:

- if R3 is enabled, it must appear in provenance
- if R3 is absent, the same MoE contract still applies
- no silent fallback from routed-expert metadata to an untracked router path

## 5. Integration Suggestions for RL-Kernel / vime

1. Reuse the existing WS1 harness and WS2 drift contract. Do not create a new
   tolerance table for DeepSeek-V4.
2. Add a model-level layer above the operator harness:
   `ModelProfile`, `LayerContract`, `GraphMaterialization`, `SemanticTrace`.
3. Keep vime responsible for orchestration and metadata transport only. vime
   can pass `rollout_routed_experts` and `use_rollout_routing_replay`, but those
   fields are provenance, not the contract source.
4. Treat vLLM's observed MoE materialization as a candidate backend, not as the
   truth source. The truth source stays in RL-Kernel reference paths and
   deterministic comparisons.
5. Keep the split between consistency path and fast path:
   - consistency path: audit / strict / fallback-visible
   - fast path: only after the contract is proven

## References

- [#83: RL-Kernel Roadmap](https://github.com/RL-Align/RL-Kernel/issues/83)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)
- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)
- [#235: CP-aware deterministic Attention](https://github.com/RL-Align/RL-Kernel/issues/235)
- [#241: TP-aware deterministic logprob](https://github.com/RL-Align/RL-Kernel/issues/241)
- [DeepSeek-V4-Flash-0731 HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [vLLM fused MoE modular kernel](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/modular_kernel/)
- [vLLM MoE features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)
