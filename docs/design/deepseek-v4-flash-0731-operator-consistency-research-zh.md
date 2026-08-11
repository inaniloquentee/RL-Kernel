# DeepSeek-V4-Flash-0731 算子级训推一致性调研报告

状态：Draft / Hopper baseline

## 边界

边界：

- 训练侧：Megatron
- 推理侧：vLLM + vime
- 硬件：NVIDIA Hopper
- 一致性规则：沿用 #83 / #108；bitwise 优先，否则 tight numerical
  tolerance；包含 forward + backward；要求 deterministic reduction order；
  覆盖 batch-invariance、padding、并行配置
- 不做：完整 WS3 真实引擎端到端对齐、时间计划、人力估算
- 性能优化：只在一致性得到保证后才考虑

## 1. 模型架构与关键算子清单

公开信息：

- DeepSeek-V4-Flash-0731 是 DeepSeek-V4-Flash 的正式 release，和
  DeepSeek-V4-Flash-DSpark 具有相同模型结构，并附带 speculative decoding
  module。[HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- model card 显示模型参数 304B，权重为 BF16，并给出了 vLLM 启动配置：
  `--enable-expert-parallel`、`--moe-backend deep_gemm_mega_moe`、
  `--kv-cache-dtype fp8`、`--attention-config '{"use_fp4_indexer_cache": true}'`、
  `--speculative-config '{"method":"dspark"...}'`、
  `--chunked-prefill-size 4096`、`--swa-full-tokens-ratio 0.1`。
- vLLM fused MoE modular kernel 把 MoE 路径拆成
  `FusedMoEPrepareAndFinalizeModular`、`FusedMoEExpertsModular`、
  `TopKWeightAndReduce`。[vLLM modular kernel](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/modular_kernel/)
- vLLM MoE 设计把 all2all backend、activation format、quantization format、
  shared-expert overlap 都作为 MoE materialization 的一部分。
  [vLLM MoE features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)

### 关键算子表

| 语义块 | 需要跟踪的单算子 / 边界 | Megatron 训练侧 materialization | vLLM + vime 推理侧 materialization |
| --- | --- | --- | --- |
| Embedding / hidden-state entry | embedding lookup、token mask、position boundary | embedding -> hidden state | embedding -> hidden state |
| MoE router | router logits、hash routing、top-k、expert bias、load balancing | router -> top-k selection -> route metadata | router logits 或 fused/modular MoE 内部 top-k path |
| Expert dispatch | permute、dispatch、all2all、activation packing | dispatch -> expert shards | `FusedMoEPrepareAndFinalizeModular` / all2all backend |
| Shared expert | shared expert gate、shared expert MLP、shared-expert overlap | 独立 shared branch 或 fused branch | shared-expert overlap 可能发生在 combine 阶段 |
| Expert compute | W1、activation、W2、grouped GEMM、quantized expert matmul | expert MLP / grouped GEMM | `FusedMoEExpertsModular` / fused experts kernel |
| Final combine | top-k weight application、reduction order、downcast point | combine -> post-MoE hidden | `TopKWeightAndReduce` |
| Hybrid attention | SWA / CSA / HCA、mHC、cache layout、long-context merge | attention graph 可能不同于 rollout | attention backend 可能不同于 training materialization |
| Logprob / loss | LM head、selected-logprob、log-softmax、backward gradient | teacher-forcing scoring | rollout-side selected-logprob / vime audit |
| Speculative decoding | draft / target accept-reject、speculative module | 通常不存在 | DSpark speculative decoding module |
| Long-context compression | fp8 KV cache、FP4 indexer cache、chunked prefill | training-side long sequence layout | rollout-side cache / chunking / compression layout |

MoE 最终真正的浮点归约点是 weighted expert combine。它上面才是
batch、pack、padding、EP、TP、CP 和通信 provenance。

## 2. 训推不一致风险分析

### P0

| 风险面 | 为什么会漂 |
| --- | --- |
| MoE gating + expert dispatch + shared expert | router 实现、top-k tie-breaking、expert-bias、activation layout、shared-expert overlap 任一不同，都会在后续 reduction 之前改变 routed tokens |
| Hybrid Attention (SWA/CSA/HCA) | rollout 和 training 可能使用不同 attention graph、cache 规则或 merge 边界 |
| mHC | compression state、long-context windowing、hidden-state materialization 可能改变进入 MoE block 的输入 |

### P1

| 风险面 | 为什么会漂 |
| --- | --- |
| FP4 / FP8 mixed precision | scale 生成、accumulation dtype、downcast 点不同，会在 routing 相同的情况下改变 expert output |
| Deterministic logprob | reduction order、TP vocab sharding、batch / padding layout 会改变 `dlogp` |

### P2

| 风险面 | 为什么会漂 |
| --- | --- |
| Speculative decoding | draft-target accept/reject 改变 rollout control flow，但 training graph 没有同构路径 |
| 长上下文压缩细节 | chunking、cache layout、indexer semantics 可能是 rollout-only，不能误判成 MoE 算子正确性 |

## 3. WS1：单 GPU / 单算子级方案

WS1 要扩展现有 ground-truth harness，而不是重写一套。阈值策略完全沿用
#83 / #108。

### WS1-0：Ground-truth harness 扩展

问题算子：

- router logits / top-k
- hash routing
- expert dispatch
- shared expert
- expert W1 / activation / W2
- final top-k combine
- hybrid-attention boundary operators
- selected-logprob

解决方案：

- 在当前 WS1 harness 上扩展 `ModelProfile`、`LayerContract` 和 MoE reference；
- 增加 routing metadata、dispatch metadata、reduction metadata；
- 只保留一套数值合同表，不为 DeepSeek-V4 单独建 tolerance table。

严格验收标准：

- checkpoint、tokens、masks、seed 完全一致；
- bitwise 优先；数学上不可避免时才使用 tight tolerance；
- forward + backward 都覆盖；
- unsupported shape / dtype / layout 必须 fail closed。

### WS1-1：各算子 batch sweep

每个 MoE 算子需要 sweep：

- batch size
- padding side
- contiguous vs batched activation format
- top-k
- number of experts
- expert parallel layout

严格验收标准：

- 每个单算子输出在 #83 / #108 策略下 batch-invariant；
- padding 或 packing 改变时，算子语义不变；
- reduction order 固定且写入报告。

### WS1-2：Full-chain single-GPU consistency

单卡链路覆盖：

```text
embedding -> hybrid attention -> MoE router/dispatch/experts/combine
          -> LM head -> selected-logprob
```

严格验收标准：

- full-chain forward 和 backward 稳定；
- 同一输入在 batch sweep 和 padding sweep 下输出一致；
- 所有 fallback 都必须可见且有意图；
- 报告包含 max / p95 / p99 absolute drift 和 `dlogp`。

## 4. WS2：多 GPU / 分布式方案

WS2 在 WS1 算子语义稳定之后开始。

### WS2-0：分布式合同

问题算子：

- EP all2all dispatch
- TP vocab-parallel logprob
- cross-rank expert combine
- batch / pack / padding interaction
- 如果 attention 纳入 full chain，还包括 distributed hybrid-attention boundary

解决方案：

- 给 MoE 输入 / 输出增加 `ShardingSpec`；
- 给 dispatch / combine 增加 `ReductionSpec`；
- 记录 `accum_dtype`、reduction order、downcast point；
- 在合同要求的位置使用 deterministic collectives。

严格验收标准：

- 同一个 checkpoint、token sequence、masks 在支持的 EP / TP / CP layout
  下输出对齐；
- per-rank drift 必须报告，不能隐藏；
- 不支持的 collective 或拓扑组合必须 fail closed。

### WS2-1：Cross-config logprob 对齐

比较对象：

- Megatron training forward
- vLLM + vime rollout forward

核心指标：

```text
dlogp = training_recomputed_logp - rollout_old_logp
```

严格验收标准：

- `dlogp` 只在 active response/action tokens 上计算；
- batch、padding、parallelism 差异不改变 semantic contract；
- 报告 `ratio0`、`clipfrac0`、`approx_kl0` 和 absolute-drift percentiles；
- #83 / #108 阈值继续作为唯一真源。

### WS2-2：可选 R3 routing replay

R3 有用，但不是前提。

当 rollout 能导出 routed experts 时，它可以减少一类大的 mismatch；但
R3 不是 MoE semantic contract 本身，也不应该成为 Roadmap 成立的必要条件。

严格验收标准：

- 如果启用 R3，必须写入 provenance；
- 如果没有 R3，同一份 MoE contract 仍然成立；
- 不允许从 routed-expert metadata silent fallback 到未记录的 router path。

## 5. 与 RL-Kernel / vime 的对接建议

1. 复用现有 WS1 harness 和 WS2 drift contract。不要给 DeepSeek-V4 新建
   tolerance table。
2. 在 operator harness 之上加模型级薄层：
   `ModelProfile`、`LayerContract`、`GraphMaterialization`、`SemanticTrace`。
3. vime 继续负责 orchestration 和 metadata transport。vime 可以传
   `rollout_routed_experts` 和 `use_rollout_routing_replay`，但这些字段是
   provenance，不是合同真源。
4. vLLM 的 MoE materialization 只能作为 candidate backend，不是真源。真源仍然是
   RL-Kernel reference path 和 deterministic comparison。
5. 保持 consistency path 和 fast path 分离：
   - consistency path：audit / strict / fallback-visible
   - fast path：只在 contract 证明之后再启用

## References

- [#83: RL-Kernel Roadmap](https://github.com/RL-Align/RL-Kernel/issues/83)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)
- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)
- [#235: CP-aware deterministic Attention](https://github.com/RL-Align/RL-Kernel/issues/235)
- [#241: TP-aware deterministic logprob](https://github.com/RL-Align/RL-Kernel/issues/241)
- [DeepSeek-V4-Flash-0731 HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [vLLM fused MoE modular kernel](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/modular_kernel/)
- [vLLM MoE features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)
