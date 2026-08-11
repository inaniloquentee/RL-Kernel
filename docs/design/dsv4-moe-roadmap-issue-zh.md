<!--
草稿来源：#83 总 Roadmap、WS1 #108、WS2 #111 / #207 / #241、DeepSeek-V4
Flash model card、DeepSeek-V4 的 hash routing 讨论，以及 vLLM fused MoE
modular kernel 设计。本文是
docs/design/model-level-train-rollout-consistency-plan.md 的 DeepSeek-V4
Flash MoE 专项版。
-->

# [Roadmap] DeepSeek-V4 Flash MoE 训推一致

## 背景

DeepSeek-V4 Flash 是一个 MoE 模型：总参数 284B，激活参数 13B，支持
1M context，并且采用 FP4 + FP8 mixed precision。官方 model card 还说明，
它的 post-training 是两阶段：先通过 SFT / GRPO 做 domain-specific experts
的独立培养，再通过 on-policy distillation 做统一收敛。这意味着 training
和 rollout 不一定会落到同一张计算图。

这份 Roadmap 只管 MoE 子图，不管整棵 transformer。Hybrid attention、
mHC，以及其它上下游模块，只在它们改变 MoE 输入或输出边界时才算进来。

MoE 之所以不是“单算子问题”，是因为 vLLM 的 fused MoE modular kernel
本身就已经把语义拆成了几段：

- `FusedMoEPrepareAndFinalizeModular`
- `FusedMoEExpertsModular`
- `TopKWeightAndReduce`

而且 All2All dispatch 下，输入 activation 还可能是 contiguous 或 batched。
training 侧则可能把同一套语义拆成 router / dispatch / expert / combine
几个算子。也就是说，MoE 的一致性合同必须定义在 kernel 边界之上。

## 会导致训推不一致的地方

| 边界 | 常见不一致来源 | 对应解决面 |
| --- | --- | --- |
| Router | 前 3 个 MoE layer 的 hash routing、top-k、expert bias、router dtype、load-balancing policy | 确定性 router contract + provenance |
| Dispatch | contiguous / batched activation layout、token packing、all2all 顺序、`expert_num_tokens` | 显式 materialization map + `ShardingSpec` |
| Expert compute | permute、W1、act + mul、quantization、W2、unpermute | reference expert contract + grouped GEMM 确定性 |
| Final combine | top-k weight 应用、reduction 顺序、accumulator dtype、downcast 点 | `ReductionSpec`，FP32 merge，明确 downcast |
| Distributed layout | MoE 边界附近的 EP / TP / 可能的 CP、batch size、padding、pack 顺序 | cross-config `dlogp` audit + per-rank drift report |

最底层真正的浮点归约点，是 weighted expert combine。它上面才是 batch、
pack、padding、EP、TP、CP 和通信语义。

## 共享合同

只有下面这些东西一致，才算可比：

- 同一个 checkpoint 和 tokenizer；
- 同一组 input token ids 和 masks；
- 进入 MoE block 的 hidden-state 边界一致；
- 同一套 routing metadata：`n_hash_layers`、`topk`、
  `router_score_function`、`router_dtype`、`expert_bias`、`load_balancing`、
  `activation_format`、`topk_id_dtype`；
- 同一套 reduction metadata：`accum_dtype`、`order`、`downcast_point`、
  `combine_backend`；
- 同一个 fallback 规则：不允许 silent fallback。

如果上游 attention / embedding / LM-head 已经脏了，那就不要把锅甩给
MoE。这个 Roadmap 只负责 MoE 子图和它的直接接口。

R3 routing replay 可以用，但不建议作为前提条件。它的价值是：
如果 rollout 能导出 routed experts，R3 能显著降低一类大的路由漂移，
也能让 audit 更干净；但它不是 MoE 语义合同本身，Roadmap 也不应该
依赖它才能成立。没有 R3，MoE 的 contract 仍然要能独立成立。

## 建议的 Issue 划分

| 阶段 | 建议 Issue 标题 | 重点 | 完成标准 |
| --- | --- | --- | --- |
| 0 | `[DSV4-MOE] semantic contract and model profile` | model profile、materialization map、provenance schema | 能报告 model / route / layout，且 unsupported config fail closed |
| 1 | `[DSV4-MOE] router and hash-routing parity` | 前 3 个 hash-routed layer、top-k、expert bias、router dtype | router 输出在 dispatch 前对齐 |
| 2 | `[DSV4-MOE] fused vs unfused MoE materialization` | training graph vs rollout graph | fused 和 unfused 统一到同一个 semantic contract |
| 3 | `[DSV4-MOE] expert compute and reduction-order consistency` | permute、grouped GEMM、quantization、top-k reduce | final expert output 在同一 reduction contract 下对齐 |
| 4 | `[DSV4-MOE] EP/TP dispatch and cross-config dlogp` | all2all、pack/padding、batch shape、per-rank drift | rollout vs training 的 dlogp 在支持布局下对齐 |
| 5 | `[DSV4-MOE] vime integration and benchmark gate` | adapter、audit/strict mode、smoke tests、CI | 一条命令跑通 smoke，且 fallback 和 report 都清楚 |

## 阶段 0：语义合同和模型画像

交付物：

- `DeepSeekV4FlashMoEProfile`
- `MoEMaterialization`
- `MoERoutingContract`
- `MoEReductionSpec`
- router mode、dispatch mode、expert precision、fallback reason 等 report 字段

验收标准：

- 可以枚举全部 MoE layers；
- 前 3 个 hash-routed layer 在 report 里是显式的；
- contiguous 和 batched activation layout 分开记录；
- unsupported 的 shape、dtype、layout 必须 fail closed。

## 阶段 1：Router 和 hash routing 对齐

DeepSeek-V4 Flash 在前 3 个 MoE layer 上显式使用 hash routing，所以 router
本身就是一致性合同的一部分。

交付物：

- 确定性的 hash-routing reference；
- `n_hash_layers`、top-k、expert bias、router score 的 provenance；
- router-only drift report，先于 expert dispatch；
- routing metadata 缺失时可见的失败模式。

验收标准：

- 同样的 hidden state 会得到同样的 `topk_ids` / `topk_weights`；
- hash-routed layer 和 learned-routed layer 使用同一套 report schema；
- drift 能在 dispatch 之前被看见；
- 不允许 silent router fallback。

## 阶段 2：Fused vs unfused MoE materialization

training 侧可能是 `router -> dispatch -> experts -> combine`，
rollout 侧可能是 fused MoE kernel。这里要证明：两张图都能映射到同一份
semantic contract。

交付物：

- fused / unfused 共享的 MoE semantic boundary；
- contiguous 和 batched activation adapter；
- 显式的 `prepare / finalize / experts / topk-weight-reduce` provenance；
- 两种 materialization 的 reference path。

验收标准：

- fused 和 unfused path 在 dispatch metadata 和最终输出上对齐；
- contiguous / batched 两种变体都满足同一份 contract；
- `TopKWeightAndReduce` 的位置显式且可复现。

## 阶段 3：Expert compute 和 reduction 顺序一致

这里才是真正的浮点归约点。

交付物：

- 确定性的 permute / unpermute reference；
- grouped GEMM candidate 与 reference 的对比；
- quantization / dequantization 边界记录；
- 显式的 FP32 accumulation 和 downcast 点。

验收标准：

- `Permute -> W1 -> Act+Mul -> Quant -> W2 -> Unpermute -> TopKWeightAndReduce`
  在约定容差下对齐 reference；
- report 里能看到 accumulation dtype 和 reduction order；
- FP4 expert weights 和 FP8 non-expert 参数只有在 provenance 明确时才接受；
- all-masked / empty-expert 情况不能产出 NaN。

## 阶段 4：EP / TP dispatch 和 cross-config dlogp

在最终 reduction 点之上，MoE drift 大多会表现为 batch / pack / padding /
TP / EP / communication drift。

交付物：

- MoE 输入 / 输出的 `ShardingSpec`；
- dispatch / combine 的 `ReductionSpec`；
- per-rank 和 per-expert drift report；
- training 和 rollout 的支持布局矩阵。

验收标准：

- 同一个 checkpoint 和 token sequence 在支持的 EP / TP layout 下
  `dlogp` 对齐；
- batch size、packing、padding 不改变 MoE 语义；
- 不支持的 all2all / TP / EP 组合要显式声明并 fail closed。

## 阶段 5：vime 接入和 benchmark 门槛

交付物：

- 按现有 RL-Kernel 风格做 DeepSeek-V4 Flash MoE adapter；
- audit / strict mode 接线；
- 固定 DeepSeek-V4 Flash MoE fixture 的 benchmark 和 CI smoke test；
- 结构化 fallback reason 和 provenance logging。

验收标准：

- 一条命令能跑通 audit 和 strict 两种 smoke case；
- 支持配置下 strict path 的 `fallback=0`；
- benchmark report 里必须包含 model、checkpoint、routing mode、
  materialization、reduction order、backend 和 `dlogp`；
- 能清楚区分 operator-level consistency 和 WS3 full-engine alignment。

## 作用范围说明

- 这份 Roadmap 不解决 MoE 外围的 hybrid attention，除非它改变了 MoE
  输入边界；
- 这份 Roadmap 不替代 vLLM 或 Megatron，而是定义它们接入 RL-Kernel
  时必须满足的语义合同；
- 这份 Roadmap 不宣称 real-engine alignment。那是 MoE 合同稳定之后，
  才能继续推进的 WS3 问题。

## 参考

- [#83: RL-Kernel Roadmap](https://github.com/RL-Align/RL-Kernel/issues/83)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)
- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)
- [#235: CP-aware deterministic Attention](https://github.com/RL-Align/RL-Kernel/issues/235)
- [#241: TP-aware deterministic logprob](https://github.com/RL-Align/RL-Kernel/issues/241)
- [DeepSeek-V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [MoE hash-routing discussion](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/22)
- [vLLM fused MoE modular kernel](https://docs.vllm.ai/en/latest/design/fused_moe_modular_kernel/)
