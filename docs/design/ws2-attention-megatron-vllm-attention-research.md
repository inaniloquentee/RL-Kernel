# WS2 Attention：Megatron 与 vLLM/FlashInfer 调研结论

状态：针对 [#235](https://github.com/RL-Align/RL-Kernel/issues/235) 和
[#249](https://github.com/RL-Align/RL-Kernel/issues/249) 的核心调研结论  
调研快照：2026-08-05

## 1. 核心结论

是的，#249 的 Attention ablation 需要调研 Megatron/TE 和
vLLM/FlashInfer 的实现差异，但目标不是让两边强行使用同一个 kernel。

推荐边界：

```text
Megatron / TE：
  训练侧 full-prefill candidate，TE 可作为 CP merge oracle

vLLM / FlashInfer：
  rollout 侧 paged-prefill/decode candidate

RL-Kernel：
  定义 contract、确定性 reference、逻辑 merge 顺序、drift report
```

TE 和 FlashInfer 都不能成为 #235 的语义来源。最终一致性必须由
RL-Kernel contract 和 reference path 判定。

所以更准确的理解是：可以模仿 Megatron/vLLM 复用 TE/FlashInfer 的
工程模式，但要用 RL-Kernel adapter 把它们约束到同一个语义 contract。
允许不同的是 kernel、layout、物理 page、chunk 方式、通信方式；不允许
不同的是会改变结果的 Attention 语义和分布式逻辑语义。

## 2. #235 的最小 contract

目标配置：

| 项目 | 约束 |
| --- | --- |
| 模型 | Qwen3-8B |
| 精度 | BF16，关键 merge 使用 FP32 |
| 并行 | TP=2，CP=2 |
| Attention | `Hq=32`，`Hkv=8`，`D=128` |
| 场景 | full-prefill、chunked-prefill、decode replay |
| 对比量 | `out`、attention-domain `lse`、active-token `dlogp` |
| 反向 | 训练路径比较 `dq/dk/dv`，rollout 只要求 forward |

每个 CP/KV block 产生：

```text
(partial_out, partial_lse, global_block_index)
```

merge 必须：

- 使用 FP32 online-softmax merge；
- 按 `global_block_index` 排序；
- 不能使用通信到达顺序；
- 最后一步才 downcast；
- all-masked/empty-KV 行返回 `out=0`、`lse=-inf`，不能产生 NaN。

## 3. 上游实现差异

| 维度 | Megatron / TE | vLLM / FlashInfer | 对 #235 的要求 |
| --- | --- | --- | --- |
| 训练/推理 | Megatron 训练多为 full-prefill；CP 通常走 TE | vLLM 将 prefill 和 decode 分开处理 | 统一到同一个 Attention contract |
| backend | 由 `attention_backend`、FlashAttention 版本等配置选择 | 根据 dtype、head size、KV cache、batch 等动态选择 | 记录 requested/actual backend，禁止 silent fallback |
| layout | Megatron 常用 `[S,B,H,D]`，TE 支持多种 QKV layout | 常见 packed-varlen、paged KV metadata | contract 明确 layout、stride、GQA 映射 |
| GQA | Megatron 显式处理 Q/KV head 映射 | backend 负责 grouped-query 映射 | `Hq/Hkv` 和 TP head ownership 必须显式记录 |
| RoPE | 可单独执行，也可由融合路径执行 | 可与 KV-cache update 或 attention 融合 | 记录 pre/post-RoPE、位置、theta、cast 边界 |
| CP | TE 支持 `p2p`、`all_gather`、`a2a` 等通信模式 | 分布式路径和 backend 能力不同 | 通信只负责搬运，数值 merge 仍按逻辑顺序 |
| KV cache | 训练侧通常没有 rollout 式 paged cache | FlashInfer 使用 page table、indptr、slot mapping | 先恢复逻辑 KV 顺序，再比较 attention |
| LSE | TE correction helper 支持 `(out,lse)` 合并 | FlashInfer 可返回 attention LSE | 明确 LSE domain、base、shape |
| split-KV | FlashAttention/TE 内部可能自行分块 | FlashInfer 支持 disabled/fixed/auto | `auto` 不能默认宣称 batch-invariant |
| backward | TE 可能依赖内部 saved state | FlashInfer rollout 路径是 forward-only | PR8 单独验证 `dq/dk/dv` |

其中最重要的差异不是 API，而是：

1. QKV/RoPE/cache 的 materialization 边界不同；
2. paged KV 的物理顺序不等于逻辑顺序；
3. split-KV、tile、通信顺序会改变浮点 reduction 顺序；
4. 某些 backend 不导出兼容的 attention-domain LSE 或 backward state。

## 4. #249 的 Attention knob

#249 用户侧建议只暴露一个 Attention knob：

```text
attention = on / off
```

含义：

| 状态 | 含义 |
| --- | --- |
| `attention=off` | 不单独归因 Attention mismatch |
| `attention=on` | 将 Attention 纳入 mismatch attribution，比较训练侧和 rollout 侧是否满足同一 #235 contract |

打开 `attention=on` 后，mismatch 的判定仍然是一个整体：

```text
same input / weights / positions / cache identity
  -> compare out, attention-domain lse, active-token dlogp
  -> drift 超过 #108 tolerance 就是 Attention mismatch
```

前一节提到的 backend、RoPE、split-KV、paged KV、merge order 等不应作为
用户侧独立 knobs。它们更适合作为 Attention mismatch report 里的内部归因维度：

| 归因维度 | 用途 |
| --- | --- |
| backend | 说明实际走的是 TE、FlashInfer、reference 还是 fallback |
| materialization | 区分 full-prefill、chunked-prefill、paged-prefill、decode |
| RoPE boundary/state | 判断 mismatch 是否来自 fused/unfused RoPE 或 pre/post-RoPE 不一致 |
| paged KV | 判断 mismatch 是否来自物理 page 与逻辑 token 顺序不一致 |
| merge order | 判断 mismatch 是否来自 CP/split-KV reduction 顺序 |
| split-KV policy | 判断 mismatch 是否来自 disabled/fixed/auto split-KV |
| LSE domain | 判断 mismatch 是否来自 LSE base/domain/shape 不兼容 |
| precision/downcast | 判断 mismatch 是否来自 accumulation dtype 或 downcast 边界 |

所以更准确的关系是：

```text
Attention knob:
  是否打开 Attention mismatch attribution

Attention report dimensions:
  mismatch 打开后，用来定位 drift 来源的 provenance/diagnostic fields
```

当某个优化路径需要被关闭以做归因时，关闭后的路径必须回到
RL-Kernel deterministic reference，而不是普通 PyTorch SDPA。

## 5. 与 #235 各 PR 的对应关系

| PR | 核心职责 | TE/FlashInfer 角色 |
| --- | --- | --- |
| [PR1/#236](https://github.com/RL-Align/RL-Kernel/pull/236) | 完善 Attention contract、TP/CP、RoPE、cache、backend metadata | 不执行 TE/FlashInfer，只定义能力和 provenance 字段 |
| [PR2/#253](https://github.com/RL-Align/RL-Kernel/pull/253) | 单卡 full/chunked/paged、RoPE、`out/lse/dlogp` attribution | TE 作为可选 merge oracle |
| [PR3/#238](https://github.com/RL-Align/RL-Kernel/pull/238) | CP deterministic reference，固定 `global_block_index` merge | 对比 TE correction helper，但 RL-Kernel 自己定义语义 |
| [PR4/#263](https://github.com/RL-Align/RL-Kernel/pull/263) | Qwen3 TP=2、CP=2 集成和运行时 provenance | 记录 requested/actual backend，不默认启用 |
| PR5 | 分布式 prefill/chunked-prefill drift benchmark | 对比 reference 与 TE/候选 backend |
| [PR6/#260](https://github.com/RL-Align/RL-Kernel/pull/260) | decode KV-cache replay，校验 page/cache/logical order | TE 只校验 merge，cache 语义由 RL-Kernel 校验 |
| [PR7/#279](https://github.com/RL-Align/RL-Kernel/pull/279) | 评估生产候选 backend | 训练侧 TE full-prefill，rollout 侧 FlashInfer paged attention |
| PR8 | 训练 backward 和 `dq/dk/dv` | 只有拿到兼容 saved state 才能评估 TE backward |

推荐顺序：

```text
PR1 -> PR2 -> PR3 -> PR6 -> PR4/PR5 -> PR7 -> PR8
```

PR7 不能在 PR2/PR3/PR6 的 reference 和 drift report 之前成为正确性依据。

## 6. 最小验收矩阵

候选 backend 至少要比较：

- `out`；
- attention-domain `lse`；
- active-token `dlogp`；
- 训练 backward 的 `dq/dk/dv`。

至少覆盖：

| 场景 | 验证内容 |
| --- | --- |
| CP=1/2 | 结果不依赖通信到达顺序 |
| full/chunked | 同一逻辑 query 行结果一致 |
| full KV/paged KV | 物理 page 变化不改变逻辑结果 |
| pre/post/fused RoPE | 位置、theta、cast 边界一致 |
| split-KV disabled/fixed/auto | drift 能归因到 split policy |
| batch-alone/inside-batch | 验证 batch invariance |
| backend unavailable | fail closed 或显式 reference fallback |

## 7. 最终判断

#249 的正确实现不是“把 Megatron 或 vLLM 的 Attention 原样搬进来”，而是：

```text
研究上游 materialization 和优化点
  -> 用户侧只暴露一个 Attention mismatch knob
  -> 把实现差异记录成 report/provenance 里的归因维度
  -> 关闭优化做归因时回到 RL-Kernel deterministic reference
  -> 用统一的 out/lse/dlogp/gradient report 验证
```

TE 适合复用为训练侧 candidate 和 `(out,lse)` merge oracle；FlashInfer
适合复用为 rollout 侧 paged-prefill/decode candidate。两者都必须服从
RL-Kernel 的逻辑 KV 顺序、LSE 约定、精度边界和 fallback 规则。

只保证 split-K、reduction、merge 顺序还不够。mask、scale、GQA head
映射、RoPE 位置和状态、cache identity、TP/CP ownership、LSE domain
这些语义也必须一致；否则即使 reduction 顺序一致，`out/lse/dlogp`
仍然可能不是同一个 Attention 问题的结果。

## 参考链接

- [#235](https://github.com/RL-Align/RL-Kernel/issues/235)
- [#249](https://github.com/RL-Align/RL-Kernel/issues/249)
- [Megatron attention](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py)
- [Megatron TransformerConfig](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py)
- [TE CP attention helpers](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py)
- [vLLM FlashInfer backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py)
- [FlashInfer prefill](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py)
- [FlashInfer decode](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py)
- [RL-Kernel Attention 文档](../operators/attention.md)
