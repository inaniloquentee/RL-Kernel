# 模型级训推一致与多架构接入规划

Status: Future planning

## 背景

#83 对 RL-Kernel 的定位很明确：RL-Kernel 是微观 RL operator library，不接管 verl、slime、vLLM、Megatron、DeepSpeed、FSDP 这类框架的宏观调度。RL-Kernel 要负责的是低效路径替换、数值契约、drift 定位，以及 rollout 和 training 的一致性工具。

当前 WS2 的 #235 是一个收敛用的固定目标：

```text
model: Qwen3-8B dense
dtype: BF16 + required FP32 accumulation
parallelism: TP=2, CP=2
scope: Attention CP semantics
```

这仍然是 **固定模型架构、固定分布式语义下的算子级一致性**。未来要支持更多模型架构，并做到真实 post-training 场景下的训推一致，需要把一致性边界从单个 op 提升到模型级语义层。

## 核心问题

真实 post-training 里，training 和 rollout 很可能不会运行同一张计算图。

例如 Qwen3：

```text
rollout:
  RoPE + Attention 可能被融合在同一个 kernel 中

training:
  RoPE 和 Attention 可能是两个分开的算子
```

并且两侧并行策略也可能不同：

```text
rollout:
  TP=2, paged-KV, chunked-prefill, decode cache

training:
  FSDP/TP/CP 组合，teacher-forcing full sequence
```

所以未来目标不能是“training 和 rollout 调同一个 kernel”。更合理的目标是：

```text
不同计算图
不同融合边界
不同并行策略
不同 runtime backend

绑定到同一个模型级语义契约
```

只要输入、权重、position、cache identity、mask、采样后 token 序列一致，就应该在 contract 要求的位置达到 bitwise 或严格 tolerance 一致。

## 和 WS1 / WS2 / WS3 的关系

### WS1：单卡算子级一致性

WS1 解决的是单个算子或单卡 forward chain 的稳定性。

重点是：

- batch size 不改变结果；
- padding / packing 不改变语义；
- chunked-prefill 不改变 attention/logprob；
- forward 和 backward 都有 reference；
- 每个 op 有明确数值契约。

WS1 是模型级一致性的底座。没有 WS1，后面的模型级比较无法定位 drift 来源。

### WS2：固定分布式语义下的算子级一致性

现在的 WS2 是把 WS1 的契约扩展到分布式和跨配置场景。

当前 #235 的性质是：

```text
固定模型: Qwen3-8B
固定目标: TP=2, CP=2, BF16
固定对象: Attention
固定语义: CP-aware standard softmax attention
```

它验证的是 operator-level distributed consistency，不是完整的“任意模型、任意训练/推理图一致”。

WS2 的产物会成为未来模型级一致性的基础组件：

- `ShardingSpec`;
- `ReductionSpec`;
- sequence / cache metadata;
- fixed merge order;
- attention-domain `lse`;
- drift report;
- unsupported fallback policy。

### WS3：真实引擎对齐

WS3 才是 real vLLM / sglang rollout 对齐 real Megatron / FSDP training 的阶段。

模型级一致性规划服务于 WS3：它把真实引擎里的不同计算图映射到统一语义层，然后复用 WS1/WS2 的 reference、metadata 和 drift report 做判断。

## 可参考做法

### vLLM：Layer 和 Op 分层

vLLM 的插件体系和 pluggable layer 思路值得借鉴：复杂模型差异应该放在 layer 层处理，底层 op 保持相对单一。

RL-Kernel 可以借鉴这个方向，但不能只做“能插拔”。RL-Kernel 的 layer 还必须绑定数值契约、并行语义、fallback 和 drift report。

### Transformers：按 config 注册模型架构

Transformers 的 `AutoConfig` / `AutoModel` 注册方式适合参考。模型差异首先应该由结构化 config 描述，而不是散落在 backend if-else 里。

RL-Kernel 可以引入内部 `ModelProfile`：

```text
Qwen3DenseProfile
LlamaDenseProfile
DeepSeekMLAProfile
QwenMoEProfile
```

### Megatron-Core：模块规格和并行配置分离

Megatron-Core 的模块规格化设计说明：layer 的组件、并行策略和 backend 可以拆开描述。

RL-Kernel 可以参考这种方式，把“模型层语义”和“具体 kernel/backend”分开。

## 建议架构

### 1. ModelProfile

`ModelProfile` 描述模型架构，不描述具体 kernel。

示例字段：

```text
model_family
hidden_size
num_attention_heads
num_kv_heads
head_dim
rope_type
rope_theta
qk_norm
attention_type
mlp_type
moe_type
vocab_size
tie_word_embeddings
cache_layout
```

它回答的是：这个模型的语义结构是什么。

### 2. LayerContract

`LayerContract` 描述某一层在数学上应该做什么。

例如 Attention layer：

```text
input:
  hidden
  position_ids / cache_position
  attention_mask
  kv_cache_identity

semantic steps:
  qkv projection
  optional qk_norm
  rope
  standard / MLA / sliding attention
  output projection

required exported state:
  out
  attention-domain lse
  optional q/k/v debug state
  metadata provenance
```

关键是：fused kernel 和 unfused graph 都要声明自己实现的是同一个 `LayerContract`。

### 3. GraphMaterialization

`GraphMaterialization` 描述同一个 layer contract 被如何落地。

例如：

```text
training materialization:
  qkv_proj -> qk_norm -> rope -> attention -> out_proj

rollout materialization:
  fused_qkv_rope_attention -> out_proj
```

两边计算图不同，但都绑定到同一个 semantic step 列表。

### 4. ParallelismPlan

`ParallelismPlan` 描述分布式语义。

```text
tp_world_size
cp_world_size
ep_world_size
sequence_sharding
head_sharding
expert_sharding
reduction_order
collective_transport
downcast_policy
```

它不只记录“用了几个 GPU”，还要记录每个 tensor 的逻辑 ownership 和 merge 规则。

### 5. SemanticTrace

`SemanticTrace` 是模型级训推一致的核心调试产物。

它记录每个重要语义边界：

```text
layer_id
semantic_op
materialization
backend
input_fingerprint
metadata_fingerprint
output_fingerprint
provenance
```

即使 rollout 侧把 RoPE 和 Attention 融合了，也要能在 trace 里声明：

```text
this fused op implements:
  rope
  attention
```

必要时 fused backend 提供 inspection mode，导出或重建虚拟边界上的校验状态。

## 训推一致判断流程

### 1. 前置一致性检查

先检查不属于 kernel 的条件：

- checkpoint version；
- tokenizer；
- token IDs；
- position IDs；
- attention mask；
- active loss mask；
- cache position；
- KV cache identity；
- sampling 后的 token sequence。

这些不一致时，不进入 kernel drift 判断。

### 2. 绑定同一个 LayerContract

training 和 rollout 的执行图必须都声明：

```text
model_profile_id
layer_contract_id
parallelism_plan_id
materialization_id
```

如果 contract 不一致，直接 fail。

### 3. 运行不同 materialization

允许两边不同：

```text
training:
  unfused reference / training fused path

rollout:
  paged-KV / decode / fused path
```

但所有路径都要导出 contract 要求的 state。

### 4. 对齐语义边界

优先比较：

- selected-logprob；
- attention `out` / `lse`；
- layer hidden state；
- backward `dq` / `dk` / `dv` / weight grad；
- metadata fingerprint。

对 fused op，如果无法导出中间状态，需要至少能导出最终 semantic boundary，并在 provenance 中说明不可见边界。

### 5. 生成 drift report

报告需要包含：

```text
model profile
layer contract
training materialization
rollout materialization
parallelism plan
backend
dtype
reduction order
cache metadata
max / p95 / p99 drift
bitwise_equal
fallback reason
```

## 接入新模型架构的流程

### Step 1：添加 ModelProfile

先把模型结构写清楚，不急着写 kernel。

必须明确：

- attention 类型；
- RoPE / position 规则；
- KV cache layout；
- head mapping；
- MLP / MoE 结构；
- logprob head；
- 需要哪些中间 state。

### Step 2：定义 LayerContract

每类 layer 都要有 contract：

```text
Norm
Attention
MLP
MoE
LMHead
LogProb
```

新模型不能直接接 backend，必须先接 contract。

### Step 3：定义 training / rollout materialization

同一个 layer 至少列出：

```text
reference path
training path
rollout path
```

例如 Qwen3 Attention：

```text
reference:
  qkv -> rope -> attention -> out

training:
  qkv -> rope -> flash attention

rollout:
  fused rope attention -> paged-KV decode
```

### Step 4：补 metadata validator

不同模型最容易 drift 的地方通常不是 matmul，而是 metadata。

必须验证：

- token order；
- position mapping；
- cache position；
- block/page table；
- expert route；
- packed offsets；
- loss mask；
- special token span。

### Step 5：建立 reference 和 drift suite

每个新模型接入至少要有：

- 单卡 reference；
- fused vs unfused 对齐；
- training vs rollout materialization 对齐；
- distributed plan 对齐；
- fallback 测试。

## 复用和优化点

### 可以复用

- WS1 的 batch-invariant op reference；
- WS2 的 `ShardingSpec` / `ReductionSpec`；
- #235 的 CP Attention metadata 和 `(out, lse)` merge；
- #116 的 drift report 格式；
- PR2 的 single-GPU attribution harness；
- TE optional oracle；
- existing operator registry；
- existing backend fallback policy。

### 建议新增

- `ModelProfileRegistry`；
- `LayerContractRegistry`；
- `GraphMaterializationRegistry`；
- `SemanticTrace` JSON schema；
- metadata fingerprint 工具；
- fused op inspection mode；
- model fixture generator；
- backend capability matrix。

### 可以优化

- 从 Hugging Face config 自动生成部分 `ModelProfile`；
- 把 Qwen3、Llama、DeepSeek MLA 的公共字段抽到 shared schema；
- 将 RoPE、Attention、KV-cache metadata validator 复用到 decode 和 prefill；
- 将 drift report 做成统一 CLI；
- 将 fused/unfused equivalence test 放进每个新 backend 的准入条件；
- 对 optional backend 统一记录 unavailable reason，而不是散落在各测试里。

## 分阶段路线

### Phase 0：保持当前 WS2 收敛

继续完成 #235：

```text
Qwen3-8B
TP=2
CP=2
BF16
Attention CP
```

这阶段不扩大模型范围。

### Phase 1：内部模型级 contract

在 Qwen3 上引入：

- `ModelProfile`;
- `LayerContract`;
- `GraphMaterialization`;
- `SemanticTrace`。

目标是证明同一个 Qwen3 layer 可以有 fused 和 unfused 两种 materialization，并能被同一个 drift report 判断。

### Phase 2：真实 training / rollout 计算图对齐

接入真实路径：

```text
training:
  Megatron / FSDP / DeepSpeed style graph

rollout:
  vLLM / sglang paged-KV graph
```

目标不是接管框架，而是在 RL-Kernel 内生成可验证 trace 和 drift report。

### Phase 3：多模型架构接入

优先顺序：

```text
Qwen3 dense
Llama-like dense
Qwen MoE
DeepSeek MLA
VLM / multimodal profile
```

每个模型先接 profile 和 contract，再接 backend。

### Phase 4：开放插件化接口

等内部 schema 稳定后，再考虑外部 plugin API。

外部 plugin 必须提供：

- `ModelProfile`;
- `LayerContract`;
- reference path；
- metadata validator；
- drift fixtures；
- fallback policy。

## 判断标准

一个新模型架构不能只证明“能跑”。它要证明：

```text
training graph 和 rollout graph 绑定到同一个 semantic contract
fused 和 unfused materialization 输出一致
不同 parallelism plan 输出一致
metadata 不一致时 fail loudly
unsupported backend 有明确 fallback
drift report 能定位第一处 divergence
```

这是 RL-Kernel 从算子库走向模型级训推一致层的关键边界。

## References

- #83: RL-Kernel Roadmap
- #235: WS2 CP-aware deterministic Attention
- vLLM plugin system: https://docs.vllm.ai/en/latest/design/plugin_system/
- Hugging Face Transformers custom models: https://huggingface.co/docs/transformers/custom_models
- NVIDIA Megatron-Core developer guide: https://docs.nvidia.com/megatron-core/developer-guide/latest/
