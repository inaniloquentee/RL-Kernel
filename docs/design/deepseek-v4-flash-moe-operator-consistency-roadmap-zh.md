# DeepSeek V4 Flash MoE 算子级训推一致性 Roadmap

> 状态：Draft，2026-08-18  
> 训练侧：Megatron  
> 推理侧：vLLM  
> 框架侧：Miles  
> 算子侧：KLR / RL-Kernel

本文按 DeepSeek V4 Flash MoE 的实际执行模块组织工作，不再按 WS1/WS2
分阶段描述。每个模块分别回答五个问题：模型语义是什么、Miles 框架层已经
固定了什么、KLR 算子层还缺什么、需要记录哪些 trace、怎样才算验收通过。

文中的 **Miles 已完成** 只表示框架已经提供该算子进行训推对比所需的输入
身份、生命周期、并行坐标、配置合同或诊断能力。它不表示对应 CUDA/Triton
算子已经实现，也不表示数值结果已经通过验收。真正的算子数值、离散输出、
布局和归约顺序仍由 KLR issue 跟踪。

## 1. 范围和工作模型

当前工作合同以 DSV4 白盒架构和 checkpoint 配置为最终依据：

- Layer 0-2 使用 Hash Router / `tid2eid`；输入为 `input_ids` 和绝对全局层号，
  输出为 `expert_ids`、`route_rank` 以及模型合同定义的 route weight/scale。
- Layer 3+ 使用 Learned Router；典型路径为 Linear、gate transform、归一化、
  Sort/Top-K，但具体 Softplus/softmax/renorm 语义必须从 checkpoint 固化，不能
  用通用 MoE 假设代替。
- 当前蓝图为 256 个 FP4 routed experts、每 token Top-6，以及 1 个 FP8 shared
  expert。若 checkpoint 改变这些值，必须先更新 `LayerContract`，不能把它们
  当作消融变量。
- Natural Route 是主验收路径。R3 route replay 只用于固定路由后的故障归因，
  必须标记 `diagnostic_only=true`，不能产生唯一的通过结论。
- MTP/DSpark speculative decoding 暂不进入本 Roadmap 的主验收。它会改变 token
  接受数、position/cache 状态和 RNG 轨迹，应在 MoE 主链完成后建立独立场景。

目标链路如下：

```text
input_ids + hidden_states + absolute_global_layer_number
    -> RMSNorm
    -> Hash Router (Layer 0-2) or Learned Router (Layer 3+)
    -> Global Route Metadata / capacity decision
    -> Canonical Gather + Pack
    -> EP All-to-All Dispatch
    -> FP4 Routed Experts + FP8 Shared Expert
    -> EP All-to-All Return
    -> Canonical Unpermute
    -> Fixed-order Weighted Combine
    -> Shared/Residual Add
    -> backward mirror path
    -> LM Head / selected-token logprob / dlogp
```

## 2. 全局验收合同

同一个有效 token 只有同时满足以下五层条件，才可以声明算子级训推一致：

1. **身份一致**：checkpoint、tokenizer、input ids、mask、position/cache metadata、
   absolute layer、weight version、模型配置和并行坐标完全一致。
2. **离散语义一致**：`expert_ids`、`route_rank`、capacity/overflow decision、
   valid mask、expert ownership 和 token-to-expert map 必须 exact match。
3. **布局一致**：send/recv count、offset、slot、padding、permutation、inverse map
   和 collective provenance 必须 exact match；通信到达顺序不属于模型语义。
4. **数值一致**：RMSNorm、expert MLP、combine、backward 和 partial reduction 先以
   deterministic reference 对齐，再按统一 dtype tolerance 验收 fast path。
5. **RL 输出一致**：仅在有效 action token 上比较
   `dlogp = train_recomputed_logp - rollout_old_logp`；zero-active、NaN/Inf、
   identity violation 或任一离散分叉直接失败。

任何模块不得仅用最终平均 KL 或 loss 较小来掩盖更早的离散或布局分叉。报告
必须给出首个不一致边界，而不是只给出末端误差。

## 3. Miles 与 KLR 的职责边界

| 层级 | Miles 框架层已完成 | KLR 算子层负责 |
| --- | --- | --- |
| Identity | fixture、checkpoint/tokenizer/weight version、input ids、mask、global layer 和并行 rank provenance | 验证算子不重新解释身份，不使用 rank-local id 替代 global id |
| Contract | `LayerContract`、dtype/shape/stride/device、top-k、expert namespace、capacity/overflow、quant format 和 eps | 严格执行合同中的数学、离散和低精度语义 |
| Lifecycle | forward metadata、activation/RNG、stream/event、buffer ownership 和 run identity | backward 复用正确 metadata，不读未 ready/过期 buffer |
| Parallel | TP/EP/DP/CP group、expert placement、topology、bucket/chunk、collective/overlap provenance | 固定 local tile、partial reduction、pack/unpack 和算子内归约语义 |
| Observability | versioned SemanticTrace、boundary fingerprint、first-mismatch report、R3 diagnostic flag | 输出必要的 route/layout/permutation/scale/reduction hash 和边界 tensor |
| Acceptance | action mask、比较 token、统一 tolerance、zero-active/non-finite fail-closed | 让 Natural Route、边界 tensor 和最终 dlogp 达到合同要求 |

## 4. 模块 Roadmap

### M0. 输入合同与 RMSNorm

#### 模块语义

MoE 的入口不是单独的 `hidden_states`。Hash 层还依赖完整 `input_ids` 和绝对
全局层号；所有层依赖正确的 valid mask、dtype、stride 和 RMSNorm `eps`。
RMSNorm 的输出是 Router 和 shared/routed expert 的共同输入，因此入口处的
漂移会污染后续所有模块。

#### Miles 已完成

- 固定同一 checkpoint、tokenizer、weight version、input ids、valid/action mask
  和绝对 global layer mapping。
- 固定 PP/virtual-PP stage 到 global layer 的映射，禁止接收端从 stage-local
  index 重新推导层号。
- 对输入 tensor 执行 shape、stride、dtype、device、storage offset 和 lifecycle
  校验；不满足合同的 payload fail-closed。
- 在 `LayerContract` 中固定 RMSNorm `eps`、输入/输出 dtype 和 observation dtype。
- SemanticTrace 可记录入口 tensor metadata、fingerprint、NaN/Inf 和 rank 坐标。

#### KLR 剩余 issue

- 实现 RMSNorm forward 的固定 reduction tree、accumulation dtype 和 downcast 点。
- 实现 RMSNorm backward 的 `dX`/`dGamma` 固定归约顺序，禁止 shape-dependent
  algorithm、未声明的 vectorization 或 atomic 改变语义。
- 对齐 batch=1/多 batch、packed/unpacked、padding 和非连续 stride；相同有效
  token 的输出不得随无关 token 改变。
- 为 CUDA/Triton/参考路径记录 backend、kernel version、launch policy 和实际
  fallback；strict case 禁止 silent fallback。

#### Trace 与验收

- Trace：RMSNorm input/output、mean-square/rsqrt 诊断值、accum dtype、eps、
  reduction policy、downcast 点和 kernel fingerprint。
- 离散前置条件：input identity 与 valid mask exact match。
- 验收：forward、`dX` 和 `dGamma` 满足 dtype tolerance；batch/padding sweep 下
  同一有效 token 保持不变；无新增 NaN/Inf。

KLR issue：`KLR-ISSUE-M0`

### M1. Hash Router / tid2eid（Layer 0-2）

#### 模块语义

Hash Router 不计算 learned gate logits，也不产生 gate-network gradient。其语义是：

```text
(input_ids, absolute_global_layer_number, LayerContract)
    -> expert_ids[T, top_k]
    -> route_rank[T, top_k]
    -> route_weight/scale (only if model-defined)
```

`route_rank` 表示某个 token 的 Top-K 路由位置，不是 EP rank。相同 token 的多个
route 可以落到同一个 EP rank，因此每个 rank 上接收到的是 route 集合，而不是
“每 token 一个值”。

#### Miles 已完成

- 固定 `router_mode_by_layer`、absolute layer、expert global namespace、
  `top_k=6`、route weight/scale 合同和 overflow policy。
- Natural Route 主验收禁止 route override；R3 replay 单独标记为诊断模式。
- 输入 payload 携带 `input_ids`、global layer、run id、rank coordinates 和
  `LayerContract` fingerprint。
- trace/comparator 能先比较 `expert_ids`/`route_rank`，再进入下游数值边界。

#### KLR 剩余 issue

- 依据 checkpoint 实现 bit-exact `tid2eid`，明确整数宽度、溢出语义、mod/hash
  常量、token id 范围和 global layer 编码。
- 定义并实现稳定的 `route_rank`，包括重复 expert、invalid token、padding 和
  capacity boundary 的行为。
- 固定 capacity/overflow 的执行结果和 overflow reason；不能只在 trace 中
  记录 policy 而由不同 backend 自行解释。
- 为 CUDA megakernel、Triton 和 reference 路径建立同一 golden vectors；
  `expert_ids`、`route_rank`、capacity decision 必须 bit-exact。
- 验证 TP/EP/DP、batch、pack、padding 和 overlap 不改变 Hash 输出；global
  token id 不能被 rank-local token index 替代。

#### Trace 与验收

- Trace：global token id、input id、global layer、expert id、route rank、
  route weight/scale、capacity decision、overflow reason 和 router fingerprint。
- 验收：所有离散字段 exact match；任何 route 分叉直接归类
  `F1_ROUTER_DISCRETE`，不得继续用最终 dlogp 容差放行。

KLR issue：`KLR-ISSUE-M1`

### M2. Learned Router（Layer 3+）

#### 模块语义

Learned Router 的实际 gate transform 必须来自 checkpoint。工作路径可以包含
Linear、Softplus、softmax/renorm 和 Top-K，但不能把某个 backend 的融合边界当作
模型语义。先比较连续 score，再比较离散 Top-K。

#### Miles 已完成

- `LayerContract` 固定 learned/hash layer range、gate transform、bias、renorm、
  top-k、tie-break、capacity/overflow 和 expert namespace。
- 固定 checkpoint 权重身份、input hidden-state boundary、backend feature gate 和
  kernel provenance。
- SemanticTrace 支持 logits/weights/Top-K ids 的版本化边界和 first mismatch。
- near-tie case 可使用相同 fixture、mask、seed 和 rank mapping 进行 paired run。

#### KLR 剩余 issue

- 建立 FP32 oracle：固定 Linear/GEMM reduction、bias、Softplus/softmax/renorm
  计算顺序以及 downcast 点。
- 定义全序 Top-K key，例如 `(score, expert_id)`；在 equal/near-equal score、
  NaN/Inf 和 signed zero 下保持一致。
- 明确权重是在 Top-K 前还是后归一化、route weight dtype，以及 backward 是否
  对 selected weights/gate 参数求导。
- CUDA/Triton/vLLM fused path 与 Megatron path 必须输出相同 top-k ids；只有 ids
  一致后，router weights 才进入数值 tolerance。
- 固定 capacity/overflow、aux-loss/load-balance 统计和 backward reduction；
  推理侧未使用的训练辅助分支不得改变主路由。

#### Trace 与验收

- Trace：router input、logits、post-transform score、Top-K ids/weights、
  k/k+1 margin、tie-break key、capacity/overflow 和 gate gradient 边界。
- 验收：Top-K ids、rank、capacity exact match；weights/logits/dgate 满足合同；
  near-tie、batch/padding 和多 rank case 均通过。

KLR issue：`KLR-ISSUE-M2`

### M3. Global Route Metadata、Canonical Pack 与 EP Dispatch

#### 模块语义

Router 输出必须先变成与物理到达顺序无关的全局 route record：

```text
(global_token_id, expert_id, route_rank, route_weight,
 capacity_decision, overflow_reason)
```

随后按 canonical key 生成 send/recv count、per-expert offsets、expert-local slot、
padding 和 permutation，再执行 token owner 到 expert owner 的 All-to-All。

#### Miles 已完成

- 固定 EP process group、expert placement、rank coordinates、topology 和 fallback
  policy；unsupported topology 可 fail-closed。
- 固定 global route metadata schema 和 forward run identity，保存 offset、slot、
  valid mask、padding、capacity 和 collective provenance 的生命周期。
- 记录 producer stream、ready event、buffer ownership 和 overlap schedule；
  backward 可校验 metadata 是否来自同一次 forward。
- trace schema 能记录 payload/permutation hash、send/recv count 和每 rank 边界。

#### KLR 剩余 issue

- 实现 deterministic Pack：使用 global key，固定 expert 分组、slot 分配、padding
  和 capacity truncation，不依赖线程调度或 atomic 到达顺序。
- 固定 send/recv count、rank/expert offsets 和 valid mask 的生成语义；同一 route
  必须得到同一 `(expert, slot)`。
- 实现 EP=1 和 EP>1 同一语义的 dispatch；不同 expert placement 只改变物理 rank，
  不改变 global route identity。
- 明确 overlap on/off 的 ready granularity。通信完成顺序可以不同，但 buffer 的
  逻辑索引、valid bit 和消费者依赖必须相同。
- 对 capacity boundary、重复 expert、zero-token expert、all-padding rank 和
  uneven send/recv count 建立回归。

#### Trace 与验收

- Trace：canonical route records、send/recv count、offset、slot、valid mask、
  padding、permutation hash、expert input boundary 和 collective sequence id。
- 验收：route map 和布局 metadata exact match；按 canonical key 重排后的 expert
  input exact/dtype match；arrival order 不得出现在 semantic hash 中。

KLR issue：`KLR-ISSUE-M3`

### M4. FP4 Routed Expert MLP

#### 模块语义

每个被路由的 token 执行 expert-local MLP：

```text
FP4 dequant -> Gate/Up GEMM -> SiLU -> Mul -> Down GEMM -> output downcast
```

模型语义由 expert id、FP4 format、scale 语义和 MLP 数学定义共同决定。Grouped
GEMM、Graph GEMM 或 megakernel 只是实现选择。

#### Miles 已完成

- 固定 global expert id 到权重 shard 的映射、checkpoint/weight version 和 expert
  placement。
- `LayerContract` 固定 FP4 format、scale granularity/axis、scale update/load
  policy、rounding、saturation、accum dtype 和输出 dtype。
- 固定 backend feature gate、kernel/build/version provenance、recompute mode、
  activation/RNG lifecycle 和实际 fallback。
- 可按 expert、rank、global token/route 定位输入输出 boundary。

#### KLR 剩余 issue

- 实现可审计的 FP4 reference dequant 和 deterministic Graph GEMM，固定 tile id、
  K 维遍历、accumulation tree、workspace 和 downcast。
- 禁止未声明的 split-K、atomic reduction、shape-dependent algorithm 或
  `#pragma unroll` 改变浮点运算顺序；若 fast path 使用，必须作为显式 backend。
- 对齐 fused/unfused Gate+Up、SiLU、Mul 和 Down 的中间精度与舍入点。
- 固定 scale load/update 顺序、subnormal/NaN/Inf/saturation 行为和 per-expert
  zero-token path。
- backward 中分别固定 `dX`、`dW_gate/up/down` 的 GEMM、partial reduction、
  scale/STE 语义和 downcast boundary。
- 验证 expert input 顺序变化不会在 canonical unpermute 后改变同一 route 输出。

#### Trace 与验收

- Trace：expert input/output、quantized payload hash、scale、dequant boundary、
  GEMM algorithm/workspace/tile/split-K、activation boundary 和 downcast 点。
- 验收：deterministic reference 先通过；fast path forward/dX/dW 满足 dtype contract；
  batch、expert load、EP placement 和 overlap sweep 无语义漂移。

KLR issue：`KLR-ISSUE-M4`

### M5. FP8 Shared Expert

#### 模块语义

Shared expert 处理所有 valid tokens，是独立于 routed experts 的连续分支。其输出
只能在 combine 后按合同加入一次，不能因为 EP/TP placement 或 fused kernel 被
隐式混入 routed reduction。

#### Miles 已完成

- 固定 shared expert 开关、权重身份、FP8 format、scale/dtype contract 和模型中
  “shared once”的语义。
- 固定 replicated 或 TP-sharded placement、process group、backend feature gate 和
  provenance。
- 单独保存 `shared_out[T,H]`、forward activation 和 backward lifecycle；能够与
  routed branch 分开诊断。
- 保留 shared backward edge 和 shared gradient 的 collective provenance。

#### KLR 剩余 issue

- 实现 FP8 quant/dequant、scale update/load、Gate+Up、SiLU、Mul、Down 的固定数学
  和舍入点。
- 固定 replicated 与 TP-sharded 实现的 partial reduction；二者在全局语义上
  必须一致。
- forward 保证每个 valid token 计算一次，padding/invalid token 不进入 scale
  统计或输出。
- backward 保证 `dX_shared`、`dW_shared` 只计算/归约一次；固定 DP all-reduce
  bucket/order 对 shared gradient 的影响。
- 明确 shared kernel 与 routed/communication overlap 的 stream/event 依赖，避免
  读未完成输出或重复加入。

#### Trace 与验收

- Trace：shared input/output、FP8 scale、GEMM/activation boundary、placement、
  partial reduction、`dX_shared` 和 `dW_shared`。
- 验收：独立 shared branch forward/backward 通过；replicated/TP-sharded、
  batch/padding 和 overlap on/off 下同一 valid token 结果满足合同。

KLR issue：`KLR-ISSUE-M5`

### M6. All-to-All Return、Canonical Unpermute 与 Inverse Map

#### 模块语义

Expert output 返回 token owner 后，不能按网络到达顺序恢复 token。Pack 时必须
生成 canonical inverse map：

```text
packed_index = rank_offset[route_rank]
             + expert_offset[expert_id]
             + slot

inverse_map[packed_index] = (global_token_id, topk_index)

permutation_hash = H(permutation, inverse_map, offsets,
                     slots, valid_mask, capacity)
```

inverse map 固定的是“逻辑 packed index 到 token/Top-K 位置”的映射，不是强制
通信消息按固定物理顺序到达。

#### Miles 已完成

- 保存本次 forward 的 offset、slot、valid mask、capacity、run identity 和 payload
  version，禁止 backward/return 复用错误 run 的 metadata。
- 固定 All-to-All group/split/topology 和 stream/event lifecycle；通信到达顺序只
  作为执行 provenance，不作为模型语义。
- SemanticTrace 支持 inverse-map/layout fingerprint 和 forward/backward 对称校验。

#### KLR 剩余 issue

- 实现 canonical inverse map 和 unpermute；每个 returned expert output 必须回填到
  正确 `(global_token_id, topk_index)`。
- 固定 rank/expert offset 的定义和冲突检查；invalid/padding/overflow slot 不得
  产生有效输出。
- 在 forward return 和 backward gradient return 复用同一布局语义，并校验
  `permutation_hash`。
- 支持 chunked/overlapped return：已 ready 的 canonical segment 可以提前回填，
  但不能提前改变最终 route_rank 归约顺序。
- 建立 out-of-order arrival、zero-count peer、uneven expert load 和 metadata
  corruption 的 fail-closed 测试。

#### Trace 与验收

- Trace：packed index、inverse map、offset/slot/valid mask、permutation hash、
  arrival provenance 和 restored `[T,K,H]` boundary。
- 验收：逻辑映射 exact match；任意消息到达顺序下恢复结果相同；错误/过期
  metadata 必须拒绝，不能静默猜测布局。

KLR issue：`KLR-ISSUE-M6`

### M7. Fixed-order Weighted Combine、Shared Add 与 Residual

#### 模块语义

对 token `t`，routed output 的数学语义为：

```text
routed[t] = sum(route_rank=0..K-1,
                route_weight[t, route_rank] *
                expert_out[t, route_rank])
output[t] = residual[t] + routed[t] + shared_out[t]
```

权重属于 token 的某个 Top-K route，而不是 EP rank 的权重。某个 EP rank 可以拥有
该 token 的多个 routes，因此不能先按 rank 任意归约再假设结果等价。

#### Miles 已完成

- 固定 route weight/scale、accum/output dtype、downcast contract 和 route metadata
  lifecycle。
- 固定 shared branch 只加入一次、residual forward/backward edge 必须保留。
- trace/comparator 可分别观察 routed combine、shared output、residual 和最终 MoE
  output，不让 fused boundary 隐藏首个漂移点。
- 固定 TP/EP group、partial-output ownership 和 combine backend provenance。

#### KLR 剩余 issue

- 固定 expert output 与 `(global_token_id, topk_index, route_weight)` 的配对。
- 按 `route_rank=0..5` 固定累加顺序、accum dtype 和 downcast 点；通信 ready 顺序
  不得决定浮点加法顺序。
- 设计支持 overlap 的实现：可以提前完成乘法、写入 per-route staging buffer，
  但只有下一个 canonical route ready 时才能推进该 token 的 ordered reducer。
- 明确 shared、routed、TP partial 和 residual 的加法顺序，禁止 shared/residual
  漏加或重复加入。
- backward 中 `dExpertOut[t,r] = route_weight[t,r] * dY[t]` 只乘一次权重，并对
  overflow/invalid route 执行 mask；`dResidual=dY` 必须保留。
- 若 Learned Router 需要 weight gradient，单独固定 `dWeight` dot/reduction；
  Hash Router 不得产生不存在的 gate gradient。

#### Trace 与验收

- Trace：output-weight pairing、route-ready bitmap、combine order hash、accum dtype、
  downcast 点、routed/shared/residual boundaries 和 backward mask。
- 验收：不同 EP arrival/overlap、batch/padding 和 TP layout 下最终 ordered combine
  相同；forward、dExpertOut、dWeight（如适用）和 residual gradient 通过。

KLR issue：`KLR-ISSUE-M7`

### M8. MoE Backward 编排与 Expert Gradient

#### 模块语义

Backward 必须镜像 forward 布局：先从 combine backward 生成每个 route 的
`dExpertOut`，按 forward inverse map dispatch 到 expert owner，执行 routed/shared
expert backward，再 All-to-All return，canonical unpermute，并按固定 route 顺序
归约 `dX_route + dX_shared`。Hash 层不产生 gate-network gradient。

#### Miles 已完成

- 保存 forward global route key、weight、capacity、slot、offset、valid mask、
  inverse map、activation/RNG 和 run identity。
- 固定 forward/backward 对称的 EP group/split/topology 和 metadata lifecycle。
- 固定 TE deterministic/reference mode feature gate、recompute policy、RNG/activation
  provenance 和 backend version。
- 保留 residual backward edge，并固定 DP bucket/chunk/collective order provenance。

#### KLR 剩余 issue

- combine backward 只应用一次 route weight；overflow、padding 和 invalid route
  不得收到梯度。
- gradient dispatch/return 复用 forward inverse map 和 layout fingerprint，不得按
  backward 的物理 arrival 重新分配 slot。
- routed/shared expert backward 固定 dX/dW GEMM、split-K/atomic、workspace、fusion、
  scale 和 downcast boundary。
- `dX_route` 按 `route_rank=0..5` 固定归约，再与 `dX_shared` 相加一次；禁止二次
  乘 weight、arrival-order reduction 或 shared 重复加入。
- RMSNorm backward、residual gradient add 和 TP/DP partial reduction 必须使用明确
  的 accum dtype 和 order。
- activation recompute on/off 应在相同数学合同下产生一致梯度；若不能，必须
  显式分 backend 并 fail-closed。

#### Trace 与验收

- Trace：saved forward fingerprint、dExpertOut、gradient dispatch layout、expert
  dX/dW、return inverse map、dX reduce order、dX_shared、dGamma 和 dResidual。
- 验收：forward metadata corruption/错 run 可被拒绝；dX/dW/dGamma/dResidual 在
  recompute、EP/TP/DP 和 overlap sweep 下通过；Hash 层无多余 dgate。

KLR issue：`KLR-ISSUE-M8`

### M9. TP/EP/DP Collective 与通信计算 Overlap

#### 模块语义

Collective 自身不是新的模型数学，但它会改变 pack 布局、partial output、归约树
和 ready 顺序。框架固定“参与者与调度合同”，算子必须固定“局部 tile 到全局
结果”的数值与布局语义。

#### Miles 已完成

- 固定 TP/EP/DP/CP process groups、rank coordinates、expert placement、topology、
  NCCL/backend、collective sequence、bucket/chunk 和 overlap provenance。
- 构建/进程生命周期配置可读回实际值；strict case 下未声明 fallback 失败。
- stream/event、producer/consumer readiness 和 buffer ownership 可追踪。
- paired report 可输出 per-rank boundary、collective fingerprint 和 first mismatch。

#### KLR 剩余 issue

- TP Graph GEMM 固定 local tile、K partition、partial output dtype 和全局 reduction
  tree；不能用 TP MoE 测试替代单独 Graph GEMM 证明。
- EP dispatch/return 固定 counts/offset/slot/inverse map；EP=1 与 EP>1 只允许物理
  placement 差异。
- DP expert/shared gradients 固定 bucket/chunk/order 或提供 deterministic reference；
  不得把 collective drift误归因于 MoE kernel。
- overlap on/off 必须保持相同 semantic hash 和最终 reduction order。可以提前做
  独立 per-route compute，但不能以 ready order 代替 route order。
- unsupported topology、NCCL algorithm 或 capacity/layout 组合必须 fail-closed，
  并记录实际 backend；不能静默切换未验证实现。

#### Trace 与验收

- Trace：group membership、rank mapping、collective sequence、algorithm、counts、
  partial output、ready events、overlap windows 和 reduction hash。
- 验收：TP/EP/DP 单独 OAT 以及必要 pairwise case 通过；per-rank 边界和 global
  output 均满足合同；overlap on/off 不改变结果。

KLR issue：`KLR-ISSUE-M9`

### M10. Natural Route、SemanticTrace 与最终 dlogp Gate

#### 模块语义

最终验收必须把 rollout 侧自然计算的 Router 与 training recompute 侧自然计算的
Router 对齐。比较顺序为：identity -> route metadata -> dispatch/layout -> expert
boundary -> combine -> backward -> valid-action dlogp。R3 replay 只能在 Natural
Route 失败后帮助判断问题属于 Router 还是后续算子。

#### Miles 已完成

- 固定同一 fixture、checkpoint/tokenizer/weight version、action/attention mask、
  position/cache metadata 和 TP/EP/DP/CP mapping。
- versioned SemanticTrace、payload schema、boundary fingerprint、tolerance lookup、
  first-mismatch 分类和 per-rank report 已具备。
- 固定比较 token、zero-active/non-finite fail-closed 规则和 dlogp 统计口径。
- R3 validated hand-off 校验 stride/dtype/shape/version/provenance，并强制
  `diagnostic_only=true`。

#### KLR 剩余 issue

- 为 M0-M9 算子暴露统一 trace hook，确保 fused kernel 不隐藏 route/layout/
  expert/combine 等语义边界。
- 实现 Natural Route paired adapter，先 exact compare 离散和布局字段，再执行数值
  comparator；任一 identity violation 直接停止。
- 汇总 per-rank first mismatch、worst token/route/expert、kernel/backend fingerprint、
  permutation/collective hash 和 NaN/Inf。
- 对有效 action token 计算 `max_abs_dlogp`、percentile 和 worst-token metadata，
  pass/fail 只能引用统一 tolerance contract，不能复制阈值。
- R3 case 必须与 Natural Route 结果同时报告：Replay 通过但 Natural Route 失败时，
  结论只能是 Router contract 未对齐。

#### Trace 与验收

- Trace：M0-M9 全边界、requested/actual provenance、valid-action selected logprob、
  dlogp、first mismatch 和诊断分类。
- 验收：Natural Route 的 identity/route/layout 先通过，再满足所有数值 boundary 和
  active-token dlogp；R3 不得覆盖主结论。

KLR issue：`KLR-ISSUE-M10`

## 5. 模块依赖关系

Roadmap 不按单卡/多卡阶段切分，但模块仍有严格依赖：

```text
M0 Input/RMSNorm
  -> M1 Hash Router / M2 Learned Router
  -> M3 Route Metadata + Pack + Dispatch
  -> M4 Routed Expert + M5 Shared Expert
  -> M6 Return + Unpermute
  -> M7 Combine + Residual
  -> M8 Backward
  -> M9 Parallel/Overlap closure
  -> M10 Natural Route + dlogp gate
```

M9 不是最后才开始的“大并行阶段”。每个前置模块实现时都要带对应的 parallel
metadata 和局部测试；M9 只负责把 TP/EP/DP collective 与 overlap 的跨模块不变量
收口。M10 同理，不负责修复算子，只负责用统一合同判定和定位。

## 6. 按模块使用跨配置消融框架

复用 [RL-Kernel PR #230](https://github.com/RL-Align/RL-Kernel/pull/230) 的
`cross_config.experiment_config.v1`：每个模块定义一个 reference baseline，先跑
`one_at_a_time`，只有 OAT 不能解释失败时才添加少量 `pairwise_paths`。不构造
无约束 Cartesian product。

不能作为消融轴的模型语义包括：router mode、absolute layer mapping、expert 数、
Top-K、shared-expert 开关、FP4/FP8 format、gate transform、tie-break、capacity/
overflow、checkpoint/tokenizer、position/cache 语义和 action mask。

| 模块 | Reference baseline | OAT 实现变量 | 必要 pairwise |
| --- | --- | --- | --- |
| M0 RMSNorm | FP32/fixed-tree oracle | backend、accum dtype、vectorization、downcast | backend x packed stride |
| M1 Hash Router | integer golden vectors | CUDA/Triton/reference、fusion | backend x padding；backend x EP overlap |
| M2 Learned Router | FP32 logits/Top-K oracle | GEMM、gate transform backend、Top-K backend | backend x near-tie；backend x capacity |
| M3 Dispatch | deterministic pack、EP=1 | EP size、placement、padding、overlap | placement x overlap；padding x capacity |
| M4 Routed Expert | FP4 deterministic Graph GEMM | GEMM、dequant/fusion/unroll、scale path | GEMM x scale；GEMM x EP layout |
| M5 Shared Expert | FP8 independent reference | backend、scale path、replicated/TP-sharded | shared backend x combine backend |
| M6 Unpermute | canonical inverse map | chunking、arrival order、overlap | return overlap x uneven load |
| M7 Combine | route-rank ordered FP32 reference | combine backend、tree、downcast/fusion | combine x TP/EP partial output |
| M8 Backward | saved-forward deterministic reference | recompute、TE mode、dX/dW backend | recompute x overlap；GEMM x collective |
| M9 Parallel | no-overlap deterministic collectives | TP/EP/DP、algorithm、bucket/chunk、overlap | EP x overlap；TP x combine |
| M10 Acceptance | Natural Route | trace on/off、R3 diagnostic | Natural vs R3 仅用于归因 |

每个 case 必须记录 requested 和 actual provenance。silent fallback、不同 identity、
zero-active 或 non-finite case 不属于“数值未通过”，而属于无效实验。

## 7. Issue 拆分与完成标准

| 模块 | KLR issue | 完成信号 |
| --- | --- | --- |
| M0 Input/RMSNorm | `KLR-ISSUE-M0` | forward/backward fixed reduction，通过 batch/padding/stride sweep |
| M1 Hash Router | `KLR-ISSUE-M1` | tid2eid/route_rank/capacity bit-exact |
| M2 Learned Router | `KLR-ISSUE-M2` | logits/weights 在合同内，Top-K/tie-break exact |
| M3 Pack/Dispatch | `KLR-ISSUE-M3` | global route map、counts/offset/slot/permutation exact |
| M4 FP4 Routed Expert | `KLR-ISSUE-M4` | deterministic Graph GEMM 与 FP4 forward/dX/dW 通过 |
| M5 FP8 Shared Expert | `KLR-ISSUE-M5` | shared branch forward/backward 独立通过且只加入一次 |
| M6 Return/Unpermute | `KLR-ISSUE-M6` | inverse map/layout fingerprint exact，arrival-order independent |
| M7 Combine/Residual | `KLR-ISSUE-M7` | route-rank ordered combine 与 gradient 通过 |
| M8 Backward | `KLR-ISSUE-M8` | forward-layout reuse、expert dX/dW、dX reduce 和 residual gradient 通过 |
| M9 Parallel/Overlap | `KLR-ISSUE-M9` | TP/EP/DP 与 overlap on/off 不改变语义 |
| M10 Acceptance | `KLR-ISSUE-M10` | Natural Route 全边界与 valid-action dlogp 通过 |

单个 issue 的关闭必须同时满足：

1. 有 reference implementation 和明确的 semantic/tensor contract；
2. 有至少一个失败复现和对应 first-mismatch trace；
3. 离散/布局字段 exact match，数值字段引用统一 tolerance contract；
4. 覆盖该模块的 batch、padding、pack、dtype 和相关 parallel OAT；
5. 记录 actual kernel/backend/build/collective provenance；
6. unsupported configuration fail-closed；
7. 有性能/显存数据，但不能为了性能放宽模型语义或验收阈值。

## 8. 全链完成定义

DSV4 Flash MoE 算子级训推一致完成时，应满足：

- Hash 与 Learned Router 都在 Natural Route 下独立通过，所有离散 route exact；
- Dispatch/Return 的 global route map、slot、inverse map 和 permutation hash exact；
- FP4 routed 与 FP8 shared expert 的 forward/backward 通过，scale 和 downcast 可追踪；
- combine/gradient reduce 不依赖网络 arrival、atomic 或 overlap ready 顺序；
- TP/EP/DP、batch、pack、padding 和目标 dtype 配置下无 silent fallback；
- 报告能定位首个 mismatch 到 M0-M10，而不是只给最终 loss/KL；
- 有效 action token 的 dlogp 通过统一合同，无 zero-active 和新增 NaN/Inf；
- R3 仅作为诊断，不是通过 Natural Route 验收的必要条件。
