<!--
Draft source: #83 roadmap, WS1 / WS2 Hopper planning, #108 / #111 / #207, and
the vime train-inference consistency note in vime/home.txt. "B-card" here means
NVIDIA Blackwell, not Biren or any domestic accelerator. For a public GitHub
issue, keep the local vime/home.txt as background context and link the public
tracking issues when possible.
-->

# Context

This issue scopes the Blackwell ("B-card") version of the WS1 / WS2 operator
consistency plan for RL-Kernel.

B-card means NVIDIA Blackwell-class CUDA GPUs, such as B200 / GB200 data-center
systems and, when relevant, RTX Blackwell systems. It does not mean Biren or a
domestic accelerator. Domestic accelerator work belongs under #83's P3 domestic
accelerator lane and #196-style scoping issues; this issue belongs under the
CUDA architecture-specific lane.

The Hopper / H-card plan in #83 currently focuses WS1 and WS2 on:

- WS1: a full batch-invariant standard-Transformer forward chain.
- WS2: distributed / parallelism invariance across TP, SP, CP, and mismatched
  rollout-vs-training configs.
- The first WS2 end-to-end target: Qwen3-8B, TP=2, CP=2, BF16.
- Shared contract work from #108 / #111 / #207: `dlogp`, tolerance policy,
  per-rank drift reports, `ShardingSpec`, `ReductionSpec`, and deterministic
  reduction discipline.

This Blackwell issue does not replace the Hopper work. It ports the same
operator-level consistency contract to Blackwell, adds Blackwell build and
dispatch gates, and verifies that Blackwell fast paths do not silently diverge
from the Hopper-tested semantics.

Relevant public references:

- #83: RL-Kernel roadmap and WS1 / WS2 tracking.
- #108: WS1 numerical contract.
- #111: WS2 cross-config alignment.
- #207: WS2 cross-config logprob drift contract.
- NVIDIA Blackwell compatibility guide:
  https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html
- NVIDIA Blackwell tuning guide:
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- CUTLASS Blackwell documentation:
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell.html

# Goal

Make the WS1 / WS2 operator consistency stack run on Blackwell without weakening
the Hopper contract.

Concretely, the Blackwell path must prove:

1. Blackwell kernels are selected only when the exact architecture and capability
   checks pass.
2. Every enabled Blackwell operator has a correctness-preserving fallback.
3. WS1 batch-invariant operators remain invariant on Blackwell across batch
   size, padding layout, chunked prefill, and KV-cache mode.
4. WS2 TP / SP / CP distributed paths produce aligned logprobs under the shared
   #108 / #207 tolerance policy.
5. Any Blackwell-specific fast path, NCCL behavior, NVLink / NVLS path, or
   library heuristic is visible in the report and does not become a silent source
   of rollout-training mismatch.

The success claim should be:

```text
Operator-level train-inference consistency, validated on Blackwell for the
standard Transformer chain and TP/SP/CP distributed paths.
```

It should not claim real vLLM == real Megatron full-engine alignment unless WS3
also validates that separately.

# Reuse-first implementation rule

This issue is a backend-extension issue, not a parallel WS1 / WS2 framework.
Blackwell work should extend the Hopper path by adding architecture and backend
axes to the same operators, specs, runners, reports, and tolerance contracts.

Default rule:

```text
reuse first; add a Blackwell axis second; fork only with an explicit reason
```

Concretely:

- Do not create a Blackwell-only numerical contract. Reuse #108 / #207.
- Do not create a Blackwell-only operator registry or gtest path. Extend the
  existing WS1 operator specs and candidate backend selection.
- Do not create a Blackwell-only distributed runner. Extend the WS2
  cross-config / TP / SP / CP runners with architecture, topology, and NCCL axes.
- Do not create Blackwell-only report formats. Add Blackwell fields to the
  existing dlogp / per-rank / A0-A5 reports.
- Do not fork vime integration semantics. Reuse the same operator adapter,
  fallback, decision record, and consistency-path hooks used by the Hopper plan.
- Add new kernels only when the existing CUDA / Triton / PyTorch reference path
  is missing, unsafe on Blackwell, or too slow after correctness has already
  been established.

# Hopper-to-Blackwell reuse matrix

| Area | Reuse from Hopper WS1 / WS2 | Blackwell-only delta |
| --- | --- | --- |
| Numerical contract | #108 tolerance table; #207 `dlogp`, `ratio0`, `clipfrac0`, `approx_kl0`, per-rank drift | Add `cuda_sm100` / `cuda_sm120` / PTX-JIT fields to reports |
| Operator specs | Existing gtest `OperatorSpec` entries and forward-chain registration | Add Blackwell candidates as new backend choices |
| Reference path | PyTorch FP32 / BF16 reference and Hopper-validated CUDA/Triton semantics | Validate native Blackwell cubin vs PTX fallback vs reference |
| WS1 chain | Same RMSNorm, matmul, attention, LM head, logprob, RoPE, SwiGLU, embedding chain | Add Blackwell arch axis to the same batch-invariance sweep |
| WS2 sharding | Existing `ShardingSpec`, `Placement`, TP/SP/CP metadata | Add Blackwell topology and NCCL/NVLS metadata |
| WS2 reduction | Existing `ReductionSpec`, deterministic `in_op` reference, fixed reduction order | Compare NCCL / NVLS / NVLink-Sharp fast candidates against reference |
| Audit profiles | Existing A0-A5 matrix and result cube | Add Blackwell arch, native cubin/PTX, library algorithm axes |
| vime smoke | Existing operator adapter, fallback policy, teacher-forcing replay, `dlogp` report | Run the same smoke on B200 / GB200 once hardware is available |
| CI / packaging | Existing CUDA extension and hardware-aware dispatch infrastructure | Add sm100/sm120 build, load, and negative-dispatch gates |

# Scope

## In scope

- Blackwell architecture detection and dispatch:
  - `sm100` / compute capability 10.x data-center Blackwell when exposed by the
    validation node.
  - `sm120` / compute capability 12.x RTX Blackwell if that is the available
    validation hardware.
  - PTX fallback and native cubin behavior recorded separately.
- CUDA build and packaging updates for Blackwell:
  - native Blackwell cubins where supported by the selected CUDA toolkit;
  - PTX compatibility path;
  - no accidental SM90-only dispatch on Blackwell.
- Blackwell backend-axis validation for the existing WS1 single-card
  batch-invariant operator chain:
  - RMSNorm;
  - matmul / GEMM / projection;
  - attention and KV-cache attention;
  - LM head + selected logprob;
  - `linear_logp`;
  - RoPE, SwiGLU / SiLU, embedding;
  - ratio / KL and GRPO loss fragments where already in the forward / loss
    validation chain.
- Blackwell backend-axis validation for the existing WS2 distributed and
  cross-config invariance runners:
  - TP matmul / projection and vocab-parallel `linear_logp`;
  - SP-aware logprob / loss reductions and RMSNorm boundary cases;
  - CP attention LSE merge and CP layout preservation;
  - AllReduce, ReduceScatter, AllGather, and A2A metadata;
  - NCCL algorithm / protocol / topology / NVLink / NVLS / NVLink-Sharp axes when
    exposed by the system.
- vime smoke validation using the same operator-level contract as the Hopper
  plan.
- Validation reports with hardware facts, CUDA / driver / library versions,
  dtype, shape, selected backend, fallback reason, and drift metrics.

## Out of scope for the first Blackwell milestone

- Broad performance claims before correctness and fallback are locked.
- Real-engine WS3 closure between production vLLM / sglang and production
  Megatron / FSDP. This issue prepares the operator layer for WS3.
- FP8 / MXFP8 / NVFP4 strict rollout-training alignment. Blackwell may make these
  attractive, but the first gate remains FP32 reference + BF16 target, matching
  the current WS1 / WS2 scope.
- MoE routing consistency, pipeline parallelism, and multimodal operators.
- Domestic accelerator support.
- Duplicating Hopper WS1 / WS2 harnesses, tolerance tables, operator specs,
  sharding APIs, reduction APIs, or report schemas.

# Hardware Facts Required Before Enabling Blackwell Fast Paths

Please fill this table from the validation node before enabling any optimized
Blackwell backend by default.

| Field | Value |
| --- | --- |
| Vendor | NVIDIA |
| Device model | TBD, e.g. B200, GB200, RTX 50-series |
| Number of GPUs | TBD |
| Interconnect | PCIe / NVLink / NVSwitch / GB200 NVL topology TBD |
| Driver version | TBD |
| CUDA toolkit version | TBD |
| CUDA runtime version | TBD |
| PyTorch version | TBD |
| `torch.version.cuda` | TBD |
| `torch.cuda.is_available()` | TBD |
| `torch.cuda.get_device_name(0)` | TBD |
| `torch.cuda.get_device_capability(0)` | TBD |
| Native cubins built | TBD, e.g. sm100 / sm120 |
| PTX fallback included | TBD |
| CUTLASS version | TBD |
| cuBLAS / cuBLASLt version | TBD |
| cuDNN version if used | TBD |
| NCCL version | TBD |
| NCCL topology report | TBD |
| Available collectives | NCCL Ring / Tree / NVLS / other TBD |
| Supported dtypes for this issue | FP32 reference, BF16 target; FP16 optional |
| Blackwell validation owner | TBD |

Acceptance rule: optimized Blackwell dispatch remains disabled until the platform
detector and the operator-specific capability check both pass. Blackwell support
must not be inferred only from `device.type == "cuda"`.

# Done Definition

An item in this issue is done only when it has all of the following:

1. A code path in the repository.
2. Focused tests or a reproducible smoke test.
3. A validation or benchmark report with hardware, dtype, shapes, backend,
   fallback behavior, command lines, and drift metrics.
4. Clear fallback behavior for unsupported shapes, dtypes, toolkit versions,
   architectures, or optional libraries.
5. For consistency-sensitive operators, a `dlogp` or operator-drift report using
   the shared #108 / #207 tolerance policy.

# Workstream 0 - Blackwell Platform Detection, Build, and Dispatch Gates

Goal: make Blackwell a CUDA architecture-specific target without accidentally
reusing Hopper-only assumptions. This is the main Blackwell-only workstream; it
should feed architecture facts into the existing CUDA dispatch infrastructure.

- [ ] Add or extend a CUDA architecture probe that records device capability,
      device name, native cubins available, PTX fallback availability, driver,
      CUDA runtime, PyTorch, CUTLASS, cuBLASLt, and NCCL versions.
- [ ] Define stable backend keys for dispatch and reports, for example:
      `cuda_sm90`, `cuda_sm100`, `cuda_sm120`, and `cuda_ptx_jit`.
- [ ] Update CUDA extension build flags so Blackwell-compatible binaries include
      the required PTX and, when supported by the selected toolkit, native
      Blackwell cubins.
- [ ] Add a PTX JIT smoke mode using `CUDA_FORCE_PTX_JIT=1` to prove forward
      compatibility where native cubins are absent.
- [ ] Prevent SM90-only kernels from being selected on Blackwell unless they are
      explicitly proven compatible and reported as a PTX/JIT fallback path.
- [ ] Add negative tests for unsupported arch-specific paths, including SM90 TMA
      / WGMMA specializations that compile but should not be treated as strict
      Blackwell paths.
- [ ] Extend benchmark and telemetry reports with the selected CUDA architecture
      backend and fallback reason.

Suggested PR split:

- PR0-a: Blackwell platform facts probe and report fields.
- PR0-b: CUDA architecture backend keys and registry priority skeleton.
- PR0-c: Blackwell build flag / PTX compatibility updates.
- PR0-d: negative dispatch tests preventing silent SM90 fallback.

# Workstream 1 - Shared Numerical Contract and Harness Reuse

Goal: reuse the Hopper WS1 / WS2 consistency contract instead of inventing a
Blackwell-only tolerance story. Blackwell PRs should add backend / architecture
axes to the existing harness, not create a separate harness.

Core metric:

```text
dlogp = training-side recomputed logp - rollout-side old logp
```

Only active response / action tokens count. Prompt tokens, padding, and masked
tokens are excluded. Training-side scoring must be teacher-forcing on the exact
same token sequence.

Required diagnostics:

- `max_abs_dlogp`
- mean / p95 / p99 / max of `abs_dlogp`
- `ratio0 = exp(dlogp)`
- `approx_kl0 = mean(exp(dlogp) - 1 - dlogp)`
- `clipfrac0`
- per-rank versions for distributed tests
- selected backend, fallback reason, dtype, shape, seed, and platform
  fingerprint

Harness tasks:

- [ ] Ensure `scripts/check_operator.py`, gtest specs, and the existing
      cross-config comparison harness can select Blackwell candidates by backend
      key.
- [ ] Reuse the same WS1 forward-chain operator set, not only `logp` and
      `linear_logp`: RMSNorm, matmul, attention, KV-cache attention, RoPE,
      SwiGLU / SiLU, embedding, LM head, ratio / KL, and GRPO loss. If an entry
      is missing, add it to the shared WS1 specs instead of a Blackwell-only
      list.
- [ ] Add Blackwell axes to the existing result cube: arch, native cubin vs PTX
      JIT, cuBLASLt algorithm where observable, CUTLASS kernel id where
      observable, NCCL algorithm/protocol where observable.
- [ ] Add input-parity prechecks: token IDs, masks, padding side, position IDs,
      temperature, top-p, top-k, cache metadata, sharding metadata, and reduction
      contract must match before operator drift is reported.
- [ ] Add a fallback correctness mode where Blackwell uses PyTorch native for all
      ops and must produce a zero-update `dlogp` report within the shared
      tolerance table.

# Workstream 2 - WS1 Blackwell Single-Card Operator Parity

Goal: before testing communication, add Blackwell fallback / candidate validation
to each existing WS1 operator. Operator signatures, metadata, mask semantics, and
report fields should stay shared with Hopper unless there is a documented
architecture-specific reason.

Priority follows the Hopper/vime drift analysis:

```text
attention > LM head + logprob > matmul / GEMM > RMSNorm > RoPE / SwiGLU / embedding
```

| Operator | Why it matters | First fallback | Blackwell candidate | Validation |
| --- | --- | --- | --- | --- |
| selected `logp` | PPO / GRPO ratio and KL depend directly on this value | PyTorch native online log-softmax | Blackwell selected-logprob kernel, portable Triton, or adapted CUDA path after proof | batch-size / padding / row-index invariance, forward + backward |
| `linear_logp` / LM head | avoids materializing `[tokens, vocab]`; large-vocab softmax is numerically sensitive | PyTorch native chunked path | Blackwell fused GEMM + online softmax; CUTLASS/cuBLASLt-backed candidate where deterministic enough | compare to FP32 reference, large vocab, global target, optional TP |
| attention | highest drift impact; flash vs paged / chunked paths diverge easily | PyTorch SDPA / standard attention | Blackwell FlashAttention candidate with exported LSE | causal, varlen, LSE export, KV-cache prefill vs decode |
| matmul / GEMM | QKV, MLP, output, and LM-head projection all depend on reductions | PyTorch `matmul` | deterministic Blackwell GEMM or locked CUTLASS/cuBLASLt path | batch-invariant, split-K disabled or locked, forward + backward |
| RMSNorm | small per-layer drift can accumulate across depth | PyTorch fp32-accum reference | Blackwell fused RMSNorm | reduction order locked, forward + backward |
| RoPE | layout and sin/cos precision must match | PyTorch fp32 sin/cos reference | fused RoPE only after layout proof | interleaved / half-rotation layout parity |
| SwiGLU / SiLU | fusion order can change rounding | PyTorch reference | fused activation | forward + backward, shape sweep |
| embedding | usually stable but input alignment depends on it | PyTorch lookup | optimized gather only if needed | no token remapping, no dtype surprise |
| ratio / KL | front-end for PPO / GRPO; sensitive to selected logprob | PyTorch native | fused CUDA / Triton candidate | forward + backward, mask semantics |
| GRPO loss | final RL objective reduction | PyTorch native | fused loss candidate | group normalization, leave-one-out, masks |

Single-card exit gate:

- [ ] Every operator above has an explicit fallback.
- [ ] Unsupported Blackwell candidates fail closed with a clear message.
- [ ] The same harness commands used for Hopper pass with only the backend /
      architecture axis changed.
- [ ] The assembled single-card forward chain reports `dlogp` within the shared
      #108 tolerance table for BF16.
- [ ] Same logical sample is stable across batch=1/N, padding side, packed vs
      padded layout, and chunked-prefill on/off where the operator claims
      support.
- [ ] Backward consistency is covered for operators that participate in training
      gradients.

# Workstream 3 - WS2 Blackwell Distributed and Parallelism Invariance

Goal: only after single-card parity passes, test Blackwell multi-GPU behavior.
Communication is treated as part of the distributed operator output, not as an
external concern. This workstream should extend the existing WS2 runners rather
than create Blackwell-only distributed tests.

Design requirements:

- [ ] Reuse the existing `ShardingSpec` / `Placement` for TP / SP / CP paths:
      `Shard(dim)`, `Replicate()`, and `Partial()`. Add only backward-compatible
      Blackwell metadata fields if the current spec cannot record the topology.
- [ ] Reuse the existing `ReductionSpec` for every distributed operator:
      accumulator dtype, reduction order, downcast point, reduction engine, and
      special merge semantics. Add fields only if Blackwell/NCCL behavior cannot
      be represented by the shared contract.
- [ ] Extend the existing WS2 cross-config / TP / SP / CP runners with
      Blackwell architecture, topology, NCCL algorithm, and native-cubin/PTX
      axes.
- [ ] For TP matmul and vocab-parallel `linear_logp`, lock reduction order and
      accumulation dtype so TP=1 and TP>1 compare cleanly.
- [ ] For SP-aware logprob / loss, ensure sequence shards are gathered or
      reduced in global token order.
- [ ] For SP RMSNorm boundary cases, bind boundary communication to the same
      `ReductionSpec`; do not let boundary collectives become invisible drift
      sources.
- [ ] For CP attention, merge `(out, lse)` by global block index, not by ring
      arrival order or backend-dependent block arrival.
- [ ] Add distributed reports with per-rank `dlogp`, p95 / p99, max, backend,
      world size, topology, collective library, NCCL algorithm/protocol, and
      Blackwell architecture key.
- [ ] Provide an `engine="in_op"` deterministic reference path: gather data,
      reduce inside the operator in fp32 with a fixed order, and downcast only at
      final write.
- [ ] Treat NCCL / NVLS / NVLink-Sharp / backend-default collectives as fast
      candidates until compared against the deterministic reference.
- [ ] Add A0-A5 audit profiles:
      A0 fully aligned reference;
      A1 arithmetic-only;
      A2 reduction/topology-only;
      A3 representation-only;
      A4 pairwise mismatch;
      A5 full production mismatch.

Distributed launch targets:

- [ ] 1 GPU fallback/reference.
- [ ] 2 GPUs: first parity target, matching the current WS2 Qwen3-8B TP=2 / CP=2
      plan where hardware allows.
- [ ] 4 GPUs: reduction tree and topology sweep.
- [ ] 8 GPUs: same-node B200 / GB200 target if available.
- [ ] Multi-node: deferred until single-node distributed reports are stable.

Distributed commands to preserve or adapt:

```bash
torchrun --standalone --nproc_per_node=2 tests/linear_logp_tp.py --dtype bf16 --op-source registry
torchrun --standalone --nproc_per_node=4 tests/linear_logp_tp.py --dtype bf16 --op-source registry --uneven-shards
```

Distributed exit gate:

- [ ] TP=1 vs TP=2/4 selected logprob and `linear_logp` pass within the shared
      tolerance table.
- [ ] Same logical sequence stays stable across batch=1/N, padding layout,
      chunked prefill on/off, and prefix / KV-cache mode.
- [ ] The report distinguishes operator arithmetic drift from collective-order
      drift.
- [ ] Any NCCL / NVLS / vendor-library drift has a deterministic fallback path.

# Workstream 4 - vime Blackwell Smoke

Goal: prove the Blackwell backend can serve vime as the operator-level
consistency layer without changing vime scheduling semantics.

- [ ] Add or reuse a vime flag/config path that requests RL-Kernel Blackwell
      backend candidates and records the selected operator backend per op.
- [ ] Add a minimal vime fixture with fixed checkpoint, prompt set, tokenizer,
      masks, position IDs, and seed.
- [ ] First target: Qwen3-8B dense, BF16, with TP=2 and CP=2 when hardware is
      available, mirroring the current WS2 Hopper target.
- [ ] Run teacher-forcing logprob on the same generated token sequence and dump
      old logprob, recomputed logprob, and `dlogp`.
- [ ] Validate fallback-only path first.
- [ ] Enable optimized Blackwell kernels one at a time and report the delta in
      `dlogp`, memory, and latency.
- [ ] Keep sampling and logprob explicitly decoupled: temperature / top-p / top-k
      affect token generation, but the training-side logprob comparison scores
      the same token IDs.
- [ ] Pure inference remains unaffected when the RL-Kernel backend is off.

vime exit gate:

- [ ] One documented command runs the Blackwell fallback path and emits a drift
      report.
- [ ] At least one optimized Blackwell operator can be enabled without worsening
      the `dlogp` report beyond tolerance.
- [ ] Any fallback or unsupported operator is visible in the report.

# Workstream 5 - Profiling, CI, Packaging, and Release Rules

Goal: make the Blackwell path reproducible without turning a single kernel
speedup into an unsupported full-step claim.

- [ ] Add Blackwell rows to benchmark reports with device model, runtime, dtype,
      shapes, command line, selected backend, fallback status, peak memory, and
      latency.
- [ ] Add manual GPU-CI instructions until hosted Blackwell CI exists.
- [ ] Extend dynamic architecture CI coverage so extension loading errors are
      caught for `sm90`, `sm100`, and `sm120` where toolchains support them.
- [ ] Document minimum CUDA toolkit and driver requirements for native
      Blackwell cubins and PTX fallback.
- [ ] Add warmup/stable-window profiling for CUDA library initialization,
      cuBLAS/cuBLASLt, NCCL communicator initialization, kernel time,
      collective time, host wait, and vime actor windows.
- [ ] Separate operator speedup, actor-window speedup, and full-step speedup in
      reports.
- [ ] Keep package installation safe on non-Blackwell hosts.

# Proposed Child Issues

- [ ] `[Blackwell][platform-extension] Add architecture probe, runtime facts report, and CUDA backend keys`
- [ ] `[Blackwell][build-extension] Add sm100/sm120/PTX build and compatibility checks to the existing CUDA build path`
- [ ] `[Blackwell][dispatch-extension] Prevent silent SM90-only kernel selection on Blackwell`
- [ ] `[Blackwell][WS1-extension] Add Blackwell backend axis to operator gtest and dlogp reports`
- [ ] `[Blackwell][WS1-logp-extension] Validate selected logprob using the existing WS1 spec and report format`
- [ ] `[Blackwell][WS1-linear_logp-extension] Validate large-vocab linear_logp and LM-head path using existing linear_logp TP tests`
- [ ] `[Blackwell][WS1-attention-extension] Validate standard, varlen, LSE-export, and KV-cache attention through existing attention specs`
- [ ] `[Blackwell][WS1-matmul-extension] Add Blackwell deterministic GEMM / split-K policy to existing matmul issue`
- [ ] `[Blackwell][WS1-rms_norm-extension] Validate fused RMSNorm or keep explicit fallback in existing RMSNorm spec`
- [ ] `[Blackwell][WS1-elementwise-extension] Add Blackwell axis to RoPE, SiLU, SwiGLU, and embedding audits`
- [ ] `[Blackwell][WS1-loss-extension] Add Blackwell axis to ratio-KL and GRPO loss forward/backward checks`
- [ ] `[Blackwell][WS2-extension] Add Blackwell topology/NCCL/NVLS axes to existing TP/SP/CP runners`
- [ ] `[Blackwell][vime-extension] Run the existing vime fallback smoke with Blackwell backend reporting`
- [ ] `[Blackwell][docs-extension] Add Blackwell build, fallback, validation, and benchmark notes to existing CUDA docs`

# Risks

- CUDA compatibility illusion: Blackwell is CUDA, but SM90-specialized kernels,
  TMA/WGMMA assumptions, launch bounds, or inline PTX may not be valid for the
  exact Blackwell target.
- PTX JIT success is not the same as a strict optimized Blackwell path.
- cuBLASLt / CUTLASS / NCCL may choose shape-dependent algorithms that change
  reduction order.
- Multi-GPU collectives may be deterministic for one topology or world size but
  not another.
- BF16 accumulation details may produce drift that only appears after many
  Transformer layers.
- FP8 / MXFP8 / NVFP4 support is tempting on Blackwell but can hide scale,
  rounding, and quantization-policy mismatch; keep it out of the first strict
  milestone.
- vime smoke tests can show false drift if tokenization, padding, temperature,
  top-p, top-k, position IDs, masks, checkpoint state, or weight version are not
  aligned first.
- Hardware access may be the bottleneck; no optimized backend should be marked
  active without a reproducible command log from real Blackwell hardware.

# Open Questions

- What exact Blackwell validation system should be first: B200, GB200, or RTX
  Blackwell?
- Which compute capabilities should be first-class in the release matrix:
  `sm100`, `sm120`, or both?
- Which CUDA toolkit version should be the minimum for native Blackwell cubins?
- Is PTX JIT fallback acceptable for audit-only runs, or only for compatibility
  smoke tests?
- Which CUTLASS/cuBLASLt versions are stable enough for deterministic candidate
  kernels?
- Which NCCL version and topology expose NVLS / NVLink-Sharp behavior on the
  validation node?
- Should the first vime smoke target be Qwen3-8B dense, Qwen3-4B dense, or a
  smaller local fixture?
- Who can run Blackwell validation and how often can maintainers access the
  hardware?

# Final Exit Criteria

The Blackwell / B-card plan can be considered successful when:

1. Blackwell is detected and reported as a CUDA architecture-specific backend
   instead of silently inheriting Hopper-only semantics.
2. Every registered RL-Kernel operator has an explicit fallback on Blackwell.
3. The selected WS1 operator set passes single-card forward and backward
   accuracy: selected `logp`, `linear_logp`, attention, matmul, RMSNorm, RoPE,
   SwiGLU, embedding, LM head, ratio / KL, and GRPO loss.
4. The assembled single-card forward chain reports `dlogp` within the shared
   #108 / #207 tolerance policy on BF16.
5. TP / SP / CP distributed tests either pass against the deterministic
   reference or clearly report the collective-order drift and use fallback.
6. vime can run a documented Blackwell smoke test with a visible backend /
   fallback report.
7. At least one optimized Blackwell operator shows a concrete memory or latency
   win over fallback without changing logprob / loss semantics.
8. Unsupported kernels, dtypes, shapes, architectures, and libraries are
   documented instead of half-enabled.
9. No Blackwell-only duplicate numerical contract, operator harness,
   distributed runner, or report schema remains unless an explicit follow-up
   issue justifies the fork.
