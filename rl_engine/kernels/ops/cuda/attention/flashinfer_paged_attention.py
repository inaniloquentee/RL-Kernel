# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""FlashInfer paged-attention candidate for WS2 PR7.

This module is intentionally opt-in.  It adapts RL-Kernel's
``[B, H, S, D]`` attention tensors and PR6-style paged-KV metadata to
FlashInfer's paged attention wrappers, while recording the three PR7 contract
choices that affect rollout/training alignment:

* Qwen3-exact RoPE fused into attention through ``ROPE_LLAMA``;
* split-KV policy, with auto split rejected when batch invariance is required;
* LSE export and provenance for downstream drift reports.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPCommunication,
    AttentionCPMergedState,
    AttentionCPPartialState,
    AttentionCPCommunicationPlan,
    AttentionParallelSpec,
    sort_attention_cp_partial_states,
)
from rl_engine.kernels.ops.pytorch.attention.cp_attention import (
    AttentionPartialState,
    merge_attention_partial_states,
)
from rl_engine.kernels.attention_contract import (
    AttentionContractError,
    SplitKVExecutionPlan,
    SplitKVMode,
    SplitKVSpec,
    SplitKVRuntimeCoordinate,
    SplitKVRuntimePlanEntry,
    SplitKVRuntimePlanSet,
    validate_split_kv_alignment,
)

RoPEState = Literal["pre_rope", "post_rope"]
FlashInferAttentionMode = Literal["prefill", "decode"]
_FLASHINFER_MODULE = "flashinfer"


class FlashInferUnavailable(RuntimeError):
    """Raised when FlashInfer cannot be imported or lacks required symbols."""


@dataclass(frozen=True)
class FlashInferRoPEFusionConfig:
    """Qwen3 RoPE settings used when FlashInfer performs RoPE inside attention."""

    pos_encoding_mode: str = "ROPE_LLAMA"
    rope_theta: float = 1_000_000.0
    rope_scale: float = 1.0
    rotary_dim: int | None = None
    q_rope_state: RoPEState = "pre_rope"
    k_cache_rope_state: RoPEState = "pre_rope"

    def validate(self, head_dim: int) -> None:
        if self.pos_encoding_mode != "ROPE_LLAMA":
            raise ValueError("PR7 RoPE fusion requires FlashInfer pos_encoding_mode='ROPE_LLAMA'")
        if float(self.rope_theta) != 1_000_000.0:
            raise ValueError("Qwen3-8B RoPE fusion requires rope_theta=1_000_000.0")
        if float(self.rope_scale) != 1.0:
            raise ValueError("Qwen3-8B RoPE fusion requires rope_scale=1.0")
        rotary_dim = head_dim if self.rotary_dim is None else int(self.rotary_dim)
        if rotary_dim != head_dim:
            raise ValueError("FlashInfer PR7 candidate supports full-head Qwen3 RoPE only")
        if self.q_rope_state != "pre_rope" or self.k_cache_rope_state != "pre_rope":
            raise ValueError(
                "FlashInfer ROPE_LLAMA attention fusion expects pre-RoPE Q and pre-RoPE K cache; "
                "post-RoPE tensors would be rotated twice"
            )

    def provenance(self, head_dim: int) -> dict[str, Any]:
        rotary_dim = head_dim if self.rotary_dim is None else int(self.rotary_dim)
        return {
            "rope_fusion": True,
            "rope_fusion_boundary": "flashinfer_attention_kernel",
            "pos_encoding_mode": self.pos_encoding_mode,
            "rope_backend": "flashinfer",
            "rope_theta": float(self.rope_theta),
            "rope_scale": float(self.rope_scale),
            "rotary_dim": rotary_dim,
            "rope_layout": "qwen3_rotate_half_non_interleaved",
            "q_rope_state": self.q_rope_state,
            "k_cache_rope_state": self.k_cache_rope_state,
        }


FlashInferSplitKVPolicy = SplitKVSpec


@dataclass(frozen=True)
class FlashInferPagedAttentionConfig:
    """Runtime knobs for the opt-in FlashInfer paged attention candidate."""

    mode: FlashInferAttentionMode = "prefill"
    causal: bool = True
    kv_layout: str = "NHD"
    softmax_scale: float | None = None
    return_lse: bool = True
    require_batch_invariant: bool = True
    workspace_size_bytes: int = 128 * 1024 * 1024
    rope: FlashInferRoPEFusionConfig = field(default_factory=FlashInferRoPEFusionConfig)
    split_kv: SplitKVSpec = field(default_factory=SplitKVSpec.disabled)
    cp_comm_plan: AttentionCPCommunicationPlan = field(
        default_factory=lambda: AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        )
    )
    require_cp_comm: bool = False
    require_verified_arithmetic: bool = True
    cp_communication: AttentionCPCommunication | None = None

    def validate(self, *, head_dim: int, query_len: int) -> None:
        if self.mode not in {"prefill", "decode"}:
            raise ValueError("mode must be 'prefill' or 'decode'")
        if self.mode == "decode" and query_len != 1:
            raise ValueError("BatchDecodeWithPagedKVCacheWrapper requires Sq == 1")
        if self.kv_layout != "NHD":
            raise ValueError("PR7 FlashInfer adapter currently supports kv_layout='NHD' only")
        if not self.return_lse:
            raise ValueError("PR7 requires attention-domain LSE export")
        if self.workspace_size_bytes <= 0:
            raise ValueError("workspace_size_bytes must be positive")
        self.rope.validate(head_dim)
        if not isinstance(self.split_kv, SplitKVSpec):
            raise ValueError("split_kv must be a SplitKVSpec")
        if self.require_batch_invariant and self.split_kv.mode is SplitKVMode.AUTO:
            raise ValueError(
                "FlashInfer auto split-KV is not a batch-invariant candidate; "
                "use disabled split-KV or a fixed split size"
            )
        self.cp_comm_plan.validate()
        if self.require_cp_comm:
            if self.cp_comm_plan.backend != "p2p_nccl_reference":
                raise ValueError(
                    "executable CP communication currently requires p2p_nccl_reference"
                )
            if self.cp_comm_plan.status != "implemented":
                raise ValueError("executable CP communication requires status='implemented'")
            if self.cp_communication is None:
                raise ValueError("require_cp_comm=True requires a CP communication implementation")
            local_blocks = tuple(
                block
                for block in self.cp_comm_plan.expected_blocks
                if block.owner_cp_rank == self.cp_comm_plan.parallel.cp_rank
            )
            if len(local_blocks) != 1:
                raise ValueError(
                    "FlashInfer outer CP communication requires exactly one manifest block "
                    "per CP owner; backend-local Split-KV remains inside that state"
                )
        elif self.cp_comm_plan.status != "interface_only":
            raise ValueError(
                "implemented CP communication plans require require_cp_comm=True"
            )
        if not isinstance(self.require_verified_arithmetic, bool):
            raise ValueError("require_verified_arithmetic must be a bool")


@dataclass(frozen=True)
class FlashInferPagedKVPlan:
    """FlashInfer paged-KV tensors derived from PR6-style metadata."""

    qo_indptr: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    kv_seq_lens: torch.Tensor
    seq_lens_q: torch.Tensor
    page_size: int
    physical_page_count_per_batch: int
    logical_block_counts: tuple[int, ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "page_size": self.page_size,
            "physical_page_count_per_batch": self.physical_page_count_per_batch,
            "logical_block_counts": list(self.logical_block_counts),
            "qo_indptr": self.qo_indptr.detach().cpu().tolist(),
            "paged_kv_indptr": self.paged_kv_indptr.detach().cpu().tolist(),
            "paged_kv_indices": self.paged_kv_indices.detach().cpu().tolist(),
            "paged_kv_last_page_len": self.paged_kv_last_page_len.detach().cpu().tolist(),
            "kv_seq_lens": self.kv_seq_lens.detach().cpu().tolist(),
            "seq_lens_q": self.seq_lens_q.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class FlashInferAttentionResult:
    """Output of the FlashInfer PR7 candidate."""

    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]


def build_flashinfer_paged_kv_plan(
    metadata: Any,
    *,
    batch_size: int,
    query_len: int,
    cache_capacity: int,
    device: torch.device,
) -> FlashInferPagedKVPlan:
    """Convert PR6-style paged metadata to FlashInfer page table tensors."""

    page_size = _positive_int(int(metadata.page_size), "page_size")
    if cache_capacity % page_size != 0:
        raise ValueError("physical KV cache capacity must be divisible by page_size")
    physical_page_count = cache_capacity // page_size
    if metadata.kv_seq_lens.shape != (batch_size,):
        raise ValueError("kv_seq_lens must have shape [B]")
    if metadata.block_table.ndim != 2 or metadata.block_table.size(0) != batch_size:
        raise ValueError("block_table must have shape [B, max_blocks]")

    qo_indptr = [0]
    paged_kv_indptr = [0]
    paged_kv_indices: list[int] = []
    paged_kv_last_page_len: list[int] = []
    kv_seq_lens: list[int] = []
    seq_lens_q: list[int] = []
    logical_block_counts: list[int] = []
    for batch_index in range(batch_size):
        seq_len = _positive_int(int(metadata.kv_seq_lens[batch_index].item()), "kv_seq_len")
        if seq_len > cache_capacity:
            raise ValueError("kv_seq_len must not exceed cache capacity")
        block_count = (seq_len + page_size - 1) // page_size
        if block_count > metadata.block_table.size(1):
            raise ValueError("block_table does not contain enough logical KV blocks")
        kv_seq_lens.append(seq_len)
        seq_lens_q.append(query_len)
        logical_block_counts.append(block_count)
        qo_indptr.append(qo_indptr[-1] + query_len)
        paged_kv_indptr.append(paged_kv_indptr[-1] + block_count)
        last_len = ((seq_len - 1) % page_size) + 1
        paged_kv_last_page_len.append(last_len)
        for logical_block in range(block_count):
            local_page = int(metadata.block_table[batch_index, logical_block].item())
            if local_page < 0 or local_page >= physical_page_count:
                raise ValueError("block_table contains an out-of-range physical page")
            paged_kv_indices.append(batch_index * physical_page_count + local_page)
        _validate_metadata_logical_positions(
            metadata,
            batch_index=batch_index,
            seq_len=seq_len,
            page_size=page_size,
            block_count=block_count,
            device=device,
        )

    return FlashInferPagedKVPlan(
        qo_indptr=torch.tensor(qo_indptr, device=device, dtype=torch.int32),
        paged_kv_indptr=torch.tensor(paged_kv_indptr, device=device, dtype=torch.int32),
        paged_kv_indices=torch.tensor(paged_kv_indices, device=device, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(
            paged_kv_last_page_len,
            device=device,
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.tensor(kv_seq_lens, device=device, dtype=torch.int32),
        seq_lens_q=torch.tensor(seq_lens_q, device=device, dtype=torch.int32),
        page_size=page_size,
        physical_page_count_per_batch=physical_page_count,
        logical_block_counts=tuple(logical_block_counts),
    )


def materialize_flashinfer_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``[B, Hkv, P*page, D]`` caches to FlashInfer NHD pages."""

    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have matching shape")
    if k_cache.ndim != 4:
        raise ValueError("k_cache and v_cache must have shape [B, Hkv, cache_capacity, D]")
    batch, heads, cache_capacity, head_dim = k_cache.shape
    if cache_capacity % page_size != 0:
        raise ValueError("cache capacity must be divisible by page_size")
    page_count = cache_capacity // page_size
    k_pages = (
        k_cache.contiguous()
        .reshape(batch, heads, page_count, page_size, head_dim)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch * page_count, page_size, heads, head_dim)
        .contiguous()
    )
    v_pages = (
        v_cache.contiguous()
        .reshape(batch, heads, page_count, page_size, head_dim)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch * page_count, page_size, heads, head_dim)
        .contiguous()
    )
    return k_pages, v_pages


class FlashInferQwen3PagedAttentionOp:
    """Opt-in FlashInfer paged attention backend candidate for #235 PR7."""

    op_class = "attention"

    def __init__(self, *, flashinfer_module: Any | None = None) -> None:
        self._flashinfer_module = flashinfer_module

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Any,
        *,
        config: FlashInferPagedAttentionConfig | None = None,
    ) -> FlashInferAttentionResult:
        return self.forward(q, k_cache, v_cache, metadata, config=config)

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Any,
        *,
        config: FlashInferPagedAttentionConfig | None = None,
    ) -> FlashInferAttentionResult:
        """Run FlashInfer paged attention and return RL-Kernel shaped tensors.

        Args:
            q: pre-RoPE query tensor, ``[B, Hq, Sq, D]``.
            k_cache: pre-RoPE paged key cache, ``[B, Hkv, cache_capacity, D]``.
            v_cache: paged value cache, ``[B, Hkv, cache_capacity, D]``.
            metadata: PR6 ``DecodeKVCacheMetadata``-compatible object.
            config: PR7 FlashInfer backend knobs.
        """

        _validate_qkv_cache(q, k_cache, v_cache)
        cfg = FlashInferPagedAttentionConfig() if config is None else config
        batch_size, q_heads, query_len, head_dim = q.shape
        kv_heads = k_cache.size(1)
        cfg.validate(head_dim=head_dim, query_len=query_len)
        if self._flashinfer_module is None and q.device.type != "cuda":
            raise FlashInferUnavailable("FlashInfer PR7 candidate requires CUDA tensors")

        plan = build_flashinfer_paged_kv_plan(
            metadata,
            batch_size=batch_size,
            query_len=query_len,
            cache_capacity=k_cache.size(2),
            device=q.device,
        )
        q_flat = q.transpose(1, 2).reshape(batch_size * query_len, q_heads, head_dim).contiguous()
        k_pages, v_pages = materialize_flashinfer_paged_kv_cache(
            k_cache,
            v_cache,
            page_size=plan.page_size,
        )
        wrapper = self._make_wrapper(cfg, q)
        applied_plan_kwargs = self._plan_wrapper(
            wrapper,
            cfg,
            plan,
            q_dtype=q.dtype,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_len=query_len,
        )
        actual_split_plans = self._actual_split_kv_plans(wrapper, cfg, plan)
        actual_split_plan_set = self._actual_split_kv_plan_set(
            wrapper,
            cfg,
            plan,
        )
        arithmetic = self._actual_arithmetic_semantics(wrapper, cfg)
        out_flat, lse_flat = self._run_wrapper(wrapper, q_flat, (k_pages, v_pages), cfg)
        self._validate_runtime_outputs(
            out_flat,
            lse_flat,
            q,
            require_fp32_output=cfg.require_cp_comm,
        )
        out = _restore_out(out_flat, batch_size=batch_size, query_len=query_len)
        lse = _restore_lse(
            lse_flat,
            batch_size=batch_size,
            query_len=query_len,
            q_heads=q_heads,
        )
        if cfg.require_cp_comm:
            out, lse = self._communicate_cp_partial(out, lse, cfg)
        provenance = {
            "attention_backend": "flashinfer",
            "requested_backend": "flashinfer_qwen3_rope_paged_attention",
            "actual_backend": f"flashinfer_batch_{cfg.mode}_paged_kv",
            "attention_mode": cfg.mode,
            "materialization": "flashinfer_rope_llama_paged_kv",
            "kv_layout": cfg.kv_layout,
            "causal": cfg.causal,
            "softmax_scale": cfg.softmax_scale,
            "lse_domain": "attention",
            "lse_exported": True,
            **arithmetic,
            "fallback": False,
            "fallback_reason": None,
            "paged_kv_policy": "flashinfer_page_table",
        }
        provenance.update(cfg.rope.provenance(head_dim))
        provenance.update(
            _split_kv_provenance(
                cfg.split_kv,
                actual_split_plans,
                applied_plan_kwargs=applied_plan_kwargs,
                require_batch_invariant=cfg.require_batch_invariant,
            )
        )
        provenance["actual_split_kv_plan_set"] = (
            None if actual_split_plan_set is None else actual_split_plan_set.to_dict()
        )
        provenance.update(cfg.cp_comm_plan.provenance())
        provenance["cp_comm_required"] = cfg.require_cp_comm
        provenance.update(plan.provenance())
        return FlashInferAttentionResult(
            out=out.to(dtype=q.dtype),
            lse=lse,
            provenance=provenance,
        )

    @staticmethod
    def _communicate_cp_partial(
        out: torch.Tensor,
        lse: torch.Tensor,
        cfg: FlashInferPagedAttentionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        communication = cfg.cp_communication
        assert communication is not None
        local_blocks = tuple(
            block
            for block in cfg.cp_comm_plan.expected_blocks
            if block.owner_cp_rank == cfg.cp_comm_plan.parallel.cp_rank
        )
        local = AttentionCPPartialState(out=out, lse=lse, block=local_blocks[0])
        gathered = communication.all_gather_partial_states((local,), cfg.cp_comm_plan)
        ordered = sort_attention_cp_partial_states(gathered, plan=cfg.cp_comm_plan)
        merged = merge_attention_partial_states(
            [
                AttentionPartialState(
                    out=state.out,
                    lse=state.lse,
                    block_start=state.block.kv_block_start,
                    block_end=state.block.kv_block_end,
                )
                for state in ordered
            ]
        )
        local_merged = communication.reduce_scatter_merged_state(
            AttentionCPMergedState(out=merged.out, lse=merged.lse),
            cfg.cp_comm_plan,
        )
        return local_merged.out, local_merged.lse

    def _load_flashinfer(self) -> Any:
        if self._flashinfer_module is not None:
            return self._flashinfer_module
        try:
            self._flashinfer_module = importlib.import_module(_FLASHINFER_MODULE)
        except (ImportError, OSError, RuntimeError) as exc:
            raise FlashInferUnavailable(str(exc)) from exc
        return self._flashinfer_module

    def _make_wrapper(self, cfg: FlashInferPagedAttentionConfig, q: torch.Tensor) -> Any:
        module = self._load_flashinfer()
        namespace_name = "decode" if cfg.mode == "decode" else "prefill"
        class_name = (
            "BatchDecodeWithPagedKVCacheWrapper"
            if cfg.mode == "decode"
            else "BatchPrefillWithPagedKVCacheWrapper"
        )
        namespace = getattr(module, namespace_name, None)
        wrapper_cls = getattr(namespace, class_name, None) if namespace is not None else None
        if wrapper_cls is None:
            raise FlashInferUnavailable(f"flashinfer.{namespace_name}.{class_name} is unavailable")

        workspace = torch.zeros(cfg.workspace_size_bytes, dtype=torch.uint8, device=q.device)
        try:
            return wrapper_cls(workspace, kv_layout=cfg.kv_layout)
        except TypeError:
            try:
                return wrapper_cls(float_workspace_buffer=workspace, kv_layout=cfg.kv_layout)
            except TypeError as exc:
                raise FlashInferUnavailable(
                    f"could not instantiate flashinfer.{namespace_name}.{class_name}"
                ) from exc

    @staticmethod
    def _plan_wrapper(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
        plan: FlashInferPagedKVPlan,
        *,
        q_dtype: torch.dtype,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        query_len: int,
    ) -> dict[str, Any]:
        plan_kwargs = {
            "qo_indptr": plan.qo_indptr,
            "paged_kv_indptr": plan.paged_kv_indptr,
            "paged_kv_indices": plan.paged_kv_indices,
            "paged_kv_last_page_len": plan.paged_kv_last_page_len,
            "indptr": plan.paged_kv_indptr,
            "indices": plan.paged_kv_indices,
            "last_page_len": plan.paged_kv_last_page_len,
            "num_qo_heads": q_heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "head_dim_qk": head_dim,
            "page_size": plan.page_size,
            "causal": cfg.causal,
            "pos_encoding_mode": cfg.rope.pos_encoding_mode,
            "rope_scale": float(cfg.rope.rope_scale),
            "rope_theta": float(cfg.rope.rope_theta),
            "q_data_type": q_dtype,
            "kv_data_type": q_dtype,
            "o_data_type": torch.float32 if cfg.require_cp_comm else q_dtype,
            "data_type": q_dtype,
            "seq_lens": plan.kv_seq_lens,
            "seq_lens_q": plan.seq_lens_q,
            "q_len_per_req": query_len,
        }
        scale = cfg.softmax_scale
        if scale is not None:
            plan_kwargs["softmax_scale"] = float(scale)
            plan_kwargs["sm_scale"] = float(scale)
        plan_kwargs.update(_flashinfer_split_kv_plan_kwargs(cfg.split_kv))
        applied = _call_with_supported_kwargs(wrapper.plan, plan_kwargs, return_applied=True)
        assert isinstance(applied, dict)
        required_knob = (
            "fixed_split_size"
            if cfg.split_kv.mode is SplitKVMode.FIXED
            else "disable_split_kv"
        )
        if cfg.split_kv.mode is not SplitKVMode.AUTO and required_knob not in applied:
            raise FlashInferUnavailable(
                f"FlashInfer plan() did not accept required Split-KV knob {required_knob!r}"
            )
        return applied

    @staticmethod
    def _actual_split_kv_plans(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
        plan: FlashInferPagedKVPlan,
    ) -> tuple[SplitKVExecutionPlan, ...]:
        getter = getattr(wrapper, "get_actual_split_kv_plan", None)
        if not callable(getter):
            if cfg.split_kv.mode is SplitKVMode.DISABLED:
                return tuple(
                    cfg.split_kv.resolve(int(seq_len), backend="flashinfer_disabled_verified")
                    for seq_len in plan.kv_seq_lens.tolist()
                )
            if cfg.require_batch_invariant:
                raise FlashInferUnavailable(
                    "strict fixed Split-KV consistency requires runtime actual-plan provenance; "
                    "FlashInfer wrapper has no get_actual_split_kv_plan() callback. A requested "
                    "max-splits/count knob is not proof of token boundaries"
                )
            return tuple(
                cfg.split_kv.resolve(int(seq_len), backend="flashinfer_requested_only")
                for seq_len in plan.kv_seq_lens.tolist()
            )
        raw_plans = getter()
        if not isinstance(raw_plans, (list, tuple)) or len(raw_plans) != len(plan.kv_seq_lens):
            raise FlashInferUnavailable(
                "get_actual_split_kv_plan() must return one plan per batch request"
            )
        result: list[SplitKVExecutionPlan] = []
        for batch_index, (raw, seq_len) in enumerate(
            zip(raw_plans, plan.kv_seq_lens.tolist(), strict=True)
        ):
            if not isinstance(raw, dict):
                raise FlashInferUnavailable("actual Split-KV runtime plan entries must be dicts")
            try:
                required_keys = {
                    "mode",
                    "split_size",
                    "boundaries",
                    "fallback",
                    "fallback_reason",
                }
                missing_keys = sorted(required_keys.difference(raw))
                if missing_keys:
                    raise FlashInferUnavailable(
                        "actual Split-KV runtime plan is missing required fields: "
                        + ", ".join(missing_keys)
                    )
                execution = SplitKVExecutionPlan(
                    requested_mode=cfg.split_kv.mode,
                    requested_split_size=cfg.split_kv.fixed_split_size,
                    actual_mode=raw.get("mode"),
                    actual_split_size=raw.get("split_size"),
                    boundaries=tuple(
                        tuple(boundary) for boundary in raw.get("boundaries", ())
                    ),
                    backend="flashinfer",
                    source="runtime_callback",
                    fallback=raw["fallback"],
                    fallback_reason=raw["fallback_reason"],
                )
            except AttentionContractError as exc:
                raise FlashInferUnavailable(
                    f"invalid actual Split-KV plan for batch {batch_index}: {exc}"
                ) from exc
            expected = cfg.split_kv.resolve(int(seq_len), backend="flashinfer_contract")
            if cfg.require_batch_invariant:
                try:
                    validate_split_kv_alignment(expected, execution)
                except AttentionContractError as exc:
                    raise FlashInferUnavailable(
                        f"FlashInfer actual Split-KV plan for batch {batch_index} "
                        f"does not match the requested strict logical plan: {exc}"
                    ) from exc
            result.append(execution)
        return tuple(result)

    @staticmethod
    def _actual_split_kv_plan_set(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
        plan: FlashInferPagedKVPlan,
    ) -> SplitKVRuntimePlanSet | None:
        getter = getattr(wrapper, "get_actual_split_kv_plan_set", None)
        if not callable(getter):
            if cfg.require_batch_invariant:
                raise FlashInferUnavailable(
                    "strict Split-KV consistency requires a complete "
                    "batch/TP/CP/owner "
                    "runtime plan set; FlashInfer wrapper has no "
                    "get_actual_split_kv_plan_set() callback"
                )
            return None
        raw = getter()
        if not isinstance(raw, dict):
            raise FlashInferUnavailable(
                "get_actual_split_kv_plan_set() must return a dict"
            )
        try:
            entries = tuple(
                SplitKVRuntimePlanEntry(
                    coordinate=SplitKVRuntimeCoordinate(
                        batch_index=int(entry["batch_index"]),
                        tp_rank=int(entry["tp_rank"]),
                        cp_rank=int(entry["cp_rank"]),
                        owner_cp_rank=int(entry["owner_cp_rank"]),
                    ),
                    expected_kv_range=tuple(entry["expected_kv_range"]),
                    execution=SplitKVExecutionPlan(
                        requested_mode=cfg.split_kv.mode,
                        requested_split_size=cfg.split_kv.fixed_split_size,
                        actual_mode=entry["mode"],
                        actual_split_size=entry["split_size"],
                        boundaries=tuple(
                            tuple(boundary) for boundary in entry["boundaries"]
                        ),
                        merge_order=entry["merge_order"],
                        acc_dtype=entry["accum_dtype"],
                        downcast_at=entry["downcast_at"],
                        backend="flashinfer",
                        source="runtime_plan_set_callback",
                        fallback=entry["fallback"],
                        fallback_reason=entry["fallback_reason"],
                    ),
                )
                for entry in raw["entries"]
            )
            plan_set = SplitKVRuntimePlanSet(
                batch_size=int(raw["batch_size"]),
                tp_world_size=int(raw["tp_world_size"]),
                cp_world_size=int(raw["cp_world_size"]),
                total_kv_tokens=tuple(int(value) for value in raw["total_kv_tokens"]),
                entries=entries,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FlashInferUnavailable(
                f"invalid FlashInfer runtime Split-KV plan set: {exc}"
            ) from exc
        parallel = cfg.cp_comm_plan.parallel
        expected_topology = (
            len(plan.kv_seq_lens),
            parallel.tp_world_size,
            parallel.cp_world_size,
            tuple(int(value) for value in plan.kv_seq_lens.tolist()),
        )
        actual_topology = (
            plan_set.batch_size,
            plan_set.tp_world_size,
            plan_set.cp_world_size,
            plan_set.total_kv_tokens,
        )
        if actual_topology != expected_topology:
            raise FlashInferUnavailable(
                "FlashInfer runtime Split-KV plan-set topology does not match the request"
            )
        if cfg.require_batch_invariant:
            for entry in plan_set.entries:
                start, end = entry.expected_kv_range
                expected_local = cfg.split_kv.resolve(
                    end - start,
                    backend="flashinfer_contract",
                )
                expected = SplitKVExecutionPlan(
                    requested_mode=expected_local.requested_mode,
                    requested_split_size=expected_local.requested_split_size,
                    actual_mode=expected_local.actual_mode,
                    actual_split_size=expected_local.actual_split_size,
                    boundaries=tuple(
                        (start + local_start, start + local_end)
                        for local_start, local_end in expected_local.boundaries
                    ),
                    backend="flashinfer_contract",
                    source="contract_exact",
                )
                try:
                    validate_split_kv_alignment(expected, entry.execution)
                except AttentionContractError as exc:
                    raise FlashInferUnavailable(
                        "FlashInfer runtime Split-KV plan-set entry does not match "
                        f"the strict owner-local plan at {entry.coordinate}: {exc}"
                    ) from exc
        return plan_set

    @staticmethod
    def _actual_arithmetic_semantics(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
    ) -> dict[str, Any]:
        getter = getattr(wrapper, "get_attention_arithmetic_provenance", None)
        if not callable(getter):
            if cfg.require_verified_arithmetic:
                raise FlashInferUnavailable(
                    "strict attention consistency requires runtime arithmetic provenance; "
                    "FlashInfer wrapper has no get_attention_arithmetic_provenance() callback"
                )
            return {
                "accum_dtype": None,
                "downcast_at": None,
                "lse_dtype": None,
                "arithmetic_plan_source": "unverified_backend_internal",
                "arithmetic_semantics_verified": False,
            }
        raw = getter()
        if not isinstance(raw, dict):
            raise FlashInferUnavailable(
                "get_attention_arithmetic_provenance() must return a dict"
            )
        required = {
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "lse_dtype": "fp32",
        }
        mismatches = [
            key for key, expected in required.items() if raw.get(key) != expected
        ]
        source = raw.get("source")
        if not isinstance(source, str) or not source.strip():
            mismatches.append("source")
        if mismatches:
            raise FlashInferUnavailable(
                "FlashInfer runtime arithmetic semantics do not satisfy the strict "
                "attention contract: " + ", ".join(mismatches)
            )
        return {
            **required,
            "arithmetic_plan_source": source,
            "arithmetic_semantics_verified": True,
        }

    @staticmethod
    def _run_wrapper(
        wrapper: Any,
        q_flat: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        cfg: FlashInferPagedAttentionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(wrapper, "run_return_lse"):
            result = wrapper.run_return_lse(q_flat, paged_kv_cache)
        else:
            result = _call_with_supported_kwargs(
                wrapper.run,
                {"q": q_flat, "paged_kv_cache": paged_kv_cache, "return_lse": cfg.return_lse},
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise FlashInferUnavailable("FlashInfer PR7 candidate must return (out, lse)")
        out_flat, lse_flat = result
        return out_flat, lse_flat

    @staticmethod
    def _validate_runtime_outputs(
        out_flat: torch.Tensor,
        lse_flat: torch.Tensor,
        q: torch.Tensor,
        *,
        require_fp32_output: bool,
    ) -> None:
        if not isinstance(out_flat, torch.Tensor) or not isinstance(lse_flat, torch.Tensor):
            raise FlashInferUnavailable("FlashInfer output and LSE must be tensors")
        if out_flat.device != q.device or lse_flat.device != q.device:
            raise FlashInferUnavailable(
                "FlashInfer output and LSE must remain on the query device"
            )
        expected_out_dtype = torch.float32 if require_fp32_output else q.dtype
        if out_flat.dtype != expected_out_dtype:
            raise FlashInferUnavailable(
                "FlashInfer final output dtype does not match the requested output dtype"
            )
        if lse_flat.dtype != torch.float32:
            raise FlashInferUnavailable("FlashInfer attention-domain LSE must be FP32")


def flashinfer_qwen3_paged_attention_available() -> bool:
    """Return whether the FlashInfer paged attention wrappers are importable."""

    try:
        module = FlashInferQwen3PagedAttentionOp()._load_flashinfer()
        prefill = getattr(
            getattr(module, "prefill", None),
            "BatchPrefillWithPagedKVCacheWrapper",
            None,
        )
        decode = getattr(
            getattr(module, "decode", None),
            "BatchDecodeWithPagedKVCacheWrapper",
            None,
        )
        if not callable(prefill) or not callable(decode):
            return False
    except FlashInferUnavailable:
        return False
    return True


def _validate_metadata_logical_positions(
    metadata: Any,
    *,
    batch_index: int,
    seq_len: int,
    page_size: int,
    block_count: int,
    device: torch.device,
) -> None:
    if not hasattr(metadata, "global_token_positions"):
        return
    global_token_positions = metadata.global_token_positions
    if global_token_positions.ndim != 2 or global_token_positions.size(0) <= batch_index:
        raise ValueError("global_token_positions must have shape [B, cache_capacity]")
    physical_slots: list[int] = []
    for logical_block in range(block_count):
        local_page = int(metadata.block_table[batch_index, logical_block].item())
        token_count = min(page_size, seq_len - logical_block * page_size)
        for page_offset in range(token_count):
            physical_slots.append(local_page * page_size + page_offset)
    slot_index = torch.tensor(physical_slots, device=device, dtype=torch.long)
    actual = global_token_positions[batch_index, slot_index]
    position_offset = int(actual[0].item())
    expected = torch.arange(
        position_offset,
        position_offset + seq_len,
        device=device,
        dtype=global_token_positions.dtype,
    )
    if not torch.equal(actual, expected):
        raise ValueError(
            "block_table/global_token_positions must reconstruct logical positions "
            "as one contiguous global range"
        )
    if hasattr(metadata, "key_position_ids"):
        key_positions = metadata.key_position_ids[batch_index, slot_index]
        if not torch.equal(key_positions, expected.to(dtype=key_positions.dtype)):
            raise ValueError("key_position_ids must match cached global token positions")


def _validate_qkv_cache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("q, k_cache, and v_cache must have shape [B, H, S, D]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have matching shape")
    if q.size(0) != k_cache.size(0) or q.size(3) != k_cache.size(3):
        raise ValueError("q and KV cache must share batch size and head_dim")
    if q.size(1) % k_cache.size(1) != 0:
        raise ValueError("Q head count must be divisible by KV head count")


def _restore_out(out_flat: torch.Tensor, *, batch_size: int, query_len: int) -> torch.Tensor:
    if out_flat.ndim != 3:
        raise FlashInferUnavailable("FlashInfer output must have shape [B*Sq, Hq, D]")
    _, q_heads, head_dim = out_flat.shape
    expected = batch_size * query_len
    if out_flat.size(0) != expected:
        raise FlashInferUnavailable(
            f"FlashInfer output first dim must be B*Sq={expected}, got {out_flat.size(0)}"
        )
    return out_flat.reshape(batch_size, query_len, q_heads, head_dim).transpose(1, 2).contiguous()


def _restore_lse(
    lse_flat: torch.Tensor,
    *,
    batch_size: int,
    query_len: int,
    q_heads: int,
) -> torch.Tensor:
    expected_tokens = batch_size * query_len
    if lse_flat.shape == (expected_tokens, q_heads):
        return lse_flat.reshape(batch_size, query_len, q_heads).transpose(1, 2).contiguous()
    if lse_flat.shape == (q_heads, expected_tokens):
        return lse_flat.transpose(0, 1).reshape(batch_size, query_len, q_heads).transpose(1, 2)
    raise FlashInferUnavailable(
        "FlashInfer LSE must have shape [B*Sq, Hq] or [Hq, B*Sq]; " f"got {tuple(lse_flat.shape)}"
    )


def _call_with_supported_kwargs(
    fn: Any,
    kwargs: dict[str, Any],
    *,
    return_applied: bool = False,
) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        result = fn(**kwargs)
        return dict(kwargs) if return_applied else result
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        result = fn(**kwargs)
        return dict(kwargs) if return_applied else result
    supported = {name: value for name, value in kwargs.items() if name in parameters}
    missing_required = [
        name
        for name, param in parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and name not in supported
    ]
    if missing_required:
        raise FlashInferUnavailable(
            f"{getattr(fn, '__qualname__', fn)} missing supported arguments: "
            f"{', '.join(missing_required)}"
        )
    result = fn(**supported)
    return supported if return_applied else result


def _flashinfer_split_kv_plan_kwargs(spec: SplitKVSpec) -> dict[str, Any]:
    if spec.mode is SplitKVMode.DISABLED:
        return {"disable_split_kv": True}
    if spec.mode is SplitKVMode.FIXED:
        assert spec.fixed_split_size is not None
        return {
            "fixed_split_size": int(spec.fixed_split_size),
            "disable_split_kv": False,
        }
    return {"disable_split_kv": False}


def _split_kv_provenance(
    spec: SplitKVSpec,
    plans: tuple[SplitKVExecutionPlan, ...],
    *,
    applied_plan_kwargs: dict[str, Any],
    require_batch_invariant: bool,
) -> dict[str, Any]:
    policy = spec.mode.value
    if spec.mode is SplitKVMode.FIXED:
        policy = f"fixed:{spec.fixed_split_size}"
    return {
        "split_kv_policy": policy,
        "requested_split_kv_policy": spec.mode.value,
        "requested_split_kv_size": spec.fixed_split_size,
        "actual_split_kv_plans": [plan.to_dict() for plan in plans],
        "backend_native_split_kv_knobs": {
            key: value
            for key, value in applied_plan_kwargs.items()
            if key in {"fixed_split_size", "disable_split_kv"}
        },
        "batch_invariant_required": bool(require_batch_invariant),
        "batch_invariant_claim": (
            "strict_runtime_verified" if require_batch_invariant else "diagnostic_only"
        ),
    }


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


__all__ = [
    "FlashInferAttentionMode",
    "FlashInferAttentionResult",
    "FlashInferPagedAttentionConfig",
    "FlashInferPagedKVPlan",
    "FlashInferQwen3PagedAttentionOp",
    "FlashInferRoPEFusionConfig",
    "FlashInferSplitKVPolicy",
    "FlashInferUnavailable",
    "build_flashinfer_paged_kv_plan",
    "flashinfer_qwen3_paged_attention_available",
    "materialize_flashinfer_paged_kv_cache",
]
