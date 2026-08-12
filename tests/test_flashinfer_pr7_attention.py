# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import types

import pytest
import torch

from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionCPCommunicationUnavailable,
    AttentionCPMergedState,
    AttentionCPPartialState,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
    P2PNCCLAttentionCPCommunication,
    sort_attention_cp_partial_states,
)
from rl_engine.kernels.ops.cuda.attention.flashinfer_paged_attention import (
    FlashInferPagedAttentionConfig,
    FlashInferQwen3PagedAttentionOp,
    FlashInferRoPEFusionConfig,
    FlashInferSplitKVPolicy,
    FlashInferUnavailable,
    build_flashinfer_paged_kv_plan,
    materialize_flashinfer_paged_kv_cache,
)
from rl_engine.kernels.attention_contract import SplitKVSpec
from rl_engine.testing.attention_comparison import DecodeKVCacheMetadata


class _FakeFlashInferWrapper:
    instances: list["_FakeFlashInferWrapper"] = []

    def __init__(self, workspace_buffer, *, kv_layout):
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self.plan_kwargs = None
        self.run_q = None
        self.run_cache = None
        self.instances.append(self)

    def plan(self, **kwargs):
        self.plan_kwargs = kwargs

    def run_return_lse(self, q, paged_kv_cache):
        self.run_q = q
        self.run_cache = paged_kv_cache
        out = torch.zeros(
            q.shape,
            dtype=self.plan_kwargs.get("o_data_type", q.dtype),
            device=q.device,
        )
        lse = torch.zeros(q.size(0), q.size(1), dtype=torch.float32, device=q.device)
        return out, lse

    def get_actual_split_kv_plan(self):
        seq_lens = self.plan_kwargs["seq_lens"].tolist()
        disabled = bool(self.plan_kwargs.get("disable_split_kv", False))
        split_size = self.plan_kwargs.get("fixed_split_size")
        plans = []
        for seq_len in seq_lens:
            if disabled:
                boundaries = [(0, seq_len)]
                mode = "disabled"
                actual_size = None
            else:
                assert split_size is not None
                boundaries = [
                    (start, min(start + split_size, seq_len))
                    for start in range(0, seq_len, split_size)
                ]
                mode = "fixed"
                actual_size = split_size
            plans.append(
                {
                    "mode": mode,
                    "split_size": actual_size,
                    "boundaries": boundaries,
                    "fallback": False,
                    "fallback_reason": None,
                }
            )
        return plans

    def get_attention_arithmetic_provenance(self):
        return {
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "lse_dtype": "fp32",
            "source": "fake_runtime_capability",
        }

    def get_actual_split_kv_plan_set(self):
        seq_lens = self.plan_kwargs["seq_lens"].tolist()
        disabled = bool(self.plan_kwargs.get("disable_split_kv", False))
        split_size = self.plan_kwargs.get("fixed_split_size")
        entries = []
        for batch_index, total in enumerate(seq_lens):
            owner_ranges = ((0, total // 2), (total // 2, total))
            for tp_rank in range(2):
                for cp_rank in range(2):
                    for owner_cp_rank, (owner_start, owner_end) in enumerate(owner_ranges):
                        if disabled:
                            mode = "disabled"
                            actual_size = None
                            boundaries = [(owner_start, owner_end)]
                        else:
                            mode = "fixed"
                            actual_size = split_size
                            boundaries = [
                                (start, min(start + split_size, owner_end))
                                for start in range(owner_start, owner_end, split_size)
                            ]
                        entries.append(
                            {
                                "batch_index": batch_index,
                                "tp_rank": tp_rank,
                                "cp_rank": cp_rank,
                                "owner_cp_rank": owner_cp_rank,
                                "expected_kv_range": [owner_start, owner_end],
                                "mode": mode,
                                "split_size": actual_size,
                                "boundaries": boundaries,
                                "merge_order": "global_block_index",
                                "accum_dtype": "fp32",
                                "downcast_at": "final_write",
                                "fallback": False,
                                "fallback_reason": None,
                            }
                        )
        return {
            "batch_size": len(seq_lens),
            "tp_world_size": 2,
            "cp_world_size": 2,
            "total_kv_tokens": seq_lens,
            "entries": entries,
        }


def _fake_flashinfer():
    _FakeFlashInferWrapper.instances = []
    return types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_FakeFlashInferWrapper,
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_FakeFlashInferWrapper,
        ),
    )


def _metadata(*, batch: int = 2, query_len: int = 1) -> DecodeKVCacheMetadata:
    page_size = 2
    cache_capacity = 6
    positions = torch.arange(cache_capacity, dtype=torch.long).repeat(batch, 1)
    return DecodeKVCacheMetadata(
        cache_position=torch.full((batch, query_len), cache_capacity - 1, dtype=torch.long),
        kv_seq_lens=torch.full((batch,), cache_capacity, dtype=torch.long),
        block_table=torch.tensor([[0, 1, 2]] * batch, dtype=torch.long),
        global_token_positions=positions,
        query_position_ids=torch.full((batch, query_len), cache_capacity - 1, dtype=torch.long),
        key_position_ids=positions.clone(),
        page_size=page_size,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )


def _qkv(*, batch: int = 2, query_len: int = 1):
    gen = torch.Generator().manual_seed(7)
    q = torch.randn(batch, 4, query_len, 8, generator=gen)
    k = torch.randn(batch, 2, 6, 8, generator=gen)
    v = torch.randn(batch, 2, 6, 8, generator=gen)
    return q, k, v


def _partial_state(global_block_index: int) -> AttentionCPPartialState:
    return AttentionCPPartialState(
        out=torch.full((1, 2, 1, 4), float(global_block_index)),
        lse=torch.full((1, 2, 1), float(global_block_index), dtype=torch.float32),
        block=AttentionCPBlockMetadata(
            global_block_index=global_block_index,
            kv_block_start=global_block_index * 2,
            kv_block_end=global_block_index * 2 + 2,
            owner_cp_rank=global_block_index % 2,
            owner_tp_rank=0,
        ),
    )


def _p2p_plan(*, cp_rank: int = 0) -> AttentionCPCommunicationPlan:
    return AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(
            tp_world_size=2,
            tp_rank=0,
            cp_world_size=2,
            cp_rank=cp_rank,
        ),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 2, 0, 0),
            AttentionCPBlockMetadata(1, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 2)),
    )


class _CompletedRequest:
    def wait(self):
        return True


class _FakeP2POp:
    def __init__(self, op, tensor, peer, *, group=None):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group


class _FakeNCCLDistributed:
    def __init__(self, *, rank: int, receive_payloads=()):
        self.rank = rank
        self.receive_payloads = list(receive_payloads)

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def get_backend(group=None):
        return "nccl"

    @staticmethod
    def get_world_size(group=None):
        return 2

    def get_rank(self, group=None):
        return self.rank

    @staticmethod
    def get_global_rank(group, rank):
        return rank

    @staticmethod
    def isend(tensor, dst, group=None):
        raise AssertionError("P2POp should defer isend")

    @staticmethod
    def irecv(tensor, src, group=None):
        raise AssertionError("P2POp should defer irecv")

    P2POp = _FakeP2POp

    def batch_isend_irecv(self, operations):
        for operation in operations:
            if getattr(operation.op, "__name__", None) == "irecv":
                operation.tensor.copy_(self.receive_payloads.pop(0))
        return [_CompletedRequest() for _ in operations]


class _FakeCPCommunication:
    def all_gather_partial_states(self, local_states, plan):
        local = local_states[0]
        remote_block = next(
            block
            for block in plan.expected_blocks
            if block.owner_cp_rank != plan.parallel.cp_rank
        )
        remote = AttentionCPPartialState(
            out=torch.ones_like(local.out),
            lse=torch.ones_like(local.lse),
            block=remote_block,
        )
        return tuple(sorted((local, remote), key=lambda state: state.block.global_block_index))

    def reduce_scatter_merged_state(self, merged_state, plan):
        start, end = plan.query_token_ranges[plan.parallel.cp_rank]
        return AttentionCPMergedState(
            out=merged_state.out[:, :, start:end, :],
            lse=merged_state.lse[:, :, start:end],
        )


def test_flashinfer_pr7_prefill_adapter_passes_qwen3_rope_and_splitk_policy():
    q, k, v = _qkv(query_len=2)
    metadata = _metadata(query_len=2)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    result = op(
        q,
        k,
        v,
        metadata,
        config=FlashInferPagedAttentionConfig(
            mode="prefill",
            workspace_size_bytes=1024,
            split_kv=SplitKVSpec.fixed(4),
        ),
    )

    wrapper = _FakeFlashInferWrapper.instances[-1]
    assert wrapper.kv_layout == "NHD"
    assert wrapper.run_q.shape == (q.size(0) * q.size(2), q.size(1), q.size(3))
    assert result.out.shape == q.shape
    assert result.lse.shape == q.shape[:3]
    assert result.provenance["actual_backend"] == "flashinfer_batch_prefill_paged_kv"
    assert result.provenance["rope_fusion_boundary"] == "flashinfer_attention_kernel"
    assert result.provenance["pos_encoding_mode"] == "ROPE_LLAMA"
    assert result.provenance["rope_theta"] == 1_000_000.0
    assert result.provenance["rope_scale"] == 1.0
    assert result.provenance["split_kv_policy"] == "fixed:4"
    assert result.provenance["batch_invariant_claim"] == "strict_runtime_verified"
    assert result.provenance["requested_split_kv_policy"] == "fixed"
    assert result.provenance["requested_split_kv_size"] == 4
    assert result.provenance["actual_split_kv_plans"][0]["actual_split_boundaries"] == [
        [0, 4],
        [4, 6],
    ]
    assert result.provenance["tp_world_size"] == 2
    assert result.provenance["cp_world_size"] == 2
    assert result.provenance["cp_comm_backend"] == "cuda_ag_rs"
    assert result.provenance["cp_comm_status"] == "interface_only"
    assert result.provenance["cp_comm_pattern"] == "ag_rs"
    assert result.provenance["cp_comm_compute_communication"] == "decoupled"
    assert result.provenance["cp_comm_merge_order"] == "global_block_index"
    assert result.provenance["cp_comm_accum_dtype"] == "fp32"
    assert result.provenance["cp_comm_return_lse"] is True
    assert result.provenance["cp_comm_contract"] == "partial_out_lse_global_block_index"
    assert result.provenance["cp_comm_required"] is False
    assert result.provenance["accum_dtype"] == "fp32"
    assert result.provenance["downcast_at"] == "final_write"
    assert result.provenance["arithmetic_semantics_verified"] is True
    assert result.provenance["actual_split_kv_plan_set"]["coverage"] == (
        "complete_batch_tp_cp_owner_cartesian_product"
    )

    plan = wrapper.plan_kwargs
    assert plan["qo_indptr"].tolist() == [0, 2, 4]
    assert plan["paged_kv_indptr"].tolist() == [0, 3, 6]
    assert plan["paged_kv_indices"].tolist() == [0, 1, 2, 3, 4, 5]
    assert plan["paged_kv_last_page_len"].tolist() == [2, 2]
    assert plan["pos_encoding_mode"] == "ROPE_LLAMA"
    assert plan["rope_theta"] == 1_000_000.0
    assert plan["rope_scale"] == 1.0
    assert plan["q_data_type"] == q.dtype
    assert plan["kv_data_type"] == q.dtype
    assert plan["fixed_split_size"] == 4
    assert plan["disable_split_kv"] is False


def test_flashinfer_pr7_p2p_cp_path_merges_fp32_before_final_downcast():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    plan = _p2p_plan()
    config = FlashInferPagedAttentionConfig(
        mode="decode",
        workspace_size_bytes=1024,
        cp_comm_plan=plan,
        require_cp_comm=True,
        cp_communication=_FakeCPCommunication(),
    )
    result = FlashInferQwen3PagedAttentionOp(
        flashinfer_module=_fake_flashinfer()
    )(q, k, v, metadata, config=config)

    wrapper = _FakeFlashInferWrapper.instances[-1]
    assert wrapper.plan_kwargs["o_data_type"] is torch.float32
    assert result.out.dtype == q.dtype
    assert result.lse.dtype == torch.float32
    assert result.provenance["cp_comm_backend"] == "p2p_nccl_reference"


def test_flashinfer_pr7_decode_adapter_can_disable_splitk_for_strict_candidate():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    result = op(
        q,
        k,
        v,
        metadata,
        config=FlashInferPagedAttentionConfig(
            mode="decode",
            workspace_size_bytes=1024,
            split_kv=SplitKVSpec.disabled(),
        ),
    )

    wrapper = _FakeFlashInferWrapper.instances[-1]
    assert result.provenance["actual_backend"] == "flashinfer_batch_decode_paged_kv"
    assert result.provenance["split_kv_policy"] == "disabled"
    assert result.provenance["batch_invariant_claim"] == "strict_runtime_verified"
    assert wrapper.plan_kwargs["disable_split_kv"] is True


def test_flashinfer_pr7_rejects_auto_splitk_when_batch_invariance_is_required():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="auto split-KV"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                split_kv=SplitKVSpec.auto(),
            ),
        )


def test_flashinfer_pr7_strict_fixed_mode_requires_actual_runtime_split_plan():
    class _NoRuntimePlanWrapper(_FakeFlashInferWrapper):
        get_actual_split_kv_plan = None

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=_NoRuntimePlanWrapper),
        decode=types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_NoRuntimePlanWrapper),
    )




    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="actual-plan provenance"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                split_kv=SplitKVSpec.fixed(2),
            ),
        )


def test_flashinfer_pr7_disabled_plan_is_exact_when_disable_knob_is_accepted():
    class _NoRuntimePlanWrapper(_FakeFlashInferWrapper):
        get_actual_split_kv_plan = None

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=_NoRuntimePlanWrapper),
        decode=types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_NoRuntimePlanWrapper),
    )
    q, k, v = _qkv(query_len=1)
    result = FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
        q,
        k,
        v,
        _metadata(query_len=1),
        config=FlashInferPagedAttentionConfig(
            mode="decode",
            workspace_size_bytes=1024,
            split_kv=SplitKVSpec.disabled(),
        ),
    )

    assert result.provenance["actual_split_kv_plans"][0][
        "actual_split_boundaries"
    ] == [[0, 6]]
    assert result.provenance["actual_split_kv_plans"][0]["split_kv_backend"] == (
        "flashinfer_disabled_verified"
    )


def test_flashinfer_pr7_rejects_actual_split_plan_mismatch():
    class _MismatchedRuntimePlanWrapper(_FakeFlashInferWrapper):
        def get_actual_split_kv_plan(self):
            return [
                {"mode": "fixed", "split_size": 3, "boundaries": [(0, 3), (3, 6)]},
                {"mode": "fixed", "split_size": 3, "boundaries": [(0, 3), (3, 6)]},
            ]

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_MismatchedRuntimePlanWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_MismatchedRuntimePlanWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(
        FlashInferUnavailable,
        match="does not match|invalid actual|missing required fields",
    ):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                split_kv=SplitKVSpec.fixed(2),
            ),
        )


def test_flashinfer_pr7_strict_mode_requires_runtime_arithmetic_provenance():
    class _NoArithmeticProvenanceWrapper(_FakeFlashInferWrapper):
        get_attention_arithmetic_provenance = None

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_NoArithmeticProvenanceWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_NoArithmeticProvenanceWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="arithmetic provenance"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_strict_mode_requires_complete_runtime_plan_set():
    class _NoRuntimePlanSetWrapper(_FakeFlashInferWrapper):
        get_actual_split_kv_plan_set = None

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_NoRuntimePlanSetWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_NoRuntimePlanSetWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="complete batch/TP/CP/owner"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_runtime_plan_set_requires_explicit_reduction_semantics():
    class _MissingReductionSemanticsWrapper(_FakeFlashInferWrapper):
        def get_actual_split_kv_plan_set(self):
            plan_set = super().get_actual_split_kv_plan_set()
            del plan_set["entries"][0]["merge_order"]
            return plan_set

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_MissingReductionSemanticsWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_MissingReductionSemanticsWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="merge_order"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_runtime_split_plan_requires_explicit_fallback_fields():
    class _MissingFallbackFieldsWrapper(_FakeFlashInferWrapper):
        def get_actual_split_kv_plan(self):
            plans = super().get_actual_split_kv_plan()
            del plans[0]["fallback"]
            return plans

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_MissingFallbackFieldsWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_MissingFallbackFieldsWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="fallback"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_non_fp32_runtime_accumulation():
    class _WrongArithmeticWrapper(_FakeFlashInferWrapper):
        def get_attention_arithmetic_provenance(self):
            return {
                "accum_dtype": "bf16",
                "downcast_at": "per_split",
                "lse_dtype": "fp32",
                "source": "fake_runtime_capability",
            }

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_WrongArithmeticWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_WrongArithmeticWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="accum_dtype, downcast_at"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_runtime_output_dtype_boundary_mismatch():
    class _WrongOutputDTypeWrapper(_FakeFlashInferWrapper):
        def run_return_lse(self, q, paged_kv_cache):
            out, lse = super().run_return_lse(q, paged_kv_cache)
            return out.double(), lse

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_WrongOutputDTypeWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_WrongOutputDTypeWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="final output dtype"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_non_fp32_runtime_lse():
    class _WrongLSEDTypeWrapper(_FakeFlashInferWrapper):
        def run_return_lse(self, q, paged_kv_cache):
            out, lse = super().run_return_lse(q, paged_kv_cache)
            return out, lse.to(torch.bfloat16)

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_WrongLSEDTypeWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_WrongLSEDTypeWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="LSE must be FP32"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_required_cp_comm_until_cuda_ag_rs_ops_exist():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="p2p_nccl_reference"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                require_cp_comm=True,
            ),
        )


def test_flashinfer_pr7_rejects_implemented_cp_comm_status_in_scaffold():
    config = FlashInferPagedAttentionConfig(
        cp_comm_plan=AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
            status="implemented",
        )
    )

    with pytest.raises(ValueError, match="require_cp_comm"):
        config.validate(head_dim=8, query_len=1)


def test_flashinfer_pr7_rejects_post_rope_inputs_for_rope_llama_fusion():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="rotated twice"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                rope=FlashInferRoPEFusionConfig(q_rope_state="post_rope"),
            ),
        )


def test_attention_cp_partial_states_sort_by_global_block_index():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
    )

    ordered = sort_attention_cp_partial_states(
        (_partial_state(2), _partial_state(0), _partial_state(1)),
        plan=plan,
    )

    assert [state.block.global_block_index for state in ordered] == [0, 1, 2]


def test_attention_cp_partial_states_reject_duplicate_global_block_index():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
    )

    with pytest.raises(ValueError, match="duplicate global_block_index"):
        sort_attention_cp_partial_states(
            (_partial_state(1), _partial_state(1)),
            plan=plan,
        )


def test_cuda_ag_rs_attention_cp_comm_is_interface_only():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
    )
    communication = CUDAAGRSAttentionCPCommunication()

    with pytest.raises(AttentionCPCommunicationUnavailable, match="CUDA AG"):
        communication.all_gather_partial_states((_partial_state(0),), plan)

    merged = AttentionCPMergedState(
        out=torch.zeros(1, 2, 1, 4),
        lse=torch.zeros(1, 2, 1, dtype=torch.float32),
    )
    with pytest.raises(AttentionCPCommunicationUnavailable, match="CUDA RS"):
        communication.reduce_scatter_merged_state(merged, plan)


def test_cp_manifest_rejects_gap_wrong_owner_and_incomplete_gather():
    with pytest.raises(ValueError, match="gap-free"):
        AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
            backend="p2p_nccl_reference",
            status="implemented",
            expected_blocks=(
                AttentionCPBlockMetadata(0, 0, 2, 0, 0),
                AttentionCPBlockMetadata(1, 3, 4, 1, 0),
            ),
            expected_kv_token_range=(0, 4),
            query_token_ranges=((0, 1), (1, 2)),
        ).validate()

    with pytest.raises(ValueError, match="owner_tp_rank"):
        AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
            backend="p2p_nccl_reference",
            status="implemented",
            expected_blocks=(
                AttentionCPBlockMetadata(0, 0, 2, 0, 1),
                AttentionCPBlockMetadata(1, 2, 4, 1, 1),
            ),
            expected_kv_token_range=(0, 4),
            query_token_ranges=((0, 1), (1, 2)),
        ).validate()

    with pytest.raises(ValueError, match="expected KV token range|complete block manifest"):
        sort_attention_cp_partial_states((_partial_state(0),), plan=_p2p_plan())


def test_cp_manifest_rejects_wrong_local_cp_owner():
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(rank=0),
        validate_cuda_tensors=False,
    )
    wrong_owner = AttentionCPPartialState(
        out=torch.zeros(1, 2, 2, 4),
        lse=torch.zeros(1, 2, 2, dtype=torch.float32),
        block=AttentionCPBlockMetadata(0, 0, 2, 1, 0),
    )

    with pytest.raises(ValueError, match="wrong CP owner"):
        communication.all_gather_partial_states((wrong_owner,), _p2p_plan())


def test_cp_manifest_allows_sparse_global_block_indices_and_rejects_short_query_state():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(10, 0, 2, 0, 0),
            AttentionCPBlockMetadata(20, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 2)),
    )
    plan.validate()
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(rank=0),
        validate_cuda_tensors=False,
    )
    short_query = AttentionCPPartialState(
        out=torch.zeros(1, 2, 1, 4),
        lse=torch.zeros(1, 2, 1, dtype=torch.float32),
        block=plan.expected_blocks[0],
    )

    with pytest.raises(ValueError, match="complete query range"):
        communication.all_gather_partial_states((short_query,), plan)


def test_p2p_nccl_reference_gathers_manifest_order_and_scatters_query_range():
    remote_out = torch.full((1, 2, 2, 4), 7.0)
    remote_lse = torch.full((1, 2, 2), 3.0, dtype=torch.float32)
    distributed = _FakeNCCLDistributed(
        rank=0,
        receive_payloads=(remote_out, remote_lse),
    )
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=distributed,
        validate_cuda_tensors=False,
    )
    local = AttentionCPPartialState(
        out=torch.zeros(1, 2, 2, 4),
        lse=torch.zeros(1, 2, 2, dtype=torch.float32),
        block=AttentionCPBlockMetadata(0, 0, 2, 0, 0),
    )

    gathered = communication.all_gather_partial_states((local,), _p2p_plan())

    assert [state.block.global_block_index for state in gathered] == [0, 1]
    torch.testing.assert_close(gathered[1].out, remote_out)
    torch.testing.assert_close(gathered[1].lse, remote_lse)

    merged = AttentionCPMergedState(
        out=torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4),
        lse=torch.arange(4, dtype=torch.float32).reshape(1, 2, 2),
    )
    shard = communication.reduce_scatter_merged_state(merged, _p2p_plan())
    torch.testing.assert_close(shard.out, merged.out[:, :, 0:1, :])
    torch.testing.assert_close(shard.lse, merged.lse[:, :, 0:1])


def test_p2p_nccl_reference_fails_closed_on_non_nccl_backend():
    class _FakeGlooDistributed(_FakeNCCLDistributed):
        @staticmethod
        def get_backend(group=None):
            return "gloo"

    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeGlooDistributed(rank=0),
        validate_cuda_tensors=False,
    )

    with pytest.raises(AttentionCPCommunicationUnavailable, match="NCCL backend"):
        communication.all_gather_partial_states((_partial_state(0),), _p2p_plan())


def test_p2p_query_ranges_allow_empty_decode_shards():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 2, 0, 0),
            AttentionCPBlockMetadata(1, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 1)),
    )
    plan.validate()
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(rank=0),
        validate_cuda_tensors=False,
    )
    merged = AttentionCPMergedState(
        out=torch.zeros(1, 2, 1, 4),
        lse=torch.zeros(1, 2, 1, dtype=torch.float32),
    )

    local = communication.reduce_scatter_merged_state(merged, plan)

    assert local.out.shape == (1, 2, 1, 4)
    assert local.lse.shape == (1, 2, 1)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.distributed.is_available()
    or not torch.distributed.is_initialized()
    or "nccl" not in str(torch.distributed.get_backend()).lower()
    or torch.distributed.get_world_size() != 2,
    reason="requires an initialized two-rank NCCL process group",
)
def test_p2p_nccl_reference_real_process_group_smoke():
    rank = torch.distributed.get_rank()
    local = AttentionCPPartialState(
        out=torch.full((1, 2, 2, 4), float(rank), device="cuda"),
        lse=torch.full((1, 2, 2), float(rank), dtype=torch.float32, device="cuda"),
        block=AttentionCPBlockMetadata(rank, rank * 2, rank * 2 + 2, rank, 0),
    )

    gathered = P2PNCCLAttentionCPCommunication().all_gather_partial_states(
        (local,),
        _p2p_plan(cp_rank=rank),
    )

    assert [state.block.global_block_index for state in gathered] == [0, 1]


def test_flashinfer_pr7_real_backend_requires_cuda_before_importing_flashinfer():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp()

    with pytest.raises(FlashInferUnavailable, match="requires CUDA"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(mode="decode", workspace_size_bytes=1024),
        )


def test_flashinfer_pr7_plan_and_cache_materialization_follow_logical_page_order():
    q, k, v = _qkv(batch=1, query_len=1)
    positions = torch.full((1, 6), -1, dtype=torch.long)
    positions[:, 4:6] = torch.tensor([0, 1], dtype=torch.long)
    positions[:, 0:2] = torch.tensor([2, 3], dtype=torch.long)
    positions[:, 2:4] = torch.tensor([4, 5], dtype=torch.long)
    metadata = DecodeKVCacheMetadata(
        cache_position=torch.tensor([[5]], dtype=torch.long),
        kv_seq_lens=torch.tensor([6], dtype=torch.long),
        block_table=torch.tensor([[2, 0, 1]], dtype=torch.long),
        global_token_positions=positions,
        query_position_ids=torch.tensor([[5]], dtype=torch.long),
        key_position_ids=positions.clone(),
        page_size=2,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )

    plan = build_flashinfer_paged_kv_plan(
        metadata,
        batch_size=1,
        query_len=1,
        cache_capacity=k.size(2),
        device=q.device,
    )
    k_pages, v_pages = materialize_flashinfer_paged_kv_cache(k, v, page_size=2)

    assert plan.paged_kv_indices.tolist() == [2, 0, 1]
    torch.testing.assert_close(k_pages[2], k[0, :, 4:6, :].transpose(0, 1))
    torch.testing.assert_close(v_pages[0], v[0, :, 0:2, :].transpose(0, 1))


def test_flashinfer_pr7_plan_rejects_position_metadata_mismatch():
    q, k, _ = _qkv(batch=1, query_len=1)
    metadata = DecodeKVCacheMetadata(
        cache_position=torch.tensor([[5]], dtype=torch.long),
        kv_seq_lens=torch.tensor([6], dtype=torch.long),
        block_table=torch.tensor([[2, 0, 1]], dtype=torch.long),
        global_token_positions=torch.arange(6, dtype=torch.long).unsqueeze(0),
        query_position_ids=torch.tensor([[5]], dtype=torch.long),
        key_position_ids=torch.arange(6, dtype=torch.long).unsqueeze(0),
        page_size=2,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )

    with pytest.raises(ValueError, match="reconstruct logical positions"):
        build_flashinfer_paged_kv_plan(
            metadata,
            batch_size=1,
            query_len=1,
            cache_capacity=k.size(2),
            device=q.device,
        )
