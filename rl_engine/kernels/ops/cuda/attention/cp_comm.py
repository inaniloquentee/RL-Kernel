# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CP/TP attention communication interfaces and a P2P NCCL reference.

PR7 evaluates fused attention backends for the #235 target
``Qwen3-8B, TP=2, CP=2, BF16``.  The self-owned CUDA communication operators are
AG/RS and compute-communication decoupled.  They are not implemented in this
scaffold, but their interface is exposed here so backend adapters cannot
silently ignore the distributed contract.

The production communication path is expected to move attention partial states:

```text
local FlashInfer/TE attention over rank-owned KV blocks
  -> AttentionCPPartialState(out, lse, global_block_index, tp/cp rank metadata)
  -> custom CUDA AG communication operator
  -> sort by global_block_index
  -> PR3 FP32 online-softmax merge
  -> custom CUDA RS communication operator
```
The custom CUDA AG/RS interface remains fail-closed.  The P2P NCCL backend is
an intentionally simple, correctness-first implementation of the same
protocol: it exchanges tensors peer-to-peer, reconstructs metadata from an
authoritative manifest, and validates complete logical coverage before merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import torch

CPCommunicationBackend = Literal["cuda_ag_rs", "p2p_nccl_reference", "local_debug"]
CPCommunicationStatus = Literal["interface_only", "implemented"]


class AttentionCPCommunicationUnavailable(RuntimeError):
    """Raised when a requested CP communication backend is not implemented."""


@dataclass(frozen=True)
class AttentionParallelSpec:
    """TP/CP identity carried by PR7 attention backend reports."""

    tp_world_size: int = 2
    tp_rank: int = 0
    cp_world_size: int = 2
    cp_rank: int = 0

    def validate(self) -> None:
        _positive_int(self.tp_world_size, "tp_world_size")
        _positive_int(self.cp_world_size, "cp_world_size")
        _rank_in_world(self.tp_rank, self.tp_world_size, "tp_rank")
        _rank_in_world(self.cp_rank, self.cp_world_size, "cp_rank")

    def provenance(self) -> dict[str, int]:
        self.validate()
        return {
            "tp_world_size": int(self.tp_world_size),
            "tp_rank": int(self.tp_rank),
            "cp_world_size": int(self.cp_world_size),
            "cp_rank": int(self.cp_rank),
        }


@dataclass(frozen=True)
class AttentionCPBlockMetadata:
    """Logical identity for one attention partial state."""

    global_block_index: int
    kv_block_start: int
    kv_block_end: int
    owner_cp_rank: int
    owner_tp_rank: int

    def validate(self, parallel: AttentionParallelSpec) -> None:
        parallel.validate()
        if (
            isinstance(self.global_block_index, bool)
            or not isinstance(self.global_block_index, int)
            or self.global_block_index < 0
        ):
            raise ValueError("global_block_index must be non-negative")
        if (
            isinstance(self.kv_block_start, bool)
            or isinstance(self.kv_block_end, bool)
            or not isinstance(self.kv_block_start, int)
            or not isinstance(self.kv_block_end, int)
            or self.kv_block_start < 0
            or self.kv_block_end <= self.kv_block_start
        ):
            raise ValueError("KV block bounds must satisfy 0 <= start < end")
        _rank_in_world(self.owner_cp_rank, parallel.cp_world_size, "owner_cp_rank")
        _rank_in_world(self.owner_tp_rank, parallel.tp_world_size, "owner_tp_rank")

    def provenance(self) -> dict[str, int]:
        return {
            "global_block_index": int(self.global_block_index),
            "kv_block_start": int(self.kv_block_start),
            "kv_block_end": int(self.kv_block_end),
            "owner_cp_rank": int(self.owner_cp_rank),
            "owner_tp_rank": int(self.owner_tp_rank),
        }


@dataclass(frozen=True)
class AttentionCPPartialState:
    """One local or received ``(out, lse)`` state before CP merge."""

    out: torch.Tensor
    lse: torch.Tensor
    block: AttentionCPBlockMetadata

    def validate(self, parallel: AttentionParallelSpec) -> None:
        self.block.validate(parallel)
        if self.out.ndim != 4:
            raise ValueError("partial out must have shape [B, Hq, Sq, D]")
        if self.lse.ndim != 3:
            raise ValueError("partial lse must have shape [B, Hq, Sq]")
        if self.out.shape[:3] != self.lse.shape:
            raise ValueError("partial out and lse must share [B, Hq, Sq]")
        if self.out.device != self.lse.device:
            raise ValueError("partial out and lse must be on the same device")
        if self.lse.dtype != torch.float32:
            raise ValueError("partial lse must be attention-domain FP32")
        if self.out.dtype != torch.float32:
            raise ValueError("partial out must remain FP32 until the final write")


@dataclass(frozen=True)
class AttentionCPMergedState:
    """Merged attention state before the CUDA RS communication operator."""

    out: torch.Tensor
    lse: torch.Tensor

    def validate(self) -> None:
        if self.out.ndim != 4:
            raise ValueError("merged out must have shape [B, Hq, Sq, D]")
        if self.lse.ndim != 3:
            raise ValueError("merged lse must have shape [B, Hq, Sq]")
        if self.out.shape[:3] != self.lse.shape:
            raise ValueError("merged out and lse must share [B, Hq, Sq]")
        if self.out.device != self.lse.device:
            raise ValueError("merged out and lse must be on the same device")
        if self.lse.dtype != torch.float32:
            raise ValueError("merged lse must be attention-domain FP32")
        if self.out.dtype != torch.float32:
            raise ValueError("merged out must remain FP32 until the final write")


@dataclass(frozen=True)
class AttentionCPCommunicationPlan:
    """Requested AG/RS communication contract for CP attention partial states."""

    parallel: AttentionParallelSpec
    backend: CPCommunicationBackend = "cuda_ag_rs"
    status: CPCommunicationStatus = "interface_only"
    pattern: str = "ag_rs"
    compute_communication: str = "decoupled"
    merge_order: str = "global_block_index"
    accum_dtype: torch.dtype = torch.float32
    return_lse: bool = True
    expected_blocks: tuple[AttentionCPBlockMetadata, ...] = ()
    expected_kv_token_range: tuple[int, int] | None = None
    query_token_ranges: tuple[tuple[int, int], ...] = ()
    merge_root_cp_rank: int = 0

    def validate(self) -> None:
        self.parallel.validate()
        if self.backend not in {"cuda_ag_rs", "p2p_nccl_reference", "local_debug"}:
            raise ValueError(f"unsupported CP communication backend: {self.backend}")
        if self.status not in {"interface_only", "implemented"}:
            raise ValueError(f"unsupported CP communication status: {self.status}")
        if self.pattern != "ag_rs":
            raise ValueError("PR7 CP communication must use the custom CUDA AG/RS interface")
        if self.compute_communication != "decoupled":
            raise ValueError("PR7 CP communication must keep compute and communication decoupled")
        if self.merge_order != "global_block_index":
            raise ValueError("PR7 CP communication must preserve global_block_index merge order")
        if self.accum_dtype is not torch.float32:
            raise ValueError("PR7 CP merge accumulation must be FP32")
        if not self.return_lse:
            raise ValueError("PR7 CP communication requires LSE-carrying partial states")
        _rank_in_world(
            self.merge_root_cp_rank,
            self.parallel.cp_world_size,
            "merge_root_cp_rank",
        )
        _validate_expected_block_manifest(self)
        _validate_query_token_ranges(self)
        if self.backend == "p2p_nccl_reference":
            if self.status != "implemented":
                raise ValueError("P2P NCCL reference plans must use status='implemented'")
            if not self.expected_blocks or self.expected_kv_token_range is None:
                raise ValueError(
                    "P2P NCCL reference requires a complete expected block manifest"
                )
            if not self.query_token_ranges:
                raise ValueError(
                    "P2P NCCL reference requires one query range per CP rank"
                )

    def provenance(self) -> dict[str, object]:
        self.validate()
        return {
            "cp_comm_backend": self.backend,
            "cp_comm_status": self.status,
            "cp_comm_pattern": self.pattern,
            "cp_comm_compute_communication": self.compute_communication,
            "cp_comm_merge_order": self.merge_order,
            "cp_comm_accum_dtype": "fp32",
            "cp_comm_return_lse": self.return_lse,
            "cp_comm_contract": "partial_out_lse_global_block_index",
            "cp_comm_expected_kv_token_range": (
                None
                if self.expected_kv_token_range is None
                else list(self.expected_kv_token_range)
            ),
            "cp_comm_expected_blocks": [
                block.provenance() for block in self.expected_blocks
            ],
            "cp_comm_query_token_ranges": [
                list(bounds) for bounds in self.query_token_ranges
            ],
            "cp_comm_merge_root_cp_rank": int(self.merge_root_cp_rank),
            **self.parallel.provenance(),
        }


class AttentionCPCommunication(Protocol):
    """Protocol future custom CUDA AG/RS communication operators must implement."""

    def all_gather_partial_states(
        self,
        local_states: tuple[AttentionCPPartialState, ...],
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[AttentionCPPartialState, ...]:
        """Run the custom CUDA AG operator and return gathered partial states."""

    def reduce_scatter_merged_state(
        self,
        merged_state: AttentionCPMergedState,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPMergedState:
        """Run the custom CUDA RS operator and return this rank's output shard."""


class CUDAAGRSAttentionCPCommunication:
    """Fail-closed placeholder for future custom CUDA AG/RS communication operators."""

    def all_gather_partial_states(
        self,
        local_states: tuple[AttentionCPPartialState, ...],
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[AttentionCPPartialState, ...]:
        plan.validate()
        for state in local_states:
            state.validate(plan.parallel)
        raise AttentionCPCommunicationUnavailable(
            "custom CUDA AG attention communication is interface-only in this PR7 scaffold; "
            "future implementation must gather AttentionCPPartialState tensors before "
            "global_block_index sorting and PR3 FP32 merge"
        )

    def reduce_scatter_merged_state(
        self,
        merged_state: AttentionCPMergedState,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPMergedState:
        plan.validate()
        merged_state.validate()
        raise AttentionCPCommunicationUnavailable(
            "custom CUDA RS attention communication is interface-only in this PR7 scaffold; "
            "future implementation must scatter the PR3-merged attention state to CP ranks"
        )


class P2PNCCLAttentionCPCommunication:
    """Correctness-first P2P NCCL implementation of the CP protocol.

    The block manifest is authoritative.  Only ``out`` and ``lse`` tensors are
    transported, in deterministic peer/block order; received metadata is
    reconstructed from the manifest and then validated as a complete set.
    ``reduce_scatter_merged_state`` uses a designated root and explicit P2P
    sends so its numerical behavior is easy to compare with a future CUDA RS.
    """

    def __init__(
        self,
        *,
        process_group: Any = None,
        dist_module: Any = None,
        validate_cuda_tensors: bool = True,
    ) -> None:
        if dist_module is None:
            import torch.distributed as dist

            dist_module = dist
        self._dist = dist_module
        self._group = process_group
        self._validate_cuda_tensors = validate_cuda_tensors

    def all_gather_partial_states(
        self,
        local_states: tuple[AttentionCPPartialState, ...],
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[AttentionCPPartialState, ...]:
        self._validate_runtime(plan)
        _validate_local_partial_states(local_states, plan)
        self._require_cuda(local_states[0].out, local_states[0].lse)
        ordered_local_states = tuple(
            sorted(local_states, key=lambda state: state.block.global_block_index)
        )
        template = ordered_local_states[0]
        if template.out.size(2) != plan.query_token_ranges[-1][1]:
            raise ValueError(
                "local partial states must cover the complete query range before gather"
            )
        received: list[AttentionCPPartialState] = []
        operations: list[Any] = []
        receive_tensors: list[
            tuple[AttentionCPBlockMetadata, torch.Tensor, torch.Tensor]
        ] = []

        for peer_cp_rank in range(plan.parallel.cp_world_size):
            if peer_cp_rank == plan.parallel.cp_rank:
                continue
            peer = self._global_peer(peer_cp_rank)
            peer_blocks = _expected_blocks_for_cp_rank(plan, peer_cp_rank)
            for block in peer_blocks:
                out = torch.empty_like(template.out)
                lse = torch.empty_like(template.lse)
                operations.extend(
                    (
                        self._dist.P2POp(
                            self._dist.irecv,
                            out,
                            peer,
                            group=self._group,
                        ),
                        self._dist.P2POp(
                            self._dist.irecv,
                            lse,
                            peer,
                            group=self._group,
                        ),
                    )
                )
                receive_tensors.append((block, out, lse))
            for state in ordered_local_states:
                operations.extend(
                    (
                        self._dist.P2POp(
                            self._dist.isend,
                            state.out.contiguous(),
                            peer,
                            group=self._group,
                        ),
                        self._dist.P2POp(
                            self._dist.isend,
                            state.lse.contiguous(),
                            peer,
                            group=self._group,
                        ),
                    )
                )

        self._run_operations(operations)
        for block, out, lse in receive_tensors:
            received.append(AttentionCPPartialState(out=out, lse=lse, block=block))
        return sort_attention_cp_partial_states(
            (*ordered_local_states, *received),
            plan=plan,
        )

    def reduce_scatter_merged_state(
        self,
        merged_state: AttentionCPMergedState,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPMergedState:
        self._validate_runtime(plan)
        merged_state.validate()
        self._require_cuda(merged_state.out, merged_state.lse)
        ranges = plan.query_token_ranges
        full_query_tokens = ranges[-1][1]
        if merged_state.out.size(2) != full_query_tokens:
            raise ValueError(
                "merged state query length does not match query_token_ranges coverage"
            )

        rank = plan.parallel.cp_rank
        root = plan.merge_root_cp_rank
        local_start, local_end = ranges[rank]
        if rank == root:
            operations: list[Any] = []
            for peer_cp_rank, (start, end) in enumerate(ranges):
                if peer_cp_rank == root or start == end:
                    continue
                peer = self._global_peer(peer_cp_rank)
                operations.extend(
                    (
                        self._dist.P2POp(
                            self._dist.isend,
                            merged_state.out[:, :, start:end, :].contiguous(),
                            peer,
                            group=self._group,
                        ),
                        self._dist.P2POp(
                            self._dist.isend,
                            merged_state.lse[:, :, start:end].contiguous(),
                            peer,
                            group=self._group,
                        ),
                    )
                )
            self._run_operations(operations)
            result = AttentionCPMergedState(
                out=merged_state.out[:, :, local_start:local_end, :].contiguous(),
                lse=merged_state.lse[:, :, local_start:local_end].contiguous(),
            )
        else:
            local_query_tokens = local_end - local_start
            if local_query_tokens == 0:
                result = AttentionCPMergedState(
                    out=merged_state.out[:, :, 0:0, :].contiguous(),
                    lse=merged_state.lse[:, :, 0:0].contiguous(),
                )
                result.validate()
                return result
            out = torch.empty(
                (*merged_state.out.shape[:2], local_query_tokens, merged_state.out.size(3)),
                dtype=merged_state.out.dtype,
                device=merged_state.out.device,
            )
            lse = torch.empty(
                (*merged_state.lse.shape[:2], local_query_tokens),
                dtype=merged_state.lse.dtype,
                device=merged_state.lse.device,
            )
            peer = self._global_peer(root)
            self._run_operations(
                [
                    self._dist.P2POp(
                        self._dist.irecv,
                        out,
                        peer,
                        group=self._group,
                    ),
                    self._dist.P2POp(
                        self._dist.irecv,
                        lse,
                        peer,
                        group=self._group,
                    ),
                ]
            )
            result = AttentionCPMergedState(out=out, lse=lse)
        result.validate()
        return result

    def _validate_runtime(self, plan: AttentionCPCommunicationPlan) -> None:
        plan.validate()
        if plan.backend != "p2p_nccl_reference" or plan.status != "implemented":
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL communication requires an implemented p2p_nccl_reference plan"
            )
        dist = self._dist
        if not dist.is_available() or not dist.is_initialized():
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL communication requires initialized torch.distributed"
            )
        backend = str(dist.get_backend(self._group)).lower()
        if "nccl" not in backend:
            raise AttentionCPCommunicationUnavailable(
                f"P2P NCCL communication requires the NCCL backend; got {backend}"
            )
        world_size = int(dist.get_world_size(self._group))
        rank = int(dist.get_rank(self._group))
        if world_size != plan.parallel.cp_world_size:
            raise AttentionCPCommunicationUnavailable(
                "process-group world size does not match cp_world_size"
            )
        if rank != plan.parallel.cp_rank:
            raise AttentionCPCommunicationUnavailable(
                "process-group rank does not match the communication plan cp_rank"
            )
        local_blocks = _expected_blocks_for_cp_rank(plan, plan.parallel.cp_rank)
        if not local_blocks:
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL communication requires every CP rank to own at least one block"
            )

    def _global_peer(self, peer_cp_rank: int) -> int:
        get_global_rank = getattr(self._dist, "get_global_rank", None)
        if self._group is not None and callable(get_global_rank):
            return int(get_global_rank(self._group, peer_cp_rank))
        return peer_cp_rank

    def _run_operations(self, operations: Sequence[Any]) -> None:
        if not operations:
            return
        requests = self._dist.batch_isend_irecv(list(operations))
        for request in requests:
            request.wait()

    def _require_cuda(self, *tensors: torch.Tensor) -> None:
        if self._validate_cuda_tensors and any(
            tensor.device.type != "cuda" for tensor in tensors
        ):
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL communication requires CUDA tensors"
            )

def sort_attention_cp_partial_states(
    states: tuple[AttentionCPPartialState, ...],
    *,
    plan: AttentionCPCommunicationPlan,
) -> tuple[AttentionCPPartialState, ...]:
    """Validate and sort partial states by ``global_block_index``."""

    plan.validate()
    if not states:
        raise ValueError("at least one CP attention partial state is required")
    for state in states:
        state.validate(plan.parallel)
    ordered = tuple(sorted(states, key=lambda state: state.block.global_block_index))
    indices = [state.block.global_block_index for state in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate global_block_index values are not allowed")
    if not plan.expected_blocks and indices != list(range(len(ordered))):
        raise ValueError(
            "partial states without a manifest must cover global_block_index "
            "values [0, block_count)"
        )
    if not plan.expected_blocks and ordered[0].block.kv_block_start != 0:
        raise ValueError(
            "partial states without a manifest must start at KV token 0"
        )
    if plan.expected_kv_token_range is not None:
        expected_start, expected_end = plan.expected_kv_token_range
        if (
            ordered[0].block.kv_block_start != expected_start
            or ordered[-1].block.kv_block_end != expected_end
        ):
            raise ValueError(
                "partial states do not cover the declared expected KV token range"
            )
    _validate_partial_state_set(ordered, plan)
    return ordered


def _validate_expected_block_manifest(plan: AttentionCPCommunicationPlan) -> None:
    blocks = plan.expected_blocks
    if not blocks:
        if plan.expected_kv_token_range is not None:
            raise ValueError("expected_kv_token_range requires expected_blocks")
        return
    for block in blocks:
        block.validate(plan.parallel)
        if block.owner_tp_rank != plan.parallel.tp_rank:
            raise ValueError(
                "expected block owner_tp_rank must match the plan TP shard"
            )
    ordered = tuple(sorted(blocks, key=lambda block: block.global_block_index))
    indices = tuple(block.global_block_index for block in ordered)
    if len(set(indices)) != len(indices):
        raise ValueError("expected block manifest contains duplicate global_block_index values")
    if len(set(blocks)) != len(blocks):
        raise ValueError("expected block manifest contains duplicate metadata")
    owners = {block.owner_cp_rank for block in blocks}
    if owners != set(range(plan.parallel.cp_world_size)):
        raise ValueError("expected block manifest must assign work to every CP rank")
    expected_range = plan.expected_kv_token_range
    if expected_range is None:
        raise ValueError("expected block manifest requires expected_kv_token_range")
    try:
        start, end = expected_range
    except (TypeError, ValueError) as exc:
        raise ValueError("expected KV range must contain exactly (start, end)") from exc
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError("expected KV range must satisfy 0 <= start < end")
    if ordered[0].kv_block_start != start or ordered[-1].kv_block_end != end:
        raise ValueError("expected block manifest does not cover the declared KV range")
    previous_end = start
    for block in ordered:
        if block.kv_block_start != previous_end:
            raise ValueError("expected block manifest must be gap-free and non-overlapping")
        previous_end = block.kv_block_end


def _validate_query_token_ranges(plan: AttentionCPCommunicationPlan) -> None:
    ranges = plan.query_token_ranges
    if not ranges:
        return
    if len(ranges) != plan.parallel.cp_world_size:
        raise ValueError("query_token_ranges must contain one range per CP rank")
    previous_end = 0
    for bounds in ranges:
        try:
            start, end = bounds
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "query token ranges must contain (start, end) pairs"
            ) from exc
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError("query token ranges must contain (start, end) pairs")
        if start != previous_end or end < start:
            raise ValueError(
                "query token ranges must be non-negative, contiguous, and start at 0"
            )
        previous_end = end
    if previous_end == 0:
        raise ValueError("query token ranges must cover at least one query token")


def _validate_local_partial_states(
    states: tuple[AttentionCPPartialState, ...],
    plan: AttentionCPCommunicationPlan,
) -> None:
    expected = _expected_blocks_for_cp_rank(plan, plan.parallel.cp_rank)
    if not states:
        raise ValueError("each CP rank must provide at least one local partial state")
    for state in states:
        state.validate(plan.parallel)
        if state.block.owner_cp_rank != plan.parallel.cp_rank:
            raise ValueError("local partial state has the wrong CP owner")
        if state.block.owner_tp_rank != plan.parallel.tp_rank:
            raise ValueError("local partial state has the wrong TP owner")
    actual = tuple(
        state.block
        for state in sorted(states, key=lambda item: item.block.global_block_index)
    )
    if actual != expected:
        raise ValueError("local partial states do not exactly match the rank manifest")
    _validate_common_state_shapes(states)


def _expected_blocks_for_cp_rank(
    plan: AttentionCPCommunicationPlan,
    cp_rank: int,
) -> tuple[AttentionCPBlockMetadata, ...]:
    return tuple(
        block
        for block in sorted(
            plan.expected_blocks,
            key=lambda item: item.global_block_index,
        )
        if block.owner_cp_rank == cp_rank
    )


def _validate_partial_state_set(
    states: tuple[AttentionCPPartialState, ...],
    plan: AttentionCPCommunicationPlan,
) -> None:
    _validate_common_state_shapes(states)
    previous_end = states[0].block.kv_block_start
    for state in states:
        if state.block.owner_tp_rank != plan.parallel.tp_rank:
            raise ValueError("partial state has the wrong TP owner for this CP group")
        if state.block.kv_block_start != previous_end:
            raise ValueError("partial state KV ranges must be gap-free and non-overlapping")
        previous_end = state.block.kv_block_end
    if plan.expected_blocks:
        actual = tuple(state.block for state in states)
        expected = tuple(
            sorted(plan.expected_blocks, key=lambda block: block.global_block_index)
        )
        if actual != expected:
            raise ValueError(
                "gathered partial states do not exactly match the complete block manifest"
            )


def _validate_common_state_shapes(states: Sequence[AttentionCPPartialState]) -> None:
    first = states[0]
    for state in states[1:]:
        if state.out.shape != first.out.shape or state.lse.shape != first.lse.shape:
            raise ValueError("all CP partial states must have matching out/lse shapes")
        if state.out.dtype != first.out.dtype or state.lse.dtype != first.lse.dtype:
            raise ValueError("all CP partial states must have matching out/lse dtypes")
        if state.out.device != first.out.device or state.lse.device != first.lse.device:
            raise ValueError("all CP partial states must be on the same device")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _rank_in_world(rank: int, world_size: int, name: str) -> None:
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank >= world_size
    ):
        raise ValueError(f"{name} must be in [0, world_size)")


__all__ = [
    "AttentionCPBlockMetadata",
    "AttentionCPCommunication",
    "AttentionCPCommunicationPlan",
    "AttentionCPCommunicationUnavailable",
    "AttentionCPMergedState",
    "AttentionCPPartialState",
    "AttentionParallelSpec",
    "CPCommunicationBackend",
    "CPCommunicationStatus",
    "CUDAAGRSAttentionCPCommunication",
    "P2PNCCLAttentionCPCommunication",
    "sort_attention_cp_partial_states",
]
