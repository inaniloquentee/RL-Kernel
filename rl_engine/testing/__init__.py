# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Testing helpers for RL-shaped kernel validation."""

from .attention_comparison import (
    AttentionComparisonInputs,
    AttentionComparisonReport,
    AttentionPathDrift,
    AttentionPathResult,
    DecodeAttentionInputs,
    DecodeKVCacheMetadata,
    DriftStats,
    TransformerEngineUnavailable,
    compare_decode_kv_replay,
    compare_single_gpu_attention,
    compare_single_gpu_rope_attention,
    decode_prefix_cache_fingerprint,
    run_chunked_query_attention,
    run_decode_full_prefill_reference,
    run_decode_kv_replay,
    run_full_attention,
    run_fused_like_rope_attention,
    run_paged_kv_attention,
    run_unfused_rope_attention,
    transformer_engine_context_parallel_available,
)
from .reference_ops import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    masked_sum,
    selected_logprobs_reference,
    summarize_kernel_drift,
)
from .rl_batch import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch

__all__ = [
    "AttentionComparisonInputs",
    "AttentionComparisonReport",
    "AttentionPathDrift",
    "AttentionPathResult",
    "DecodeAttentionInputs",
    "DecodeKVCacheMetadata",
    "DriftStats",
    "SyntheticRLKernelBatch",
    "TransformerEngineUnavailable",
    "active_token_count",
    "compare_single_gpu_rope_attention",
    "compare_single_gpu_attention",
    "compare_decode_kv_replay",
    "decode_prefix_cache_fingerprint",
    "compute_policy_ratio",
    "compute_reference_kl",
    "make_synthetic_rl_kernel_batch",
    "masked_mean",
    "masked_sum",
    "run_chunked_query_attention",
    "run_decode_full_prefill_reference",
    "run_decode_kv_replay",
    "run_fused_like_rope_attention",
    "run_full_attention",
    "run_paged_kv_attention",
    "run_unfused_rope_attention",
    "selected_logprobs_reference",
    "summarize_kernel_drift",
    "transformer_engine_context_parallel_available",
]
