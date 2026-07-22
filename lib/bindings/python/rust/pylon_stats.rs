// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::pylon_stats::{
    KvCacheSnapshot, PylonStats, RequestStatsPublishError, RequestStatsUpdate,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Narrow Python publishing surface for the runtime-owned Pylon state.
///
/// Python supplies observed facts only. Channel ownership, subscriber lag,
/// aggregation, and HTTP response derivation remain in `dynamo-runtime`.
#[pyclass(module = "dynamo._core", frozen)]
pub(crate) struct PylonStatsPublisher {
    inner: PylonStats,
}

impl PylonStatsPublisher {
    pub(crate) fn attach(inner: PylonStats) -> Self {
        inner.mark_request_stats_producer_available();
        Self { inner }
    }
}

#[pymethods]
impl PylonStatsPublisher {
    /// Publish one cumulative request-counter event without blocking inference.
    #[pyo3(signature = (request_id, model, tokens_processed=None, tokens_generated=None, finished=false))]
    fn publish_stats_event(
        &self,
        request_id: &str,
        model: &str,
        tokens_processed: Option<u64>,
        tokens_generated: Option<u64>,
        finished: bool,
    ) -> PyResult<()> {
        self.inner
            .publish_request_stats(RequestStatsUpdate {
                request_id,
                model,
                tokens_processed,
                tokens_generated,
                finished,
            })
            .map_err(request_stats_error)
    }

    /// Replace one rank's latest observed KV block state.
    #[pyo3(signature = (model, dp_rank, expected_dp_ranks, used_blocks, total_blocks, block_size_tokens))]
    fn update_kv_snapshot(
        &self,
        model: &str,
        dp_rank: u32,
        expected_dp_ranks: u32,
        used_blocks: u64,
        total_blocks: u64,
        block_size_tokens: u32,
    ) -> PyResult<()> {
        self.inner
            .update_kv_snapshot(KvCacheSnapshot {
                model,
                dp_rank,
                expected_dp_ranks,
                used_blocks,
                total_blocks,
                block_size_tokens,
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

fn request_stats_error(error: RequestStatsPublishError) -> PyErr {
    if matches!(&error, RequestStatsPublishError::Serialize(_)) {
        PyRuntimeError::new_err(error.to_string())
    } else {
        PyValueError::new_err(error.to_string())
    }
}
