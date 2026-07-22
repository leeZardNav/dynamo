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
    model: String,
}

impl PylonStatsPublisher {
    pub(crate) fn attach(
        inner: PylonStats,
        model: &str,
        expected_dp_ranks: u32,
        block_size_tokens: u32,
    ) -> PyResult<Self> {
        inner
            .configure_kv_cache(model, expected_dp_ranks, block_size_tokens)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        inner.mark_request_stats_producer_available();
        Ok(Self {
            inner,
            model: model.trim().to_owned(),
        })
    }
}

#[pymethods]
impl PylonStatsPublisher {
    /// Publish one cumulative request-counter event without blocking inference.
    #[pyo3(signature = (request_id, tokens_processed=None, tokens_generated=None, finished=false))]
    fn publish_stats_event(
        &self,
        request_id: &str,
        tokens_processed: Option<u64>,
        tokens_generated: Option<u64>,
        finished: bool,
    ) -> PyResult<()> {
        self.inner
            .publish_request_stats(RequestStatsUpdate {
                request_id,
                model: &self.model,
                tokens_processed,
                tokens_generated,
                finished,
            })
            .map_err(request_stats_error)
    }

    /// Replace one rank's latest observed KV block state.
    #[pyo3(signature = (dp_rank, used_blocks, total_blocks))]
    fn update_kv_snapshot(
        &self,
        dp_rank: u32,
        used_blocks: u64,
        total_blocks: u64,
    ) -> PyResult<()> {
        self.inner
            .update_kv_snapshot(KvCacheSnapshot {
                dp_rank,
                used_blocks,
                total_blocks,
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
