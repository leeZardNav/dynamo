// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded request-stats fanout and latest observed KV state for Pylon routes.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use bytes::Bytes;
use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::broadcast;

const DEFAULT_STATS_CHANNEL_CAPACITY: usize = 1024;

/// One cumulative request-counter update for Pylon's NDJSON stream.
#[derive(Clone, Copy, Debug)]
pub struct RequestStatsUpdate<'a> {
    pub request_id: &'a str,
    pub model: &'a str,
    pub tokens_processed: Option<u64>,
    pub tokens_generated: Option<u64>,
    pub finished: bool,
}

/// Latest reliable KV block observation for one local DP rank.
#[derive(Clone, Copy, Debug)]
pub struct KvCacheSnapshot<'a> {
    pub model: &'a str,
    pub dp_rank: u32,
    pub expected_dp_ranks: u32,
    pub used_blocks: u64,
    pub total_blocks: u64,
    pub block_size_tokens: u32,
}

/// Pylon's worker-local `/kv-cache/stats` response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KvCacheStats {
    pub model: String,
    pub kv_cache_capacity_tokens: u64,
    pub kv_cache_used_tokens: u64,
    pub kv_cache_free_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum RequestStatsPublishError {
    #[error("request_id must not be empty")]
    EmptyRequestId,
    #[error("model must not be empty")]
    EmptyModel,
    #[error("a stats event requires at least one counter unless finished is true")]
    MissingCounters,
    #[error("failed to serialize request stats: {0}")]
    Serialize(#[from] serde_json::Error),
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum KvCacheSnapshotError {
    #[error("model must not be empty")]
    EmptyModel,
    #[error("expected_dp_ranks must be greater than zero")]
    ZeroExpectedDpRanks,
    #[error("block_size_tokens must be greater than zero")]
    ZeroBlockSize,
    #[error("total_blocks must be greater than zero")]
    ZeroTotalBlocks,
    #[error("used_blocks ({used_blocks}) exceeds total_blocks ({total_blocks})")]
    UsedBlocksExceedTotal { used_blocks: u64, total_blocks: u64 },
    #[error("model {model:?} changed expected local DP ranks from {previous} to {observed}")]
    ExpectedDpRanksChanged {
        model: String,
        previous: u32,
        observed: u32,
    },
    #[error("model {model:?} changed KV block size from {previous} to {observed} tokens")]
    BlockSizeChanged {
        model: String,
        previous: u32,
        observed: u32,
    },
    #[error("rank {dp_rank} is outside model {model:?}'s {expected} local DP ranks")]
    UnexpectedDpRank {
        model: String,
        dp_rank: u32,
        expected: u32,
    },
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum KvCacheStatsUnavailable {
    #[error("KV-cache stats are not ready: no reliable snapshot has been observed")]
    NoSnapshots,
    #[error(
        "KV-cache stats are not ready for model {model:?}: observed {observed} of {expected} local DP ranks"
    )]
    IncompleteDpRanks {
        model: String,
        observed: usize,
        expected: u32,
    },
    #[error("KV-cache stats are ambiguous across local models: {models:?}")]
    AmbiguousModels { models: Vec<String> },
    #[error("KV-cache token counts overflowed for model {model:?}")]
    TokenCountOverflow { model: String },
}

impl KvCacheStatsUnavailable {
    pub(crate) fn response_code(&self) -> &'static str {
        match self {
            Self::NoSnapshots | Self::IncompleteDpRanks { .. } => "kv_cache_stats_not_ready",
            Self::AmbiguousModels { .. } | Self::TokenCountOverflow { .. } => {
                "kv_cache_stats_unavailable"
            }
        }
    }
}

#[derive(Clone)]
pub struct PylonStats {
    inner: Arc<PylonStatsInner>,
}

struct PylonStatsInner {
    request_stats_producer_status: AtomicU8,
    request_stats_tx: broadcast::Sender<Bytes>,
    kv_models: Mutex<HashMap<String, ModelKvState>>,
}

struct ModelKvState {
    expected_dp_ranks: u32,
    block_size_tokens: u32,
    ranks: HashMap<u32, RankKvSnapshot>,
}

#[derive(Clone, Copy)]
struct RankKvSnapshot {
    used_blocks: u64,
    total_blocks: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum RequestStatsProducerStatus {
    Pending,
    Available,
    Unavailable,
}

#[derive(Serialize)]
struct RequestStatsEvent<'a> {
    v: u8,
    #[serde(rename = "type")]
    event_type: &'static str,
    request_id: &'a str,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_processed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_generated: Option<u64>,
    finished: bool,
}

impl Default for PylonStats {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_STATS_CHANNEL_CAPACITY)
    }
}

impl PylonStats {
    fn with_capacity(request_stats_capacity: usize) -> Self {
        let (request_stats_tx, _) = broadcast::channel(request_stats_capacity);
        Self {
            inner: Arc::new(PylonStatsInner {
                request_stats_producer_status: AtomicU8::new(
                    RequestStatsProducerStatus::Pending as u8,
                ),
                request_stats_tx,
                kv_models: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Mark request stats available before the worker enters discovery.
    pub fn mark_request_stats_producer_available(&self) {
        self.inner.request_stats_producer_status.store(
            RequestStatsProducerStatus::Available as u8,
            Ordering::Relaxed,
        );
    }

    /// Finish backend handoff without overriding an attached producer.
    pub fn complete_request_stats_producer_registration(&self) {
        let _ = self.inner.request_stats_producer_status.compare_exchange(
            RequestStatsProducerStatus::Pending as u8,
            RequestStatsProducerStatus::Unavailable as u8,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn request_stats_producer_status(&self) -> RequestStatsProducerStatus {
        match self
            .inner
            .request_stats_producer_status
            .load(Ordering::Relaxed)
        {
            status if status == RequestStatsProducerStatus::Available as u8 => {
                RequestStatsProducerStatus::Available
            }
            status if status == RequestStatsProducerStatus::Unavailable as u8 => {
                RequestStatsProducerStatus::Unavailable
            }
            _ => RequestStatsProducerStatus::Pending,
        }
    }

    /// Validate, serialize, and publish one cumulative request update.
    ///
    /// Publishing is synchronous and non-blocking. If no stream client is
    /// connected, validation still runs but serialization is skipped.
    pub fn publish_request_stats(
        &self,
        update: RequestStatsUpdate<'_>,
    ) -> Result<(), RequestStatsPublishError> {
        let request_id = update.request_id.trim();
        if request_id.is_empty() {
            return Err(RequestStatsPublishError::EmptyRequestId);
        }
        let model = update.model.trim();
        if model.is_empty() {
            return Err(RequestStatsPublishError::EmptyModel);
        }
        if update.tokens_processed.is_none()
            && update.tokens_generated.is_none()
            && !update.finished
        {
            return Err(RequestStatsPublishError::MissingCounters);
        }

        if self.inner.request_stats_tx.receiver_count() == 0 {
            return Ok(());
        }

        let event = RequestStatsEvent {
            v: 1,
            event_type: "stats",
            request_id,
            model,
            tokens_processed: update.tokens_processed,
            tokens_generated: update.tokens_generated,
            finished: update.finished,
        };
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');

        // A receiver can disconnect between receiver_count() and send(). That
        // is normal telemetry loss, not an inference-path failure.
        let _ = self.inner.request_stats_tx.send(Bytes::from(line));
        Ok(())
    }

    pub(crate) fn subscribe_request_stats(&self) -> broadcast::Receiver<Bytes> {
        self.inner.request_stats_tx.subscribe()
    }

    /// Replace the latest reliable block observation for one model/rank.
    pub fn update_kv_snapshot(
        &self,
        snapshot: KvCacheSnapshot<'_>,
    ) -> Result<(), KvCacheSnapshotError> {
        let model = snapshot.model.trim();
        if model.is_empty() {
            return Err(KvCacheSnapshotError::EmptyModel);
        }
        if snapshot.expected_dp_ranks == 0 {
            return Err(KvCacheSnapshotError::ZeroExpectedDpRanks);
        }
        if snapshot.block_size_tokens == 0 {
            return Err(KvCacheSnapshotError::ZeroBlockSize);
        }
        if snapshot.total_blocks == 0 {
            return Err(KvCacheSnapshotError::ZeroTotalBlocks);
        }
        if snapshot.used_blocks > snapshot.total_blocks {
            return Err(KvCacheSnapshotError::UsedBlocksExceedTotal {
                used_blocks: snapshot.used_blocks,
                total_blocks: snapshot.total_blocks,
            });
        }
        if snapshot.dp_rank >= snapshot.expected_dp_ranks {
            return Err(KvCacheSnapshotError::UnexpectedDpRank {
                model: model.to_owned(),
                dp_rank: snapshot.dp_rank,
                expected: snapshot.expected_dp_ranks,
            });
        }

        let rank = RankKvSnapshot {
            used_blocks: snapshot.used_blocks,
            total_blocks: snapshot.total_blocks,
        };
        let mut models = self.inner.kv_models.lock();
        if let Some(state) = models.get_mut(model) {
            if state.expected_dp_ranks != snapshot.expected_dp_ranks {
                return Err(KvCacheSnapshotError::ExpectedDpRanksChanged {
                    model: model.to_owned(),
                    previous: state.expected_dp_ranks,
                    observed: snapshot.expected_dp_ranks,
                });
            }
            if state.block_size_tokens != snapshot.block_size_tokens {
                return Err(KvCacheSnapshotError::BlockSizeChanged {
                    model: model.to_owned(),
                    previous: state.block_size_tokens,
                    observed: snapshot.block_size_tokens,
                });
            }
            state.ranks.insert(snapshot.dp_rank, rank);
        } else {
            models.insert(
                model.to_owned(),
                ModelKvState {
                    expected_dp_ranks: snapshot.expected_dp_ranks,
                    block_size_tokens: snapshot.block_size_tokens,
                    ranks: HashMap::from([(snapshot.dp_rank, rank)]),
                },
            );
        }
        Ok(())
    }

    /// Derive Pylon's token response from the latest complete single-model state.
    pub fn kv_cache_stats(&self) -> Result<KvCacheStats, KvCacheStatsUnavailable> {
        let models = self.inner.kv_models.lock();
        if models.len() > 1 {
            let mut model_names: Vec<_> = models.keys().cloned().collect();
            model_names.sort_unstable();
            return Err(KvCacheStatsUnavailable::AmbiguousModels {
                models: model_names,
            });
        }

        let (model, state) = models
            .iter()
            .next()
            .ok_or(KvCacheStatsUnavailable::NoSnapshots)?;
        if state.ranks.len() != state.expected_dp_ranks as usize {
            return Err(KvCacheStatsUnavailable::IncompleteDpRanks {
                model: model.clone(),
                observed: state.ranks.len(),
                expected: state.expected_dp_ranks,
            });
        }

        let mut total_blocks = 0_u64;
        let mut used_blocks = 0_u64;
        for snapshot in state.ranks.values() {
            total_blocks = total_blocks
                .checked_add(snapshot.total_blocks)
                .ok_or_else(|| KvCacheStatsUnavailable::TokenCountOverflow {
                    model: model.clone(),
                })?;
            used_blocks = used_blocks
                .checked_add(snapshot.used_blocks)
                .ok_or_else(|| KvCacheStatsUnavailable::TokenCountOverflow {
                    model: model.clone(),
                })?;
        }
        let block_size = u64::from(state.block_size_tokens);
        let capacity_tokens = total_blocks.checked_mul(block_size).ok_or_else(|| {
            KvCacheStatsUnavailable::TokenCountOverflow {
                model: model.clone(),
            }
        })?;
        let used_tokens = used_blocks.checked_mul(block_size).ok_or_else(|| {
            KvCacheStatsUnavailable::TokenCountOverflow {
                model: model.clone(),
            }
        })?;

        Ok(KvCacheStats {
            model: model.clone(),
            kv_cache_capacity_tokens: capacity_tokens,
            kv_cache_used_tokens: used_tokens,
            kv_cache_free_tokens: capacity_tokens - used_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast::error::RecvError;

    fn request_update(tokens_generated: u64) -> RequestStatsUpdate<'static> {
        RequestStatsUpdate {
            request_id: "req-1",
            model: "model-a",
            tokens_processed: None,
            tokens_generated: Some(tokens_generated),
            finished: false,
        }
    }

    fn kv_snapshot(model: &str, rank: u32, expected_dp_ranks: u32) -> KvCacheSnapshot<'_> {
        KvCacheSnapshot {
            model,
            dp_rank: rank,
            expected_dp_ranks,
            used_blocks: 4,
            total_blocks: 10,
            block_size_tokens: 16,
        }
    }

    #[tokio::test]
    async fn pylon_stats_serializes_one_ndjson_line() {
        let stats = PylonStats::with_capacity(4);
        let mut rx = stats.subscribe_request_stats();
        stats
            .publish_request_stats(RequestStatsUpdate {
                request_id: " req-1 ",
                model: " model-a ",
                tokens_processed: Some(128),
                tokens_generated: Some(17),
                finished: false,
            })
            .unwrap();

        let line = rx.recv().await.unwrap();
        assert_eq!(line.last(), Some(&b'\n'));
        let event: serde_json::Value = serde_json::from_slice(&line).unwrap();
        assert_eq!(
            event,
            serde_json::json!({
                "v": 1,
                "type": "stats",
                "request_id": "req-1",
                "model": "model-a",
                "tokens_processed": 128,
                "tokens_generated": 17,
                "finished": false
            })
        );
    }

    #[test]
    fn pylon_stats_rejects_malformed_events() {
        let stats = PylonStats::default();
        assert!(matches!(
            stats.publish_request_stats(RequestStatsUpdate {
                request_id: " ",
                ..request_update(1)
            }),
            Err(RequestStatsPublishError::EmptyRequestId)
        ));
        assert!(matches!(
            stats.publish_request_stats(RequestStatsUpdate {
                model: "",
                ..request_update(1)
            }),
            Err(RequestStatsPublishError::EmptyModel)
        ));
        assert!(matches!(
            stats.publish_request_stats(RequestStatsUpdate {
                tokens_generated: None,
                ..request_update(1)
            }),
            Err(RequestStatsPublishError::MissingCounters)
        ));
        assert!(
            stats
                .publish_request_stats(RequestStatsUpdate {
                    tokens_generated: None,
                    finished: true,
                    ..request_update(1)
                })
                .is_ok()
        );
    }

    #[tokio::test]
    async fn pylon_stats_slow_subscriber_loses_old_events_with_fixed_capacity() {
        let stats = PylonStats::with_capacity(2);
        let mut rx = stats.subscribe_request_stats();
        for count in 1..=3 {
            stats.publish_request_stats(request_update(count)).unwrap();
        }

        assert!(matches!(rx.recv().await, Err(RecvError::Lagged(1))));
        let second: serde_json::Value = serde_json::from_slice(&rx.recv().await.unwrap()).unwrap();
        let third: serde_json::Value = serde_json::from_slice(&rx.recv().await.unwrap()).unwrap();
        assert_eq!(second["tokens_generated"], 2);
        assert_eq!(third["tokens_generated"], 3);
    }

    #[test]
    fn pylon_stats_disconnected_subscriber_does_not_retain_or_reject_events() {
        let stats = PylonStats::with_capacity(2);
        for count in 1..=10 {
            stats.publish_request_stats(request_update(count)).unwrap();
        }

        let mut rx = stats.subscribe_request_stats();
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn request_stats_producer_registration_preserves_late_attachment() {
        let stats = PylonStats::default();
        assert_eq!(
            stats.request_stats_producer_status(),
            RequestStatsProducerStatus::Pending
        );

        stats.complete_request_stats_producer_registration();
        assert_eq!(
            stats.request_stats_producer_status(),
            RequestStatsProducerStatus::Unavailable
        );

        stats.mark_request_stats_producer_available();
        stats.complete_request_stats_producer_registration();
        assert_eq!(
            stats.request_stats_producer_status(),
            RequestStatsProducerStatus::Available
        );
    }

    #[test]
    fn kv_cache_stats_is_not_ready_without_observations() {
        assert_eq!(
            PylonStats::default().kv_cache_stats(),
            Err(KvCacheStatsUnavailable::NoSnapshots)
        );
    }

    #[test]
    fn kv_cache_stats_waits_for_every_local_dp_rank_and_aggregates_tokens() {
        let stats = PylonStats::default();
        stats
            .update_kv_snapshot(kv_snapshot("model-a", 0, 2))
            .unwrap();
        assert_eq!(
            stats.kv_cache_stats(),
            Err(KvCacheStatsUnavailable::IncompleteDpRanks {
                model: "model-a".to_string(),
                observed: 1,
                expected: 2,
            })
        );

        let mut rank_one = kv_snapshot("model-a", 1, 2);
        rank_one.used_blocks = 6;
        rank_one.total_blocks = 20;
        stats.update_kv_snapshot(rank_one).unwrap();

        let mut mismatched_block_size = rank_one;
        mismatched_block_size.block_size_tokens = 32;
        assert_eq!(
            stats.update_kv_snapshot(mismatched_block_size),
            Err(KvCacheSnapshotError::BlockSizeChanged {
                model: "model-a".to_string(),
                previous: 16,
                observed: 32,
            })
        );

        assert_eq!(
            stats.kv_cache_stats().unwrap(),
            KvCacheStats {
                model: "model-a".to_string(),
                kv_cache_capacity_tokens: 480,
                kv_cache_used_tokens: 160,
                kv_cache_free_tokens: 320,
            }
        );
    }

    #[test]
    fn kv_cache_stats_fails_closed_for_multiple_models() {
        let stats = PylonStats::default();
        stats
            .update_kv_snapshot(kv_snapshot("model-b", 0, 1))
            .unwrap();
        stats
            .update_kv_snapshot(kv_snapshot("model-a", 0, 1))
            .unwrap();

        assert_eq!(
            stats.kv_cache_stats(),
            Err(KvCacheStatsUnavailable::AmbiguousModels {
                models: vec!["model-a".to_string(), "model-b".to_string()],
            })
        );
    }

    #[test]
    fn invalid_kv_snapshot_does_not_create_available_state() {
        let stats = PylonStats::default();
        let mut snapshot = kv_snapshot("model-a", 0, 1);
        snapshot.used_blocks = 11;

        assert_eq!(
            stats.update_kv_snapshot(snapshot),
            Err(KvCacheSnapshotError::UsedBlocksExceedTotal {
                used_blocks: 11,
                total_blocks: 10,
            })
        );
        assert_eq!(
            stats.kv_cache_stats(),
            Err(KvCacheStatsUnavailable::NoSnapshots)
        );
    }

    #[test]
    fn out_of_range_dp_rank_does_not_create_available_state() {
        let stats = PylonStats::default();

        assert_eq!(
            stats.update_kv_snapshot(kv_snapshot("model-a", 1, 1)),
            Err(KvCacheSnapshotError::UnexpectedDpRank {
                model: "model-a".to_string(),
                dp_rank: 1,
                expected: 1,
            })
        );
        assert_eq!(
            stats.kv_cache_stats(),
            Err(KvCacheStatsUnavailable::NoSnapshots)
        );
    }

    #[test]
    fn kv_cache_stats_fails_closed_on_token_count_overflow() {
        let stats = PylonStats::default();
        stats
            .update_kv_snapshot(KvCacheSnapshot {
                model: "model-a",
                dp_rank: 0,
                expected_dp_ranks: 1,
                used_blocks: 0,
                total_blocks: u64::MAX,
                block_size_tokens: 2,
            })
            .unwrap();

        assert_eq!(
            stats.kv_cache_stats(),
            Err(KvCacheStatsUnavailable::TokenCountOverflow {
                model: "model-a".to_string(),
            })
        );
    }
}
