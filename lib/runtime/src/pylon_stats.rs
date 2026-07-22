// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded request-stats fanout and latest observed KV state for Pylon routes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
pub struct KvCacheSnapshot {
    pub dp_rank: u32,
    pub used_blocks: u64,
    pub total_blocks: u64,
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
pub enum KvCacheUpdateError {
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
    #[error(
        "KV cache is already configured for model {configured_model:?} with {configured_dp_ranks} local DP ranks and {configured_block_size} tokens per block"
    )]
    ConflictingConfiguration {
        configured_model: String,
        configured_dp_ranks: u32,
        configured_block_size: u32,
    },
    #[error("KV cache must be configured before snapshots are published")]
    NotConfigured,
    #[error("rank {dp_rank} is outside the configured {expected} local DP ranks")]
    UnexpectedDpRank { dp_rank: u32, expected: u32 },
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
    #[error("KV-cache token counts overflowed for model {model:?}")]
    TokenCountOverflow { model: String },
}

impl KvCacheStatsUnavailable {
    pub(crate) fn response_code(&self) -> &'static str {
        match self {
            Self::NoSnapshots | Self::IncompleteDpRanks { .. } => "kv_cache_stats_not_ready",
            Self::TokenCountOverflow { .. } => "kv_cache_stats_unavailable",
        }
    }
}

#[derive(Clone)]
pub struct PylonStats {
    inner: Arc<PylonStatsInner>,
}

struct PylonStatsInner {
    request_stats_producer_available: AtomicBool,
    request_stats_tx: broadcast::Sender<Bytes>,
    kv_state: Mutex<Option<ModelKvState>>,
}

struct ModelKvState {
    model: String,
    block_size_tokens: u32,
    ranks: Vec<Option<RankKvSnapshot>>,
}

#[derive(Clone, Copy)]
struct RankKvSnapshot {
    used_blocks: u64,
    total_blocks: u64,
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
                request_stats_producer_available: AtomicBool::new(false),
                request_stats_tx,
                kv_state: Mutex::new(None),
            }),
        }
    }

    /// Mark request stats available before the worker enters discovery.
    pub fn mark_request_stats_producer_available(&self) {
        self.inner
            .request_stats_producer_available
            .store(true, Ordering::Relaxed);
    }

    pub(crate) fn request_stats_producer_available(&self) -> bool {
        self.inner
            .request_stats_producer_available
            .load(Ordering::Relaxed)
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

    /// Configure immutable KV-cache geometry for this worker.
    pub fn configure_kv_cache(
        &self,
        model: &str,
        expected_dp_ranks: u32,
        block_size_tokens: u32,
    ) -> Result<(), KvCacheUpdateError> {
        let model = model.trim();
        if model.is_empty() {
            return Err(KvCacheUpdateError::EmptyModel);
        }
        if expected_dp_ranks == 0 {
            return Err(KvCacheUpdateError::ZeroExpectedDpRanks);
        }
        if block_size_tokens == 0 {
            return Err(KvCacheUpdateError::ZeroBlockSize);
        }

        let mut state = self.inner.kv_state.lock();
        if let Some(configured) = state.as_ref() {
            if configured.model == model
                && configured.ranks.len() == expected_dp_ranks as usize
                && configured.block_size_tokens == block_size_tokens
            {
                return Ok(());
            }
            return Err(KvCacheUpdateError::ConflictingConfiguration {
                configured_model: configured.model.clone(),
                configured_dp_ranks: configured.ranks.len() as u32,
                configured_block_size: configured.block_size_tokens,
            });
        }

        *state = Some(ModelKvState {
            model: model.to_owned(),
            block_size_tokens,
            ranks: vec![None; expected_dp_ranks as usize],
        });
        Ok(())
    }

    /// Replace the latest reliable block observation for one local DP rank.
    pub fn update_kv_snapshot(&self, snapshot: KvCacheSnapshot) -> Result<(), KvCacheUpdateError> {
        if snapshot.total_blocks == 0 {
            return Err(KvCacheUpdateError::ZeroTotalBlocks);
        }
        if snapshot.used_blocks > snapshot.total_blocks {
            return Err(KvCacheUpdateError::UsedBlocksExceedTotal {
                used_blocks: snapshot.used_blocks,
                total_blocks: snapshot.total_blocks,
            });
        }

        let rank = RankKvSnapshot {
            used_blocks: snapshot.used_blocks,
            total_blocks: snapshot.total_blocks,
        };
        let mut state = self.inner.kv_state.lock();
        let state = state.as_mut().ok_or(KvCacheUpdateError::NotConfigured)?;
        let expected = state.ranks.len() as u32;
        let slot = state.ranks.get_mut(snapshot.dp_rank as usize).ok_or(
            KvCacheUpdateError::UnexpectedDpRank {
                dp_rank: snapshot.dp_rank,
                expected,
            },
        )?;
        *slot = Some(rank);
        Ok(())
    }

    /// Derive Pylon's token response from the latest complete state.
    pub fn kv_cache_stats(&self) -> Result<KvCacheStats, KvCacheStatsUnavailable> {
        let state = self.inner.kv_state.lock();
        let state = state.as_ref().ok_or(KvCacheStatsUnavailable::NoSnapshots)?;
        let observed = state.ranks.iter().flatten().count();
        if observed == 0 {
            return Err(KvCacheStatsUnavailable::NoSnapshots);
        }
        if observed != state.ranks.len() {
            return Err(KvCacheStatsUnavailable::IncompleteDpRanks {
                model: state.model.clone(),
                observed,
                expected: state.ranks.len() as u32,
            });
        }

        let mut total_blocks = 0_u64;
        let mut used_blocks = 0_u64;
        for snapshot in state.ranks.iter().flatten() {
            total_blocks = total_blocks
                .checked_add(snapshot.total_blocks)
                .ok_or_else(|| KvCacheStatsUnavailable::TokenCountOverflow {
                    model: state.model.clone(),
                })?;
            used_blocks = used_blocks
                .checked_add(snapshot.used_blocks)
                .ok_or_else(|| KvCacheStatsUnavailable::TokenCountOverflow {
                    model: state.model.clone(),
                })?;
        }
        let block_size = u64::from(state.block_size_tokens);
        let capacity_tokens = total_blocks.checked_mul(block_size).ok_or_else(|| {
            KvCacheStatsUnavailable::TokenCountOverflow {
                model: state.model.clone(),
            }
        })?;
        let used_tokens = used_blocks.checked_mul(block_size).ok_or_else(|| {
            KvCacheStatsUnavailable::TokenCountOverflow {
                model: state.model.clone(),
            }
        })?;

        Ok(KvCacheStats {
            model: state.model.clone(),
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

    fn configured_stats(expected_dp_ranks: u32) -> PylonStats {
        let stats = PylonStats::default();
        stats
            .configure_kv_cache("model-a", expected_dp_ranks, 16)
            .unwrap();
        stats
    }

    fn kv_snapshot(rank: u32) -> KvCacheSnapshot {
        KvCacheSnapshot {
            dp_rank: rank,
            used_blocks: 4,
            total_blocks: 10,
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
    fn request_stats_producer_is_available_after_attachment() {
        let stats = PylonStats::default();
        assert!(!stats.request_stats_producer_available());

        stats.mark_request_stats_producer_available();
        assert!(stats.request_stats_producer_available());
    }

    #[test]
    fn kv_cache_stats_is_not_ready_without_observations() {
        assert_eq!(
            PylonStats::default().kv_cache_stats(),
            Err(KvCacheStatsUnavailable::NoSnapshots)
        );
    }

    #[test]
    fn kv_cache_rejects_invalid_configuration_and_unconfigured_updates() {
        let stats = PylonStats::default();
        assert_eq!(
            stats.configure_kv_cache(" ", 1, 16),
            Err(KvCacheUpdateError::EmptyModel)
        );
        assert_eq!(
            stats.configure_kv_cache("model-a", 0, 16),
            Err(KvCacheUpdateError::ZeroExpectedDpRanks)
        );
        assert_eq!(
            stats.configure_kv_cache("model-a", 1, 0),
            Err(KvCacheUpdateError::ZeroBlockSize)
        );
        assert_eq!(
            stats.update_kv_snapshot(kv_snapshot(0)),
            Err(KvCacheUpdateError::NotConfigured)
        );
    }

    #[test]
    fn kv_cache_stats_waits_for_every_local_dp_rank_and_aggregates_tokens() {
        let stats = configured_stats(2);
        stats.update_kv_snapshot(kv_snapshot(0)).unwrap();
        assert_eq!(
            stats.kv_cache_stats(),
            Err(KvCacheStatsUnavailable::IncompleteDpRanks {
                model: "model-a".to_string(),
                observed: 1,
                expected: 2,
            })
        );

        let mut rank_one = kv_snapshot(1);
        rank_one.used_blocks = 6;
        rank_one.total_blocks = 20;
        stats.update_kv_snapshot(rank_one).unwrap();

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
    fn kv_cache_configuration_is_idempotent_but_cannot_change() {
        let stats = PylonStats::default();
        stats.configure_kv_cache("model-a", 2, 16).unwrap();
        stats.configure_kv_cache("model-a", 2, 16).unwrap();

        assert_eq!(
            stats.configure_kv_cache("model-b", 1, 32),
            Err(KvCacheUpdateError::ConflictingConfiguration {
                configured_model: "model-a".to_string(),
                configured_dp_ranks: 2,
                configured_block_size: 16,
            })
        );
    }

    #[test]
    fn invalid_kv_snapshot_does_not_create_available_state() {
        let stats = configured_stats(1);
        let mut snapshot = kv_snapshot(0);
        snapshot.used_blocks = 11;

        assert_eq!(
            stats.update_kv_snapshot(snapshot),
            Err(KvCacheUpdateError::UsedBlocksExceedTotal {
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
        let stats = configured_stats(1);

        assert_eq!(
            stats.update_kv_snapshot(kv_snapshot(1)),
            Err(KvCacheUpdateError::UnexpectedDpRank {
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
        stats.configure_kv_cache("model-a", 1, 2).unwrap();
        stats
            .update_kv_snapshot(KvCacheSnapshot {
                dp_rank: 0,
                used_blocks: 0,
                total_blocks: u64::MAX,
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
