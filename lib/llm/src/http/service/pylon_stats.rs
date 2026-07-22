// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned request counters consumed by Pylon.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, header};
use axum::response::Response;
use axum::routing::get;
use dynamo_runtime::pipeline::Context;
use serde::Serialize;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use super::{RouteDoc, service_v2};

const STATS_PATH: &str = "/pylon/v1/stats/stream";
const REQUEST_ID_HEADER: &str = "x-request-id";
const MODEL_HEADER: &str = "x-model";
const INPUT_TOKENS_HEADER: &str = "x-input-tokens";
const IDENTITY_CONTEXT_KEY: &str = "pylon_request_identity";
const DEFAULT_CHANNEL_CAPACITY: usize = 1024;
const PING_INTERVAL: Duration = Duration::from_secs(15);
const PING_LINE: &[u8] = b"{\"v\":1,\"type\":\"ping\"}\n";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PylonRequestIdentity {
    request_id: String,
    model: String,
    input_tokens: u64,
}

impl PylonRequestIdentity {
    fn from_headers(headers: &HeaderMap) -> Option<Self> {
        Some(Self {
            request_id: nonempty_header(headers, REQUEST_ID_HEADER)?,
            model: nonempty_header(headers, MODEL_HEADER)?,
            input_tokens: nonempty_header(headers, INPUT_TOKENS_HEADER)?
                .parse()
                .ok()?,
        })
    }
}

fn nonempty_header(headers: &HeaderMap, name: &'static str) -> Option<String> {
    let value = headers.get(name)?.to_str().ok()?.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

pub(super) fn attach_request_identity<T: Send + Sync + 'static>(
    request: &mut Context<T>,
    headers: &HeaderMap,
) {
    if let Some(identity) = PylonRequestIdentity::from_headers(headers) {
        request.insert(IDENTITY_CONTEXT_KEY, identity);
    }
}

pub(super) fn request_identity<T: Send + Sync + 'static>(
    request: &Context<T>,
) -> Option<PylonRequestIdentity> {
    request
        .get::<PylonRequestIdentity>(IDENTITY_CONTEXT_KEY)
        .ok()
        .map(|identity| identity.as_ref().clone())
}

#[derive(Clone)]
pub(super) struct PylonStats {
    tx: broadcast::Sender<Bytes>,
}

impl Default for PylonStats {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }
}

impl PylonStats {
    fn with_capacity(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    fn publish(&self, update: RequestStatsUpdate<'_>) -> serde_json::Result<()> {
        if self.tx.receiver_count() == 0 {
            return Ok(());
        }

        let event = RequestStatsEvent {
            v: 1,
            event_type: "stats",
            request_id: update.request_id,
            model: update.model,
            tokens_processed: update.tokens_processed,
            tokens_generated: update.tokens_generated,
            finished: update.finished,
        };
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');

        // Disconnecting between receiver_count and send is expected telemetry loss.
        let _ = self.tx.send(Bytes::from(line));
        Ok(())
    }

    fn subscribe(&self) -> broadcast::Receiver<Bytes> {
        self.tx.subscribe()
    }
}

struct RequestStatsUpdate<'a> {
    request_id: &'a str,
    model: &'a str,
    tokens_processed: Option<u64>,
    tokens_generated: Option<u64>,
    finished: bool,
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

pub(super) struct PylonRequestStats {
    stats: PylonStats,
    identity: PylonRequestIdentity,
    tokens_processed: Option<u64>,
    tokens_generated: Option<u64>,
}

impl PylonRequestStats {
    pub(super) fn new(stats: PylonStats, identity: PylonRequestIdentity) -> Self {
        Self {
            stats,
            identity,
            tokens_processed: None,
            tokens_generated: None,
        }
    }

    pub(super) fn observe(&mut self, generated_tokens: usize) {
        let generated_tokens = u64::try_from(generated_tokens).unwrap_or(u64::MAX);

        let tokens_processed = if self.tokens_processed.is_none() {
            self.tokens_processed = Some(self.identity.input_tokens);
            self.tokens_processed
        } else {
            None
        };
        let tokens_generated = if generated_tokens > 0 {
            let previous = self.tokens_generated.unwrap_or_default();
            let total = previous.saturating_add(generated_tokens);
            (total > previous).then(|| {
                self.tokens_generated = Some(total);
                total
            })
        } else {
            None
        };

        if tokens_processed.is_some() || tokens_generated.is_some() {
            self.publish(tokens_processed, tokens_generated, false);
        }
    }

    fn publish(
        &self,
        tokens_processed: Option<u64>,
        tokens_generated: Option<u64>,
        finished: bool,
    ) {
        let result = self.stats.publish(RequestStatsUpdate {
            request_id: &self.identity.request_id,
            model: &self.identity.model,
            tokens_processed,
            tokens_generated,
            finished,
        });
        if let Err(error) = result {
            tracing::debug!(
                request_id = %self.identity.request_id,
                %error,
                "failed to serialize Pylon stats"
            );
        }
    }
}

impl Drop for PylonRequestStats {
    fn drop(&mut self) {
        self.publish(self.tokens_processed, self.tokens_generated, true);
    }
}

pub(super) fn router(state: Arc<service_v2::State>) -> (Vec<RouteDoc>, Router) {
    let docs = vec![RouteDoc::new(Method::GET, STATS_PATH)];
    let router = Router::new()
        .route(STATS_PATH, get(stats_stream_handler))
        .with_state(state);
    (docs, router)
}

async fn stats_stream_handler(State(state): State<Arc<service_v2::State>>) -> Response {
    stats_stream_response(state.pylon_stats().clone(), state.cancel_token().clone())
}

fn stats_stream_response(stats: PylonStats, shutdown: CancellationToken) -> Response {
    let mut receiver = stats.subscribe();
    let stream = async_stream::stream! {
        let mut ping = tokio::time::interval_at(
            tokio::time::Instant::now() + PING_INTERVAL,
            PING_INTERVAL,
        );
        ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        yield Ok::<Bytes, Infallible>(Bytes::from_static(PING_LINE));

        loop {
            tokio::select! {
                event = receiver.recv() => match event {
                    Ok(line) => yield Ok(line),
                    Err(broadcast::error::RecvError::Lagged(dropped)) => {
                        tracing::debug!(dropped, "Pylon stats stream subscriber lagged");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                _ = ping.tick() => yield Ok(Bytes::from_static(PING_LINE)),
                _ = shutdown.cancelled() => break,
            }
        }
    };

    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use serde_json::Value;
    use tokio::sync::broadcast::error::{RecvError, TryRecvError};

    use super::*;

    fn headers(request_id: &'static str, model: &'static str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(REQUEST_ID_HEADER, HeaderValue::from_static(request_id));
        headers.insert(MODEL_HEADER, HeaderValue::from_static(model));
        headers.insert(INPUT_TOKENS_HEADER, HeaderValue::from_static("3"));
        headers
    }

    fn json(line: &Bytes) -> Value {
        serde_json::from_slice(line).unwrap()
    }

    #[test]
    fn identity_uses_pylon_headers_without_changing_context_id() {
        let mut context =
            Context::with_id_and_metadata((), "dynamo-id".to_string(), Default::default());
        attach_request_identity(&mut context, &headers("pylon-id", "external-model"));

        assert_eq!(context.id(), "dynamo-id");
        assert_eq!(
            request_identity(&context),
            Some(PylonRequestIdentity {
                request_id: "pylon-id".to_string(),
                model: "external-model".to_string(),
                input_tokens: 3,
            })
        );
    }

    #[test]
    fn identity_survives_context_parts_and_map() {
        let mut context =
            Context::with_id_and_metadata((), "dynamo-id".to_string(), Default::default());
        attach_request_identity(&mut context, &headers("pylon-id", "external-model"));

        let (_, context) = context.into_parts();
        let mapped = context.map(|_| "mapped");

        assert_eq!(
            request_identity(&mapped),
            Some(PylonRequestIdentity {
                request_id: "pylon-id".to_string(),
                model: "external-model".to_string(),
                input_tokens: 3,
            })
        );
    }

    #[test]
    fn identity_requires_all_valid_tunnel_headers() {
        let mut missing_model = HeaderMap::new();
        missing_model.insert(REQUEST_ID_HEADER, HeaderValue::from_static("pylon-id"));
        missing_model.insert(INPUT_TOKENS_HEADER, HeaderValue::from_static("3"));
        assert_eq!(PylonRequestIdentity::from_headers(&missing_model), None);
        assert_eq!(
            PylonRequestIdentity::from_headers(&headers("pylon-id", "   ")),
            None
        );
        let mut invalid_tokens = headers("pylon-id", "model");
        invalid_tokens.insert(INPUT_TOKENS_HEADER, HeaderValue::from_static("three"));
        assert_eq!(PylonRequestIdentity::from_headers(&invalid_tokens), None);
    }

    #[tokio::test]
    async fn request_stats_are_cumulative_and_finish_once() {
        let stats = PylonStats::with_capacity(8);
        let mut receiver = stats.subscribe();
        {
            let mut request = PylonRequestStats::new(
                stats.clone(),
                PylonRequestIdentity {
                    request_id: "pylon-id".to_string(),
                    model: "external-model".to_string(),
                    input_tokens: 3,
                },
            );
            request.observe(0);
            request.observe(2);
            request.observe(3);
            request.observe(0);
        }

        let processed = json(&receiver.recv().await.unwrap());
        assert_eq!(processed["request_id"], "pylon-id");
        assert_eq!(processed["model"], "external-model");
        assert_eq!(processed["tokens_processed"], 3);
        assert!(processed.get("tokens_generated").is_none());
        assert_eq!(processed["finished"], false);

        let generated = json(&receiver.recv().await.unwrap());
        assert_eq!(generated["tokens_generated"], 2);
        assert!(generated.get("tokens_processed").is_none());
        assert_eq!(generated["finished"], false);

        let generated = json(&receiver.recv().await.unwrap());
        assert_eq!(generated["tokens_generated"], 5);
        assert_eq!(generated["finished"], false);

        let finished = json(&receiver.recv().await.unwrap());
        assert_eq!(finished["tokens_processed"], 3);
        assert_eq!(finished["tokens_generated"], 5);
        assert_eq!(finished["finished"], true);
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn counter_saturation_still_emits_terminal_event() {
        let stats = PylonStats::with_capacity(4);
        let mut receiver = stats.subscribe();
        {
            let mut request = PylonRequestStats::new(
                stats.clone(),
                PylonRequestIdentity {
                    request_id: "pylon-id".to_string(),
                    model: "external-model".to_string(),
                    input_tokens: 3,
                },
            );
            request.tokens_processed = Some(3);
            request.tokens_generated = Some(u64::MAX - 1);
            request.observe(usize::MAX);
        }

        let saturated = json(&receiver.recv().await.unwrap());
        assert_eq!(saturated["tokens_generated"], u64::MAX);
        assert_eq!(saturated["finished"], false);
        let finished = json(&receiver.recv().await.unwrap());
        assert_eq!(finished["tokens_generated"], u64::MAX);
        assert_eq!(finished["finished"], true);
    }

    #[tokio::test]
    async fn bounded_stream_drops_old_cumulative_events() {
        let stats = PylonStats::with_capacity(2);
        let mut receiver = stats.subscribe();
        for generated in [2, 5, 9] {
            stats
                .publish(RequestStatsUpdate {
                    request_id: "request",
                    model: "model",
                    tokens_processed: None,
                    tokens_generated: Some(generated),
                    finished: false,
                })
                .unwrap();
        }

        assert!(matches!(receiver.recv().await, Err(RecvError::Lagged(1))));
        assert_eq!(json(&receiver.recv().await.unwrap())["tokens_generated"], 5);
        assert_eq!(json(&receiver.recv().await.unwrap())["tokens_generated"], 9);
    }

    #[tokio::test]
    async fn stream_is_ready_immediately_and_closes_on_shutdown() {
        let shutdown = CancellationToken::new();
        let response = stats_stream_response(PylonStats::default(), shutdown.clone());
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/x-ndjson"
        );

        let mut body = response.into_body().into_data_stream();
        assert_eq!(
            body.next().await.unwrap().unwrap(),
            Bytes::from_static(PING_LINE)
        );
        shutdown.cancel();
        assert!(body.next().await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn idle_stream_emits_periodic_pings() {
        let stats = PylonStats::default();
        let response = stats_stream_response(stats.clone(), CancellationToken::new());
        let mut body = response.into_body().into_data_stream();
        assert_eq!(
            body.next().await.unwrap().unwrap(),
            Bytes::from_static(PING_LINE)
        );

        let next = tokio::time::timeout(Duration::from_secs(16), body.next())
            .await
            .expect("periodic ping should arrive")
            .unwrap()
            .unwrap();
        assert_eq!(next, Bytes::from_static(PING_LINE));
    }

    #[test]
    fn frontend_registers_stats_without_the_unused_kv_route() {
        let service = service_v2::HttpService::builder().build().unwrap();
        let routes = service
            .route_docs()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();

        assert!(routes.contains(&format!("GET {STATS_PATH}")));
        assert!(!routes.contains(&"GET /kv-cache/stats".to_string()));
    }

    #[tokio::test]
    async fn frontend_serves_models_and_stats_on_one_listener_before_readiness() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let service = service_v2::HttpService::builder()
            .port(port)
            .build()
            .unwrap();
        let shutdown = CancellationToken::new();
        let handle = service
            .spawn_with_listener(shutdown.clone(), listener)
            .await;

        let client = reqwest::Client::new();
        let models = client
            .get(format!("http://127.0.0.1:{port}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(models.status(), reqwest::StatusCode::OK);

        let mut stats = client
            .get(format!("http://127.0.0.1:{port}{STATS_PATH}"))
            .send()
            .await
            .unwrap();
        assert_eq!(stats.status(), reqwest::StatusCode::OK);
        assert_eq!(
            stats.headers().get(reqwest::header::CONTENT_TYPE).unwrap(),
            "application/x-ndjson"
        );
        assert_eq!(stats.chunk().await.unwrap().unwrap(), PING_LINE);

        drop(stats);
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("frontend should stop after cancellation")
            .unwrap()
            .unwrap();
    }
}
