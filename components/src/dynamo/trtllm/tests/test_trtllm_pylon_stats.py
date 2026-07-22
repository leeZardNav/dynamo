# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRT-LLM Pylon stats adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from dynamo.trtllm.constants import DisaggregationMode

try:
    from dynamo.trtllm.publisher import Publisher
    from dynamo.trtllm.request_handlers.handler_base import HandlerBase
except ImportError as e:
    pytest.skip(f"tensorrt_llm backend not available: {e}", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _Context:
    def __init__(self, request_id: str = "req-1") -> None:
        self._request_id = request_id

    def id(self) -> str:
        return self._request_id

    def trace_headers(self) -> None:
        return None


class _PylonPublisher:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.kv_snapshots: list[tuple[Any, ...]] = []

    def publish_stats_event(
        self,
        request_id: str,
        model: str,
        tokens_processed: int | None = None,
        tokens_generated: int | None = None,
        finished: bool = False,
    ) -> None:
        self.events.append(
            {
                "request_id": request_id,
                "model": model,
                "tokens_processed": tokens_processed,
                "tokens_generated": tokens_generated,
                "finished": finished,
            }
        )

    def update_kv_snapshot(self, *args: Any) -> None:
        self.kv_snapshots.append(args)


class _Handler(HandlerBase):
    async def generate(
        self, request: dict, context: _Context
    ) -> AsyncGenerator[dict, None]:
        if False:
            yield {}


def _result(
    outputs: list[tuple[int, list[int], str | None]], finished: bool
) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                index=index,
                token_ids=tokens,
                finish_reason=reason,
                stop_reason=None,
                request_perf_metrics=None,
            )
            for index, tokens, reason in outputs
        ],
        finished=finished,
    )


def _make_handler(
    generate_async: Callable[..., Any], publisher: _PylonPublisher | None
) -> _Handler:
    handler = _Handler.__new__(_Handler)
    handler.engine = SimpleNamespace(llm=SimpleNamespace(generate_async=generate_async))
    handler.default_sampling_params = SimpleNamespace(max_tokens=None)
    handler.publisher = None
    handler.metrics_collector = None
    handler.disaggregation_mode = DisaggregationMode.AGGREGATED
    handler.multimodal_processor = None
    handler.additional_metrics = None
    handler.kv_block_size = 16
    handler.max_seq_len = 1024
    handler.first_generation = False
    handler._conversation_affinity = False
    handler._engine_conversation_affinity_override = False
    handler._pylon_stats_publisher = publisher
    handler._pylon_model_name = "served-model"
    handler._normalize_request_format = lambda request: None
    handler._setup_disaggregated_params_for_mode = lambda request, ep_params: (
        None,
        ep_params,
        None,
    )

    async def prepare_input(request, embeddings, ep_params, epd_metadata):
        return request["token_ids"]

    handler._prepare_input_for_generation = prepare_input
    handler._override_sampling_params = lambda defaults, request: SimpleNamespace(
        max_tokens=None
    )
    handler._extract_logprobs = lambda output, cursor: (None, None)

    @asynccontextmanager
    async def cancellation_monitor(generation_result, context):
        yield

    handler._cancellation_monitor = cancellation_monitor

    async def rethrow_shutdown(error: Exception) -> None:
        raise error

    handler._initiate_shutdown = rethrow_shutdown
    return handler


async def _drain(handler: _Handler) -> list[dict[str, Any]]:
    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 16},
    }
    return [
        chunk async for chunk in handler._generate_locally_impl(request, _Context())
    ]


@pytest.mark.asyncio
async def test_request_stats_sum_choices_without_recounting_regressions():
    async def results():
        yield _result([(0, [10], None)], False)
        yield _result([(0, [10, 11], None), (1, [20], None)], False)
        yield _result([(0, [10], None), (1, [20], None)], False)
        yield _result([(0, [10, 11, 12], "stop"), (1, [20, 21], "stop")], True)

    pylon = _PylonPublisher()
    chunks = await _drain(_make_handler(lambda **kwargs: results(), pylon))

    assert len(chunks) == 7
    assert pylon.events[0] == {
        "request_id": "req-1",
        "model": "served-model",
        "tokens_processed": 3,
        "tokens_generated": None,
        "finished": False,
    }
    assert [
        event["tokens_generated"]
        for event in pylon.events
        if event["tokens_generated"] is not None and not event["finished"]
    ] == [1, 2, 3, 4, 5]
    assert pylon.events[-1] == {
        "request_id": "req-1",
        "model": "served-model",
        "tokens_processed": 3,
        "tokens_generated": 5,
        "finished": True,
    }


@pytest.mark.asyncio
async def test_backend_error_closes_without_prompt_progress():
    def generate_async(**kwargs):
        raise RuntimeError("backend admission failed")

    pylon = _PylonPublisher()
    with pytest.raises(RuntimeError, match="backend admission failed"):
        await _drain(_make_handler(generate_async, pylon))

    assert pylon.events == [
        {
            "request_id": "req-1",
            "model": "served-model",
            "tokens_processed": None,
            "tokens_generated": None,
            "finished": True,
        }
    ]


@pytest.mark.asyncio
async def test_empty_error_result_does_not_claim_prompt_progress():
    async def results():
        yield _result([], False)

    pylon = _PylonPublisher()
    assert await _drain(_make_handler(lambda **kwargs: results(), pylon)) == [
        {"finish_reason": "error", "token_ids": []}
    ]
    assert pylon.events[0]["tokens_processed"] is None
    assert pylon.events[0]["finished"] is True


@pytest.mark.asyncio
async def test_cancellation_closes_once():
    async def results():
        yield _result([(0, [10], None)], False)
        await asyncio.Event().wait()

    pylon = _PylonPublisher()
    handler = _make_handler(lambda **kwargs: results(), pylon)
    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 16},
    }
    stream = handler._generate_locally_impl(request, _Context())

    await anext(stream)
    await stream.aclose()
    await stream.aclose()

    terminal = [event for event in pylon.events if event["finished"]]
    assert len(terminal) == 1
    assert terminal[0]["tokens_processed"] == 3
    assert terminal[0]["tokens_generated"] == 1


def _make_metrics_publisher(
    snapshots: list[dict[str, Any]],
    pylon: _PylonPublisher,
    *,
    attention_dp_size: int,
) -> Publisher:
    publisher = Publisher.__new__(Publisher)
    publisher.engine = SimpleNamespace(llm=SimpleNamespace())
    publisher.metrics_publisher = SimpleNamespace(publish=lambda *args, **kwargs: None)
    publisher.component_gauges = SimpleNamespace(
        set_total_blocks=lambda *args: None,
        set_gpu_cache_usage=lambda *args: None,
    )
    publisher.metrics_collector = None
    publisher.fpm_publisher = None
    publisher._fpm_schema_checked = False
    publisher._pylon_stats_publisher = pylon
    publisher._pylon_model_name = "served-model"
    publisher.attention_dp_size = attention_dp_size
    publisher.kv_block_size = 16

    async def polling_loop(
        fetch_fn,
        handler_fn,
        min_sleep,
        max_sleep,
        backoff_factor,
        batch_size_handler_fn=None,
    ):
        for snapshot in snapshots:
            handler_fn(snapshot)

    publisher._polling_loop = polling_loop
    return publisher


@pytest.mark.asyncio
async def test_kv_cache_reuses_existing_metrics_observations_for_all_ranks():
    pylon = _PylonPublisher()
    publisher = _make_metrics_publisher(
        [
            {
                "attentionDpRank": 0,
                "kvCacheStats": {"usedNumBlocks": 4, "maxNumBlocks": 10},
            },
            {
                "attentionDpRank": 1,
                "kvCacheStats": {"usedNumBlocks": 6, "maxNumBlocks": 20},
            },
        ],
        pylon,
        attention_dp_size=2,
    )

    await publisher._publish_stats_task()

    assert pylon.kv_snapshots == [
        ("served-model", 0, 2, 4, 10, 16),
        ("served-model", 1, 2, 6, 20, 16),
    ]


@pytest.mark.asyncio
async def test_kv_cache_skips_missing_capacity_observation():
    pylon = _PylonPublisher()
    publisher = _make_metrics_publisher(
        [{"attentionDpRank": 0, "kvCacheStats": {"usedNumBlocks": 4}}],
        pylon,
        attention_dp_size=1,
    )

    await publisher._publish_stats_task()

    assert pylon.kv_snapshots == []
