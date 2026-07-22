# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock, call

import pytest

from dynamo.common.pylon_stats import PylonRequestStats

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_pylon_request_stats_starts_without_claiming_prompt_progress() -> None:
    publisher = Mock()
    PylonRequestStats(publisher, "req-1")

    publisher.publish_stats_event.assert_not_called()


def test_pylon_request_stats_publishes_cumulative_multi_choice_deltas() -> None:
    publisher = Mock()
    stats = PylonRequestStats(publisher, "req-1")

    stats.mark_prompt_processed(128)
    # Deltas from interleaved choice 0 and choice 1 output.
    stats.add_generated(2)
    stats.add_generated(1)
    stats.add_generated(0)
    stats.add_generated(3)

    assert publisher.publish_stats_event.call_args_list == [
        call("req-1", 128, None, False),
        call("req-1", None, 2, False),
        call("req-1", None, 3, False),
        call("req-1", None, 6, False),
    ]


def test_pylon_request_stats_terminal_event_is_emitted_once() -> None:
    publisher = Mock()
    stats = PylonRequestStats(publisher, "req-1")
    stats.mark_prompt_processed(8)
    stats.add_generated(3)

    stats.finish()
    stats.finish()
    stats.add_generated(1)

    assert publisher.publish_stats_event.call_args_list == [
        call("req-1", 8, None, False),
        call("req-1", None, 3, False),
        call("req-1", 8, 3, True),
    ]


def test_pylon_request_stats_publisher_failure_cannot_break_inference() -> None:
    publisher = Mock()
    publisher.publish_stats_event.side_effect = RuntimeError("publisher unavailable")
    stats = PylonRequestStats(publisher, "req-1")

    stats.mark_prompt_processed(4)
    stats.add_generated(1)
    stats.finish()

    assert publisher.publish_stats_event.call_count == 1
