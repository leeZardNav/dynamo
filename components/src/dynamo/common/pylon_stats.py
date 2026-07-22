# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request cumulative counters for Dynamo's Pylon stats stream."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamo._core import PylonStatsPublisher

logger = logging.getLogger(__name__)


class PylonRequestStats:
    """Tracks one request's cumulative counters and closes it exactly once."""

    def __init__(
        self,
        publisher: PylonStatsPublisher,
        request_id: str,
    ) -> None:
        self._publisher: PylonStatsPublisher | None = publisher
        self._request_id = request_id
        self._tokens_processed: int | None = None
        self._tokens_generated: int | None = None
        self._finished = False

    def mark_prompt_processed(self, token_count: int) -> None:
        """Publish prompt progress only after backend output confirms work."""
        if self._publisher is None:
            return
        if self._finished or self._tokens_processed is not None:
            return
        self._tokens_processed = token_count
        self._publish(tokens_processed=token_count)

    def add_generated(self, token_delta: int) -> None:
        """Add one backend output delta and publish the new cumulative total."""
        if self._publisher is None:
            return
        if self._finished or token_delta == 0:
            return
        self._tokens_generated = (self._tokens_generated or 0) + token_delta
        self._publish(tokens_generated=self._tokens_generated)

    def finish(self) -> None:
        """Publish the terminal event once for completion, cancellation, or error."""
        if self._finished or self._publisher is None:
            return
        self._finished = True
        self._publish(
            tokens_processed=self._tokens_processed,
            tokens_generated=self._tokens_generated,
            finished=True,
        )

    def _publish(
        self,
        tokens_processed: int | None = None,
        tokens_generated: int | None = None,
        finished: bool = False,
    ) -> None:
        publisher = self._publisher
        if publisher is None:
            return
        try:
            publisher.publish_stats_event(
                self._request_id,
                tokens_processed,
                tokens_generated,
                finished,
            )
        except (OverflowError, RuntimeError, TypeError, ValueError):
            self._publisher = None
            logger.debug(
                "Disabling Pylon stats for request_id=%s after publish failure",
                self._request_id,
                exc_info=True,
            )
