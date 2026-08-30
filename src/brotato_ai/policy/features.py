"""Live-side builder for the event-based human-policy model input.

The event imitation model is trained on demonstration frames whose semantic
feature vector is produced by ``v4.combat_base.SemanticCombatVectorizer``
and stored in the demo dataset.  This builder reproduces the exact training
input for live states:

    input = concat(
        state_features(t),
        state_features(t) - state_features(t - 200 ms),
        state_features(t) - state_features(t - 400 ms),
        one_hot(currently held action),
    )

where ``state_features`` is the semantic vector with the old previous-action
one-hot slice zeroed out.  The training-side construction lives in
``v4.train_event_human_bc.build_examples``; the parity test
``tests/unit/test_human_feature_parity.py`` guarantees that both paths agree.

Timing is expressed in nanoseconds on a monotonic clock, matching the
demonstration recorder (``frames.timestamp_ns``); the 200/400 ms trend
offsets are relative, so any monotonic source is equivalent.
"""

from __future__ import annotations

import bisect
from typing import Any, Mapping, Protocol

import numpy as np

from brotato_ai.data.human_demo import DATASET_SCHEMA_VERSION

EVENT_FEATURE_SCHEMA_VERSION = DATASET_SCHEMA_VERSION
EVENT_HISTORY_OFFSETS_MS = (0.0, 200.0, 400.0)
EVENT_PREVIOUS_ACTION_SLICE = slice(16, 25)
EVENT_ACTION_COUNT = 9
_EVENT_HISTORY_KEEP_MS = 1_200.0
_FEATURE_DECIMALS = 6


class SemanticVectorizer(Protocol):
    """Minimal contract satisfied by ``v4.combat_base.SemanticCombatVectorizer``."""

    def build(self, state: Mapping[str, Any], previous_action: int = 0) -> np.ndarray: ...


def zero_previous_action_slice(features: np.ndarray) -> np.ndarray:
    """Remove the old explicit previous-action one-hot from a state vector."""

    result = np.asarray(features, dtype=np.float32).copy()
    result[EVENT_PREVIOUS_ACTION_SLICE] = 0.0
    return result


class HumanPolicyFeatureBuilder:
    """Incrementally build event-model inputs from a live state stream.

    The vectorizer dependency is injected or resolved lazily; this module is
    the only production seam allowed to reach into the V4 runtime for the
    training-representation builder, and it never imports V4 at module
    import time so core tests stay lightweight.
    """

    def __init__(
        self,
        vectorizer: SemanticVectorizer | None = None,
        *,
        history_offsets_ms: tuple[float, ...] = EVENT_HISTORY_OFFSETS_MS,
        action_count: int = EVENT_ACTION_COUNT,
    ):
        if not history_offsets_ms or any(offset < 0.0 for offset in history_offsets_ms):
            raise ValueError("history offsets must be non-negative milliseconds")
        self._vectorizer = vectorizer
        self._history_offsets_ms = tuple(float(offset) for offset in history_offsets_ms)
        self._action_count = int(action_count)
        self._timestamps_ms: list[float] = []
        self._features: list[np.ndarray] = []

    def _resolve_vectorizer(self) -> SemanticVectorizer:
        if self._vectorizer is None:
            from v4.combat_base import SemanticCombatVectorizer

            self._vectorizer = SemanticCombatVectorizer()
        return self._vectorizer

    def reset(self) -> None:
        self._timestamps_ms.clear()
        self._features.clear()

    def __len__(self) -> int:
        return len(self._timestamps_ms)

    def observe(
        self,
        state: Mapping[str, Any],
        held_action: int,
        *,
        timestamp_ms: float | None = None,
    ) -> np.ndarray:
        """Record one state and return its zeroed semantic feature vector.

        ``held_action`` is the action currently being held (the model answers
        "hold or change, and if change, which action").  ``timestamp_ms`` must
        come from a monotonic millisecond source; ``published_at_ms`` matches
        the recorder semantics.
        """

        if timestamp_ms is None:
            import time

            timestamp_ms = time.monotonic_ns() / 1e6
        timestamp_ms = float(timestamp_ms)
        raw = self._resolve_vectorizer().build(state, int(held_action))
        state_features = zero_previous_action_slice(
            np.round(np.asarray(raw, dtype=np.float64), _FEATURE_DECIMALS)
        ).astype(np.float32)
        # The demo recorder stores rounded float64 blobs; rounding here keeps
        # live inference byte-comparable with the training representation.
        previous = self._timestamps_ms[-1] if self._timestamps_ms else None
        if previous is not None and timestamp_ms < previous:
            timestamp_ms = previous
        self._timestamps_ms.append(timestamp_ms)
        self._features.append(state_features)
        horizon = max(self._history_offsets_ms) + _EVENT_HISTORY_KEEP_MS
        cutoff = timestamp_ms - horizon
        drop = bisect.bisect_left(self._timestamps_ms, cutoff)
        if drop > 0:
            del self._timestamps_ms[:drop]
            del self._features[:drop]
        return state_features

    def build_input(self, held_action: int) -> np.ndarray:
        """Assemble the current event-model input from observed history.

        Matches the training construction exactly: for each history offset,
        select the latest observed sample at or before ``now - offset`` (never
        later than the current sample), then concatenate the current state
        with the two trend differences and the held-action one-hot.
        """

        if not self._timestamps_ms:
            raise RuntimeError("no state observed; call observe() first")
        now = self._timestamps_ms[-1]
        position = len(self._timestamps_ms) - 1
        snapshots: list[np.ndarray] = []
        for offset_ms in self._history_offsets_ms:
            target = now - offset_ms
            # bisect_right - 1 mirrors np.searchsorted(times, target, "right") - 1.
            index = bisect.bisect_right(self._timestamps_ms, target) - 1
            index = min(position, max(0, index))
            snapshots.append(self._features[index])
        current = snapshots[0]
        trend_200 = current - snapshots[1] if len(snapshots) > 1 else np.zeros_like(current)
        trend_400 = current - snapshots[2] if len(snapshots) > 2 else np.zeros_like(current)
        action_one_hot = np.zeros(self._action_count, dtype=np.float32)
        action = int(held_action)
        if not 0 <= action < self._action_count:
            action = 0
        action_one_hot[action] = 1.0
        return np.concatenate((current, trend_200, trend_400, action_one_hot)).astype(
            np.float32
        )

    @property
    def input_width(self) -> int:
        if not self._features:
            raise RuntimeError("input width is unknown until observe() is called")
        return self._features[0].shape[0] * len(self._history_offsets_ms) + self._action_count
