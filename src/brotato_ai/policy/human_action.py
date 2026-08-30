"""Runtime adapter for the event-based human imitation model.

Contract (see docs/ARCHITECTURE.md, "Learned-model contract"):

- input: one ``HumanPolicyFeatureBuilder`` input vector (state trends plus the
  held-action one-hot), matching ``v4.train_event_human_bc.build_examples``.
- output: ``HumanProposal`` with the selected next action (argmax excluding
  the held action, mirroring the offline ``selected_action``), the full action
  probability distribution, the change-gate probability, and a diagnostic
  duration estimate.

The change-gate output is experimental.  Per docs/event_human_imitation_results.md
its hold/change detection is weak (held-out F1 ~0.14), so production modes may
use the action head at explicit decision points but must never use the change
gate to time transitions.

Every failure mode (missing file, schema mismatch, corrupt weights, NaN head
output) degrades to ``None`` plus a counter.  The production controller falls
back to the handcrafted path; learned inference can never crash the loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from brotato_ai.data.human_demo import DATASET_SCHEMA_VERSION

EVENT_CHECKPOINT_FORMAT = "brotato_event_human_bc"
EVENT_POLICY_SCHEMA_VERSION = 1
EVENT_MAX_HOLD_MS = 2_000.0

try:  # Torch is required for the adapter itself; keep import errors explicit.
    import torch
    from torch import nn
except ImportError as _exc:  # pragma: no cover - exercised only on bare envs
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise HumanPolicyError(
            f"torch is required for the human policy adapter: {_TORCH_IMPORT_ERROR}"
        )


class HumanPolicyError(RuntimeError):
    """Raised only while loading a checkpoint; inference never raises."""


if torch is not None:

    class EventHumanModel(nn.Module):
        """Multi-head event model; mirrors the offline training architecture."""

        def __init__(self, width: int, action_count: int = 9):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(width, 256), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(256, 128), nn.ReLU(),
            )
            self.change = nn.Linear(128, 1)
            self.action = nn.Linear(128, action_count)
            self.duration = nn.Linear(128, 1)

        def forward(self, values):
            hidden = self.trunk(values)
            return (
                self.change(hidden).squeeze(-1),
                self.action(hidden),
                self.duration(hidden).squeeze(-1),
            )


else:  # pragma: no cover

    class EventHumanModel:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise HumanPolicyError("torch is not available")


@dataclass(frozen=True)
class HumanProposal:
    """One learned-policy proposal at a decision point."""

    action: int
    probability: float
    probabilities: np.ndarray
    change_probability: float
    duration_ms: float
    held_action: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def save_event_checkpoint(
    path: Path,
    *,
    model: EventHumanModel,
    mean: np.ndarray,
    std: np.ndarray,
    change_threshold: float,
    metrics: Mapping[str, Any],
    dataset: str,
    seed: int,
    action_names: tuple[str, ...],
    history_offsets_ms: tuple[float, ...],
    previous_action_slice: tuple[int, int],
    max_hold_ms: float = EVENT_MAX_HOLD_MS,
    feature_schema_version: int = DATASET_SCHEMA_VERSION,
) -> Path:
    """Persist a trained event model with its full inference contract.

    The payload deliberately contains only primitives, lists, and tensors so
    ``torch.load(..., weights_only=True)`` can restore it safely.
    """

    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import datetime

    payload = {
        "format": EVENT_CHECKPOINT_FORMAT,
        "checkpoint_schema_version": EVENT_POLICY_SCHEMA_VERSION,
        "feature_schema_version": int(feature_schema_version),
        "action_names": [str(name) for name in action_names],
        "semantic_state_width": int(np.asarray(mean).shape[0]),
        "input_width": int(np.asarray(mean).shape[0]) + len(action_names),
        "history_offsets_ms": [float(offset) for offset in history_offsets_ms],
        "previous_action_slice": [int(previous_action_slice[0]), int(previous_action_slice[1])],
        "max_hold_ms": float(max_hold_ms),
        "change_threshold": float(change_threshold),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalization_mean": [float(value) for value in np.asarray(mean).ravel()],
        "normalization_std": [float(value) for value in np.asarray(std).ravel()],
        "metrics": dict(metrics),
        "dataset": str(dataset),
        "seed": int(seed),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    torch.save(payload, path)
    return path


def load_event_checkpoint(path: Path) -> dict[str, Any]:
    """Load and validate a checkpoint payload; raises ``HumanPolicyError``."""

    _require_torch()
    path = Path(path)
    if not path.is_file():
        raise HumanPolicyError(f"human policy checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HumanPolicyError(f"unreadable human policy checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != EVENT_CHECKPOINT_FORMAT:
        raise HumanPolicyError(f"not a {EVENT_CHECKPOINT_FORMAT} checkpoint: {path}")
    version = payload.get("checkpoint_schema_version")
    if version != EVENT_POLICY_SCHEMA_VERSION:
        raise HumanPolicyError(
            f"checkpoint schema mismatch: file={version!r} runtime={EVENT_POLICY_SCHEMA_VERSION}"
        )
    required = (
        "model_state", "normalization_mean", "normalization_std", "action_names",
        "history_offsets_ms", "previous_action_slice", "max_hold_ms", "change_threshold",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise HumanPolicyError(f"checkpoint is missing keys: {missing}")
    return payload


class EventHumanActionPolicy:
    """Loadable, fail-safe adapter around the event human model."""

    def __init__(self, payload: Mapping[str, Any]):
        _require_torch()
        mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
        std = np.asarray(payload["normalization_std"], dtype=np.float32)
        if mean.ndim != 1 or std.shape != mean.shape or not mean.size:
            raise HumanPolicyError("checkpoint normalization vectors are malformed")
        std = np.where(std < 1e-5, 1.0, std)
        action_names = tuple(str(name) for name in payload["action_names"])
        offsets = tuple(float(value) for value in payload["history_offsets_ms"])
        slice_start, slice_stop = (int(value) for value in payload["previous_action_slice"])
        self.state_width = int(mean.shape[0])
        self.action_count = len(action_names)
        self.input_width = self.state_width + self.action_count
        self.max_hold_ms = float(payload["max_hold_ms"])
        self.change_threshold = float(payload["change_threshold"])
        self.feature_schema_version = int(payload.get("feature_schema_version", -1))
        self.history_offsets_ms = offsets
        self.previous_action_slice = slice(slice_start, slice_stop)
        self.action_names = action_names
        self.metrics = dict(payload.get("metrics", {}))
        self.dataset = str(payload.get("dataset", ""))
        self.failure_count = 0
        self.proposal_count = 0
        self._mean = mean
        self._std = std
        self._model = EventHumanModel(self.input_width, self.action_count)
        try:
            self._model.load_state_dict(dict(payload["model_state"]))
        except Exception as exc:
            raise HumanPolicyError(f"checkpoint weights do not match the model: {exc}") from exc
        self._model.eval()

    @classmethod
    def load(cls, path: Path) -> "EventHumanActionPolicy":
        return cls(load_event_checkpoint(path))

    def _normalize(self, model_input: np.ndarray) -> np.ndarray:
        values = np.asarray(model_input, dtype=np.float32).ravel()
        if values.shape[0] != self.input_width:
            raise ValueError(
                f"model input width {values.shape[0]} does not match checkpoint {self.input_width}"
            )
        state = (values[: self.state_width] - self._mean) / self._std
        return np.concatenate((state, values[self.state_width :])).astype(np.float32)

    def propose(self, model_input: np.ndarray, held_action: int) -> HumanProposal | None:
        """Return a proposal, or ``None`` on any failure.  Never raises."""

        if torch is None:  # pragma: no cover
            return None
        try:
            normalized = self._normalize(model_input)
            with torch.no_grad():
                change_logits, action_logits, duration_log = self._model(
                    torch.tensor(normalized[None, :], dtype=torch.float32)
                )
            change_probability = float(torch.sigmoid(change_logits[0]).item())
            probabilities = torch.softmax(action_logits[0], dim=-1).cpu().numpy().astype(np.float64)
            duration_ms = float(
                np.expm1(np.clip(duration_log.cpu().numpy()[0], -10.0, math.log1p(self.max_hold_ms)))
            )
            if not np.isfinite(probabilities).all() or not math.isfinite(duration_ms):
                raise ValueError("non-finite model output")
            held = int(held_action) if 0 <= int(held_action) < self.action_count else 0
            scores = probabilities.copy()
            scores[held] = -np.inf
            action = int(np.argmax(scores))
            self.proposal_count += 1
            return HumanProposal(
                action=action,
                probability=float(probabilities[action]),
                probabilities=probabilities,
                change_probability=change_probability,
                duration_ms=float(np.clip(duration_ms, 0.0, self.max_hold_ms)),
                held_action=held,
                metadata={
                    "change_threshold": self.change_threshold,
                    "feature_schema_version": self.feature_schema_version,
                },
            )
        except Exception:
            self.failure_count += 1
            return None
