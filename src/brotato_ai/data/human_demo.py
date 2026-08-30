"""Versioned human-demonstration recording, diagnostics, and replay helpers.

The writer deliberately stores the original state and input alongside derived
values.  Derived values are useful for training and analysis, but are never
allowed to replace the source observation or the analog control signal.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
import zlib
from bisect import bisect_left
from pathlib import Path
from typing import Any, Iterable, Mapping

from brotato_ai.control import CrowdRecoveryGuard, FinalActionArbiter, UnifiedHazardScorer
from brotato_ai.domain.actions import ACTION_VECTORS, MoveAction
from brotato_ai.domain.state import StateSnapshot


DATASET_NAME = "brotato_human_demonstrations"
DATASET_SCHEMA_VERSION = 1
CAPTURE_SCHEMA_VERSION = 3
ACTION_COUNT = len(MoveAction)
TTI_BUCKETS_MS = (50, 100, 150, 250, 400)


def _number(value: Any, default: Any = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else (float(default) if default is not None else 0.0)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _xy(value: Any) -> tuple[float, float]:
    item = _mapping(value)
    return _number(item.get("x")), _number(item.get("y"))


def _json_bytes(value: Any) -> bytes:
    return zlib.compress(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        level=3,
    )


def _from_blob(value: bytes | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(zlib.decompress(value).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
        return default


def normalize_human_input(value: Mapping[str, Any] | None, action: int = 0) -> dict[str, Any]:
    """Normalize input without discarding fields from a newer bridge."""

    source = dict(value) if isinstance(value, Mapping) else {}
    processed = _mapping(source.get("processed_stick"))
    raw = _mapping(source.get("raw_stick"))
    action_vector = ACTION_VECTORS[MoveAction(int(action))]
    source.setdefault("capture_timestamp_ms", None)
    source.setdefault("source", "bridge_processed_only")
    source["raw_available"] = bool(source.get("raw_available", bool(raw)))
    source["raw_stick"] = {
        "x": max(-1.0, min(1.0, _number(raw.get("x"), action_vector[0]))),
        "y": max(-1.0, min(1.0, _number(raw.get("y"), action_vector[1]))),
    }
    source["processed_stick"] = {
        "x": max(-1.0, min(1.0, _number(processed.get("x"), action_vector[0]))),
        "y": max(-1.0, min(1.0, _number(processed.get("y"), action_vector[1]))),
    }
    source.setdefault("buttons", {})
    source.setdefault("triggers", {"left": 0.0, "right": 0.0})
    return source


def quantize_stick(x: float, y: float, *, deadzone: float = 0.18) -> int:
    """Map an analog stick to the existing nine-action vocabulary."""

    x, y = _number(x), _number(y)
    if math.hypot(x, y) < max(0.0, float(deadzone)):
        return int(MoveAction.IDLE)
    horizontal = 1 if x > deadzone else -1 if x < -deadzone else 0
    vertical = 1 if y > deadzone else -1 if y < -deadzone else 0
    if horizontal == 0:
        return int(MoveAction.UP if vertical < 0 else MoveAction.DOWN)
    if vertical == 0:
        return int(MoveAction.LEFT if horizontal < 0 else MoveAction.RIGHT)
    return int({(-1, -1): MoveAction.UP_LEFT, (1, -1): MoveAction.UP_RIGHT,
                (-1, 1): MoveAction.DOWN_LEFT, (1, 1): MoveAction.DOWN_RIGHT}[(horizontal, vertical)])


def _nearest_enemy_distance(state: Mapping[str, Any]) -> tuple[float, str]:
    px, py = _xy(_mapping(state.get("player")).get("position"))
    rows = []
    for enemy in _items(state.get("enemies")):
        ex, ey = _xy(enemy.get("position"))
        rows.append((math.hypot(ex - px, ey - py), str(enemy.get("runtime_id", ""))))
    return min(rows, default=(float("inf"), ""))


def _nearest_projectile_tti(state: Mapping[str, Any], action: int) -> tuple[float, float]:
    from brotato_ai.control.hazards import projectile_time_to_impact

    player = _mapping(state.get("player"))
    position = _xy(player.get("position"))
    combat = _mapping(state.get("combat"))
    speed = max(150.0, _number(combat.get("move_speed"), 300.0))
    movement = ACTION_VECTORS[MoveAction(int(action))]
    values = [
        projectile_time_to_impact(projectile, position, movement, speed)
        for projectile in _items(state.get("projectiles"))
        if bool(projectile.get("hostile", True))
    ]
    return min(values, default=(float("inf"), float("inf")))


def derive_frame(
    state: Mapping[str, Any],
    action: int,
    *,
    previous_action: int = 0,
    previous_timestamp_ms: float | None = None,
    arbiter: FinalActionArbiter | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the current safety/controller architecture as an offline probe.

    This calls the shared production scorer and arbiter; it does not alter
    their thresholds or policy logic and does not send an action to the game.
    """

    normalized = int(MoveAction(int(action)))
    snapshot = StateSnapshot.from_payload(state)
    if arbiter is None:
        shield = UnifiedHazardScorer(enabled=True)
        arbiter = FinalActionArbiter(
            safety_shield=shield,
            crowd_recovery_guard=CrowdRecoveryGuard(shield=shield),
        )
    timestamp = _number(state.get("published_at_ms"), 0.0)
    interval = 0.0 if previous_timestamp_ms is None else max(0.0, timestamp - previous_timestamp_ms)
    trace = arbiter.apply(
        snapshot,
        normalized,
        previous_action=int(previous_action),
        state_interval_ms=interval,
        control_interval_ms=interval,
    )
    risks = arbiter.safety_shield.all_risks(snapshot)
    risk_rows = {str(int(key)): value.to_dict() for key, value in risks.items()}
    safest = min(risks, key=lambda key: (risks[key].total, int(key)))
    nearest_distance, nearest_enemy = _nearest_enemy_distance(state)
    tti, miss_distance = _nearest_projectile_tti(state, normalized)
    player = _mapping(state.get("player"))
    combat = _mapping(state.get("combat"))
    range_target = max(1.0, _number(combat.get("weapon_range"), 170.0))
    spacing_error = nearest_distance - range_target
    candidate_safe = [key for key, risk in risks.items() if risk.total < arbiter.safety_shield.hard_risk_threshold]
    actionable = max((risk.total for risk in risks.values()), default=0.0) - min(
        (risk.total for risk in risks.values()), default=0.0
    ) >= arbiter.safety_shield.override_margin
    controller = trace.to_dict()
    controller.update({
        "candidate_risks": risk_rows,
        "safest_action": int(safest),
        "safest_action_risk": float(risks[safest].total),
        "safe_action_exists": bool(candidate_safe),
        "no_safe_action": not bool(candidate_safe),
    })
    derived = {
        "nearest_enemy_distance": nearest_distance if math.isfinite(nearest_distance) else None,
        "nearest_enemy_runtime_id": nearest_enemy,
        "mean_enemy_separation": (
            sum(
                math.hypot(_xy(enemy.get("position"))[0] - _xy(player.get("position"))[0],
                           _xy(enemy.get("position"))[1] - _xy(player.get("position"))[1])
                for enemy in _items(state.get("enemies"))
            ) / max(1, len(_items(state.get("enemies"))))
        ),
        "nearest_projectile_tti_ms": tti * 1000.0 if math.isfinite(tti) else None,
        "nearest_projectile_miss_distance": miss_distance if math.isfinite(miss_distance) else None,
        "ranged_spacing_target": range_target,
        "ranged_spacing_error": spacing_error if math.isfinite(spacing_error) else None,
        "inside_desired_ranged_spacing": bool(
            math.isfinite(nearest_distance) and 0.70 * range_target <= nearest_distance <= 1.30 * range_target
        ),
        "hazard_actionable": bool(actionable),
        "hazard_tti_ms": tti * 1000.0 if actionable and math.isfinite(tti) else None,
        "human_action": normalized,
        "escape": bool(trace.source in {"hazard", "crowd_recovery"} or trace.recovery_active),
        "safest_action": int(safest),
        "safest_action_selected": bool(normalized == int(safest)),
        "enemy_count": len(_items(state.get("enemies"))),
        "projectile_count": len(_items(state.get("projectiles"))),
        "telegraph_count": len(_items(state.get("attack_indicators"))),
    }
    return controller, derived


class HumanDemoWriter:
    """SQLite writer with stable tables for streaming and random-access replay."""

    def __init__(self, path: Path, *, session_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.RLock()
        self.session_id = session_id or str(uuid.uuid4())
        self.episode_id: str | None = None
        self._last_action = int(MoveAction.IDLE)
        self._last_timestamp_ms: float | None = None
        self._segment_id: str | None = None
        self._segment_started_timestamp_ms: float | None = None
        self._frame_number = 0
        self._last_build_snapshot: Any = None
        self._last_ui_actions: list[Any] = []
        self._arbiter: FinalActionArbiter | None = None
        self._reward_engine: Any = None
        self._bridge_session_id: str | None = None
        self._last_finished_episode_id: str | None = None
        self._last_finished_bridge_session_id: str | None = None
        self._raw_episode_bootstrap = True
        self._manual_mark_ids: set[str] = set()
        self._create_schema()
        self._meta("dataset", DATASET_NAME)
        self._meta("schema_version", DATASET_SCHEMA_VERSION)
        self._meta("capture_schema_version", CAPTURE_SCHEMA_VERSION)
        self._meta("session_id", self.session_id)
        self._meta("created_wall_time_ns", time.time_ns())

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, started_ns INTEGER NOT NULL,
                    ended_ns INTEGER, outcome TEXT, start_phase TEXT, end_phase TEXT,
                    first_frame_id INTEGER, last_frame_id INTEGER,
                    first_timestamp_ns INTEGER, last_timestamp_ns INTEGER,
                    first_bridge_timestamp_ms REAL, last_bridge_timestamp_ms REAL
                );
                CREATE TABLE IF NOT EXISTS frames (
                    frame_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
                    frame_number INTEGER NOT NULL, timestamp_ns INTEGER NOT NULL,
                    wall_time_ns INTEGER NOT NULL, bridge_timestamp_ms REAL, tick INTEGER,
                    phase TEXT NOT NULL, wave INTEGER, action INTEGER NOT NULL,
                    previous_action INTEGER NOT NULL, action_segment_id TEXT,
                    state_blob BLOB NOT NULL, input_blob BLOB NOT NULL,
                    controller_blob BLOB NOT NULL, derived_blob BLOB NOT NULL,
                    feature_blob BLOB, outcome_blob BLOB, reward_blob BLOB
                );
                CREATE TABLE IF NOT EXISTS raw_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    episode_id TEXT, bridge_session_id TEXT,
                    timestamp_ns INTEGER NOT NULL, bridge_timestamp_ms REAL, tick INTEGER,
                    state_blob BLOB NOT NULL, input_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS action_segments (
                    segment_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, action INTEGER NOT NULL,
                    started_ns INTEGER NOT NULL, ended_ns INTEGER, duration_ms REAL
                );
                CREATE TABLE IF NOT EXISTS build_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
                    frame_id INTEGER NOT NULL, timestamp_ns INTEGER NOT NULL, phase TEXT NOT NULL,
                    available_blob BLOB NOT NULL, build_before_blob BLOB NOT NULL,
                    build_after_blob BLOB, chosen_action TEXT, source TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    frame_id INTEGER PRIMARY KEY, outcome_50ms BLOB, outcome_100ms BLOB,
                    outcome_250ms BLOB, outcome_500ms BLOB, outcome_1000ms BLOB
                );
                CREATE TABLE IF NOT EXISTS labels (
                    label_id INTEGER PRIMARY KEY AUTOINCREMENT, frame_id INTEGER NOT NULL,
                    label TEXT NOT NULL, value TEXT, annotator TEXT, created_ns INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS frames_episode_time ON frames(episode_id, timestamp_ns);
                CREATE INDEX IF NOT EXISTS raw_session_time ON raw_samples(session_id, timestamp_ns);
                """
            )
            # Keep the recorder backward compatible with existing v1 datasets.
            # New captures use these columns; old source databases are never
            # rewritten by this migration unless opened for append explicitly.
            migrations = {
                "episodes": {
                    "first_frame_id": "INTEGER",
                    "last_frame_id": "INTEGER",
                    "first_timestamp_ns": "INTEGER",
                    "last_timestamp_ns": "INTEGER",
                    "first_bridge_timestamp_ms": "REAL",
                    "last_bridge_timestamp_ms": "REAL",
                },
                "frames": {"reward_blob": "BLOB"},
                "raw_samples": {
                    "episode_id": "TEXT",
                    "bridge_session_id": "TEXT",
                },
            }
            for table, columns in migrations.items():
                existing = {
                    str(row[1]) for row in self.connection.execute(f"PRAGMA table_info({table})")
                }
                for name, declaration in columns.items():
                    if name not in existing:
                        self.connection.execute(
                            f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                        )

    def set_metadata(self, key: str, value: Any) -> None:
        """Store capture provenance without changing any game/controller state."""

        with self.lock:
            self._meta(key, value)

    def _meta(self, key: str, value: Any) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                (str(key), json.dumps(value, separators=(",", ":"))),
            )

    def start_episode(
        self,
        *,
        phase: str = "unknown",
        episode_id: str | None = None,
        source_session_id: str | None = None,
    ) -> str:
        with self.lock:
            self.episode_id = episode_id or f"{self.session_id}:{uuid.uuid4().hex[:12]}"
            self._last_action = int(MoveAction.IDLE)
            self._last_timestamp_ms = None
            self._segment_id = None
            self._segment_started_timestamp_ms = None
            self._frame_number = 0
            self._last_build_snapshot = None
            self._last_ui_actions = []
            self._arbiter = None
            self._reward_engine = None
            self._bridge_session_id = (
                str(source_session_id) if source_session_id not in (None, "") else None
            )
            self._last_finished_episode_id = None
            self._last_finished_bridge_session_id = None
            self.connection.execute(
                "INSERT OR REPLACE INTO episodes(episode_id,session_id,started_ns,start_phase) VALUES(?,?,?,?)",
                (self.episode_id, self.session_id, time.monotonic_ns(), str(phase)),
            )
            self.connection.commit()
            return self.episode_id

    def record_raw_sample(self, state: Mapping[str, Any]) -> None:
        """Store the independent 60 Hz stream, including raw input when present."""

        payload = dict(state)
        action = int(payload.get("human_action", payload.get("action", 0)))
        input_value = normalize_human_input(_mapping(payload.get("human_input")), action)
        bridge_session_id = str(payload.get("session", "")) or None
        bridge_timestamp = payload.get("published_at_ms")
        try:
            bridge_timestamp = float(bridge_timestamp) if bridge_timestamp is not None else None
        except (TypeError, ValueError):
            bridge_timestamp = None
        with self.lock:
            phase = str(payload.get("phase", "unknown"))
            episode_id: str | None = None
            if self.episode_id is not None and self._bridge_session_id in {None, bridge_session_id}:
                episode_id = self.episode_id
            elif (
                self._last_finished_episode_id is not None
                and bridge_session_id == self._last_finished_bridge_session_id
            ):
                # The terminal screen can continue producing raw ticks after
                # the rich loop closes the episode. Raw payloads expose the
                # Godot scene name (for example Main or EndRun), not the rich
                # semantic game_over/victory phase, so use the bridge-session
                # boundary rather than a phase-name allowlist.
                episode_id = self._last_finished_episode_id
            if (
                episode_id is None
                and self.episode_id is None
                and self._raw_episode_bootstrap
                and bridge_session_id
                and phase not in {"game_over", "victory"}
            ):
                # The raw stream can beat the first rich frame by a few
                # milliseconds. Starting the episode here preserves its exact
                # lower boundary instead of leaving those samples orphaned.
                self.start_episode(phase=phase, source_session_id=bridge_session_id)
                episode_id = self.episode_id
            self.connection.execute(
                "INSERT INTO raw_samples(session_id,episode_id,bridge_session_id,timestamp_ns,bridge_timestamp_ms,tick,state_blob,input_blob) VALUES(?,?,?,?,?,?,?,?)",
                (self.session_id, episode_id, bridge_session_id, time.monotonic_ns(), bridge_timestamp,
                 int(_number(payload.get("tick"), -1)), _json_bytes(payload), _json_bytes(input_value)),
            )
            self.connection.commit()

    def record_frame(self, state: Mapping[str, Any], *, received_ns: int | None = None) -> int:
        with self.lock:
            if self.episode_id is None:
                self.start_episode(
                    phase=str(state.get("phase", "unknown")),
                    source_session_id=str(state.get("session", "")) or None,
                )
            assert self.episode_id is not None
            source_session_id = str(state.get("session", "")) or None
            if (
                self._bridge_session_id is not None
                and source_session_id is not None
                and source_session_id != self._bridge_session_id
            ):
                self.finish_episode(outcome="session_boundary", end_phase="session_boundary")
                self.start_episode(
                    phase=str(state.get("phase", "unknown")),
                    source_session_id=source_session_id,
                )
            if self._bridge_session_id is None and source_session_id:
                self._bridge_session_id = source_session_id
            action = int(state.get("human_action", state.get("action", 0)))
            if not 0 <= action < ACTION_COUNT:
                action = int(MoveAction.IDLE)
            timestamp_ms = _number(state.get("published_at_ms"), time.monotonic_ns() / 1e6)
            if timestamp_ms < 0.0:
                timestamp_ms = time.monotonic_ns() / 1e6
            input_value = normalize_human_input(_mapping(state.get("human_input")), action)
            now_ns = int(received_ns or time.monotonic_ns())
            if self._segment_id is None or action != self._last_action:
                if self._segment_id is not None:
                    self.connection.execute(
                        "UPDATE action_segments SET ended_ns=?,duration_ms=? WHERE segment_id=?",
                        (now_ns,
                         max(0.0, timestamp_ms - (_number(self._segment_started_timestamp_ms))), self._segment_id),
                    )
                self._segment_id = uuid.uuid4().hex
                self.connection.execute(
                    "INSERT INTO action_segments(segment_id,episode_id,action,started_ns) VALUES(?,?,?,?)",
                    (self._segment_id, self.episode_id, action, now_ns),
                )
                self._segment_started_timestamp_ms = timestamp_ms
            if self._arbiter is None:
                shield = UnifiedHazardScorer(enabled=True)
                self._arbiter = FinalActionArbiter(
                    safety_shield=shield,
                    crowd_recovery_guard=CrowdRecoveryGuard(shield=shield),
                )
            controller, derived = derive_frame(
                state, action, previous_action=self._last_action,
                previous_timestamp_ms=self._last_timestamp_ms, arbiter=self._arbiter,
            )
            reward_total, reward_components = self._record_reward(state)
            derived.update(
                {
                    "action_transition": bool(self._frame_number > 0 and action != self._last_action),
                    "action_hold_ms": max(
                        0.0,
                        timestamp_ms - _number(self._segment_started_timestamp_ms),
                    ),
                    "reward_total": float(reward_total),
                    "reward_components": reward_components,
                    "reward_source": "ApiRewardEngine",
                    "reward_exact_environment_step": False,
                }
            )
            try:
                from v3.combat_policy import SemanticCombatVectorizer
                features = [round(float(item), 6) for item in SemanticCombatVectorizer().build(state, self._last_action)]
            except Exception:
                features = None
            cursor = self.connection.execute(
                "INSERT INTO frames(episode_id,frame_number,timestamp_ns,wall_time_ns,bridge_timestamp_ms,tick,phase,wave,action,previous_action,action_segment_id,state_blob,input_blob,controller_blob,derived_blob,feature_blob,reward_blob) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.episode_id, self._frame_number, now_ns, time.time_ns(), timestamp_ms,
                 int(_number(state.get("tick"), -1)), str(state.get("phase", "unknown")),
                 int(_number(_mapping(state.get("wave")).get("number"))), action, self._last_action,
                 self._segment_id, _json_bytes(dict(state)), _json_bytes(input_value),
                 _json_bytes(controller), _json_bytes(derived), _json_bytes(features) if features else None,
                 _json_bytes({"total": reward_total, "components": reward_components, "source": "ApiRewardEngine"})),
            )
            frame_id = int(cursor.lastrowid)
            self._record_manual_marks(frame_id, state)
            if self._frame_number == 0:
                self.connection.execute(
                    "UPDATE episodes SET first_frame_id=?,first_timestamp_ns=?,first_bridge_timestamp_ms=? WHERE episode_id=?",
                    (frame_id, now_ns, timestamp_ms, self.episode_id),
                )
            self.connection.execute(
                "UPDATE episodes SET last_frame_id=?,last_timestamp_ns=?,last_bridge_timestamp_ms=? WHERE episode_id=?",
                (frame_id, now_ns, timestamp_ms, self.episode_id),
            )
            self._record_build_decision(frame_id, state, now_ns)
            self._frame_number += 1
            self._last_action = action
            self._last_timestamp_ms = timestamp_ms
            self.connection.commit()
            return frame_id

    def _record_manual_marks(self, frame_id: int, state: Mapping[str, Any]) -> int:
        """Attach every new in-game F9 bookmark to the observed frame."""

        marks = state.get("manual_marks", [])
        if not isinstance(marks, list):
            return 0
        recorded = 0
        for mark in marks:
            if not isinstance(mark, Mapping):
                continue
            marker_id = str(mark.get("marker_id", ""))
            if not marker_id or marker_id in self._manual_mark_ids:
                continue
            value = dict(mark)
            value["recorded_frame_id"] = int(frame_id)
            self.connection.execute(
                "INSERT INTO labels(frame_id,label,value,annotator,created_ns) VALUES(?,?,?,?,?)",
                (
                    int(frame_id),
                    "manual_bookmark",
                    json.dumps(value, separators=(",", ":")),
                    "in_game_f9",
                    time.time_ns(),
                ),
            )
            self._manual_mark_ids.add(marker_id)
            recorded += 1
        return recorded

    def record_manual_marks_for_last_frame(self, state: Mapping[str, Any]) -> int:
        """Save marks made on a terminal screen after its episode was closed."""

        with self.lock:
            episode_id = self._last_finished_episode_id
            if not episode_id:
                return 0
            row = self.connection.execute(
                "SELECT frame_id FROM frames WHERE episode_id=? ORDER BY frame_number DESC, frame_id DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
            if row is None:
                return 0
            recorded = self._record_manual_marks(int(row[0]), state)
            if recorded:
                self.connection.commit()
            return recorded

    def _record_reward(self, state: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
        """Capture the action-independent API reward components offline.

        The observation-only recorder does not execute ``BrotatoApiEnv.step``.
        This therefore records the shared ``ApiRewardEngine`` signal, while
        explicitly marking it as non-identical to a live environment step that
        also adds controller-dependent movement shaping.
        """

        if self._reward_engine is None:
            from v3.reward import ApiRewardEngine

            self._reward_engine = ApiRewardEngine()
            self._reward_engine.reset(state)
            return 0.0, {}
        total = float(self._reward_engine.step(state))
        return total, {
            str(key): float(value)
            for key, value in self._reward_engine.last_components.items()
        }

    def _record_build_decision(self, frame_id: int, state: Mapping[str, Any], timestamp_ns: int) -> None:
        phase = str(state.get("phase", "unknown"))
        if phase in {"combat", "unknown"}:
            return
        ui = _mapping(state.get("ui"))
        actions = ui.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        current_build = state.get("build", {})
        ui_result = _mapping(ui.get("last_result"))
        chosen = ui_result.get("target") or ui_result.get("action") or state.get("selected_ui_action")
        if chosen is None and self._last_build_snapshot is not None and current_build != self._last_build_snapshot:
            chosen = "inferred_build_change"
        before = {} if self._last_build_snapshot is None else self._last_build_snapshot
        self.connection.execute(
            "INSERT INTO build_decisions(episode_id,frame_id,timestamp_ns,phase,available_blob,build_before_blob,build_after_blob,chosen_action,source) VALUES(?,?,?,?,?,?,?,?,?)",
            (self.episode_id, frame_id, timestamp_ns, phase, _json_bytes(actions),
             _json_bytes(before), _json_bytes(current_build), chosen,
             "explicit_ui_result" if chosen and chosen != "inferred_build_change" else
             "inferred_from_build_delta" if chosen else "observed_ui_snapshot"),
        )
        self._last_build_snapshot = current_build
        self._last_ui_actions = actions

    def finish_episode(self, *, outcome: str = "unknown", end_phase: str = "unknown") -> None:
        with self.lock:
            if self.episode_id is None:
                return
            finished_episode_id = self.episode_id
            finished_bridge_session_id = self._bridge_session_id
            if self._segment_id is not None:
                self.connection.execute(
                    "UPDATE action_segments SET ended_ns=?,duration_ms=? WHERE segment_id=? AND ended_ns IS NULL",
                    (time.monotonic_ns(),
                     max(0.0, _number(self._last_timestamp_ms) - _number(self._segment_started_timestamp_ms))
                     if self._last_timestamp_ms is not None else None,
                     self._segment_id),
                )
            self.connection.execute(
                "UPDATE episodes SET ended_ns=?,outcome=?,end_phase=? WHERE episode_id=?",
                (time.monotonic_ns(), str(outcome), str(end_phase), self.episode_id),
            )
            self.connection.commit()
            self._last_finished_episode_id = finished_episode_id
            self._last_finished_bridge_session_id = finished_bridge_session_id
            self.episode_id = None
            self._segment_id = None
            self._segment_started_timestamp_ms = None
            self._bridge_session_id = None
            self._reward_engine = None
            # If capture continues after a terminal screen, the next
            # non-terminal raw tick may bootstrap a new episode. Terminal raw
            # ticks are handled above and remain attached to the finished one.
            self._raw_episode_bootstrap = True

    def add_label(
        self, frame_id: int, label: str, value: Any = True, *, annotator: str = "manual"
    ) -> None:
        """Attach intent/mistake labels without rewriting the source frame."""

        with self.lock:
            self.connection.execute(
                "INSERT INTO labels(frame_id,label,value,annotator,created_ns) VALUES(?,?,?,?,?)",
                (int(frame_id), str(label), json.dumps(value, separators=(",", ":")),
                 str(annotator), time.time_ns()),
            )
            self.connection.commit()

    def finalize(self) -> dict[str, Any]:
        """Add fixed-horizon outcomes and return validation diagnostics."""

        with self.lock:
            self._populate_transitions()
            self.connection.commit()
        return validate_dataset(self.path)

    def _populate_transitions(self) -> None:
        episodes = self.connection.execute("SELECT DISTINCT episode_id FROM frames").fetchall()
        for (episode_id,) in episodes:
            rows = self.connection.execute(
                "SELECT frame_id,timestamp_ns,state_blob,derived_blob FROM frames WHERE episode_id=? ORDER BY timestamp_ns,frame_id",
                (episode_id,),
            ).fetchall()
            times = [int(row[1]) for row in rows]
            states = [_from_blob(row[2], {}) for row in rows]
            for index, (frame_id, timestamp_ns, _state_blob, derived_blob) in enumerate(rows):
                outcomes = {}
                for horizon in (50, 100, 250, 500, 1000):
                    target = int(timestamp_ns) + horizon * 1_000_000
                    target_index = min(len(rows) - 1, bisect_left(times, target))
                    future = states[target_index]
                    current = states[index]
                    current_hp = _number(_mapping(current.get("player")).get("health"))
                    future_hp = _number(_mapping(future.get("player")).get("health"))
                    outcomes[str(horizon)] = {
                        "observed_timestamp_delta_ms": max(0.0, (times[target_index] - times[index]) / 1e6),
                        "health_delta": future_hp - current_hp,
                        "health_loss": max(0.0, current_hp - future_hp),
                        "dead": bool(future.get("dead")),
                        "victory": bool(future.get("victory")),
                        "phase": str(future.get("phase", "")),
                        "wave": int(_number(_mapping(future.get("wave")).get("number"))),
                    }
                self.connection.execute(
                    "INSERT OR REPLACE INTO transitions(frame_id,outcome_50ms,outcome_100ms,outcome_250ms,outcome_500ms,outcome_1000ms) VALUES(?,?,?,?,?,?)",
                    (frame_id, *[_json_bytes(outcomes[str(h)]) for h in (50, 100, 250, 500, 1000)]),
                )
                self.connection.execute("UPDATE frames SET outcome_blob=? WHERE frame_id=?", (_json_bytes(outcomes), frame_id))

    def close(self) -> None:
        with self.lock:
            if self.episode_id is not None:
                self.finish_episode()
            self.connection.close()

    def __enter__(self) -> "HumanDemoWriter":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    return {key: _from_blob(value.encode("utf-8"), value) for key, value in connection.execute("SELECT key,value FROM metadata")}


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _decode_checked(value: bytes | None, default: Any, label: str) -> tuple[Any, str | None]:
    if not value:
        return default, f"missing {label}"
    try:
        decoded = json.loads(zlib.decompress(value).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, zlib.error) as exc:
        return default, f"corrupt {label}: {exc.__class__.__name__}"
    return decoded, None


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def validate_dataset(path: Path, *, require_capture: bool = False) -> dict[str, Any]:
    """Validate synchronization, stream integrity, input coverage, and continuity.

    ``require_capture`` is opt-in so historical v1 datasets remain readable. A
    new manual recording should use it (or the recorder's matching flag): it
    additionally requires raw samples, reward telemetry, semantic features,
    fixed-horizon outcomes, and closed episode boundaries.
    """

    result: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "schema_version": DATASET_SCHEMA_VERSION,
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "errors": [],
        "warnings": [],
        "capture_errors": [],
    }
    if not Path(path).is_file():
        result["errors"].append(f"dataset not found: {path}")
        result["ok"] = False
        result["capture_ready"] = False
        return result

    connection = sqlite3.connect(str(path))
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        result["sqlite_integrity"] = integrity
        if integrity.lower() != "ok":
            result["errors"].append(f"sqlite integrity check failed: {integrity}")

        required_tables = {
            "metadata", "episodes", "frames", "raw_samples", "action_segments",
            "build_decisions", "transitions", "labels",
        }
        present_tables = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing_tables = sorted(required_tables - present_tables)
        if missing_tables:
            result["errors"].append(f"missing tables: {', '.join(missing_tables)}")
            result["ok"] = False
            result["capture_ready"] = False
            return result

        frame_columns = _table_columns(connection, "frames")
        raw_columns = _table_columns(connection, "raw_samples")
        episode_columns = _table_columns(connection, "episodes")
        has_reward_blob = "reward_blob" in frame_columns
        has_raw_episode = "episode_id" in raw_columns
        has_raw_bridge_session = "bridge_session_id" in raw_columns
        has_episode_boundaries = {
            "first_frame_id", "last_frame_id", "first_timestamp_ns", "last_timestamp_ns",
        }.issubset(episode_columns)
        if not has_reward_blob:
            result["warnings"].append("dataset predates explicit reward telemetry")
            result["capture_errors"].append("reward_blob column is missing")
        if not has_raw_episode:
            result["warnings"].append("dataset predates raw-sample episode association")
        if not has_episode_boundaries:
            result["warnings"].append("dataset predates explicit episode frame boundaries")

        reward_select = "reward_blob" if has_reward_blob else "NULL AS reward_blob"
        frames = connection.execute(
            f"""
            SELECT frame_id,episode_id,frame_number,timestamp_ns,bridge_timestamp_ms,tick,
                   phase,wave,action,previous_action,action_segment_id,state_blob,input_blob,
                   controller_blob,derived_blob,feature_blob,outcome_blob,{reward_select}
            FROM frames ORDER BY episode_id,timestamp_ns,frame_id
            """
        ).fetchall()
        episode_rows = connection.execute(
            "SELECT episode_id,session_id,started_ns,ended_ns,outcome,start_phase,end_phase"
            + (",first_frame_id,last_frame_id,first_timestamp_ns,last_timestamp_ns,first_bridge_timestamp_ms,last_bridge_timestamp_ms" if has_episode_boundaries else "")
            + " FROM episodes ORDER BY started_ns,episode_id"
        ).fetchall()
        raw_rows = connection.execute(
            "SELECT sample_id," + ("episode_id," if has_raw_episode else "NULL,")
            + ("bridge_session_id," if has_raw_bridge_session else "NULL,")
            + "timestamp_ns,bridge_timestamp_ms,tick,state_blob,input_blob "
            "FROM raw_samples ORDER BY bridge_session_id,timestamp_ns,sample_id"
            if has_raw_bridge_session
            else "SELECT sample_id," + ("episode_id," if has_raw_episode else "NULL,")
            + "NULL,timestamp_ns,bridge_timestamp_ms,tick,state_blob,input_blob "
            "FROM raw_samples ORDER BY timestamp_ns,sample_id"
        ).fetchall()

        result.update({
            "frames": len(frames),
            "raw_samples": len(raw_rows),
            "episodes": len(episode_rows),
            "build_decisions": int(connection.execute("SELECT COUNT(*) FROM build_decisions").fetchone()[0]),
            "manual_marks": int(connection.execute("SELECT COUNT(*) FROM labels WHERE label='manual_bookmark'").fetchone()[0]),
        })
        last_by_episode: dict[str, tuple[int, float | None, int]] = {}
        frame_gaps: list[float] = []
        timestamp_drifts: list[float] = []
        source_missing = 0
        raw_available = 0
        processed_input = 0
        feature_rows = 0
        reward_rows = 0
        outcome_rows = 0
        bridge_timestamp_rows = 0
        corrupted_blobs: list[str] = []
        frame_counts: dict[str, int] = {}
        for row in frames:
            (
                frame_id, episode, frame_number, timestamp_ns, bridge_ms, _tick, phase, _wave,
                action, previous, segment, state_blob, input_blob, controller_blob,
                derived_blob, feature_blob, outcome_blob, reward_blob,
            ) = row
            episode = str(episode)
            frame_counts[episode] = frame_counts.get(episode, 0) + 1
            timestamp_ns = int(timestamp_ns)
            bridge_value: float | None
            try:
                bridge_value = float(bridge_ms) if bridge_ms is not None else None
            except (TypeError, ValueError):
                bridge_value = None
            if bridge_value is not None and bridge_value >= 0.0:
                bridge_timestamp_rows += 1
            previous_row = last_by_episode.get(episode)
            if previous_row is not None:
                if int(frame_number) != previous_row[2] + 1:
                    result["errors"].append(f"frame-number gap in episode {episode}")
                # Equal values are valid when the platform clock exposes a
                # coarser resolution than the event stream. frame_id is the
                # stable tie-breaker in the query order above.
                if timestamp_ns < previous_row[0]:
                    result["errors"].append(f"non-monotonic timestamp in episode {episode}")
                local_delta = (timestamp_ns - previous_row[0]) / 1e6
                frame_gaps.append(local_delta)
                if local_delta > 250.0:
                    result["warnings"].append(f"frame gap {local_delta:.1f} ms in episode {episode}")
                if bridge_value is not None and previous_row[1] is not None:
                    source_delta = bridge_value - previous_row[1]
                    if source_delta < 0.0:
                        result["errors"].append(f"non-monotonic bridge timestamp in episode {episode}")
                    timestamp_drifts.append(local_delta - max(0.0, source_delta))
            last_by_episode[episode] = (timestamp_ns, bridge_value, int(frame_number))
            if not 0 <= int(action) < ACTION_COUNT:
                result["errors"].append(f"frame action out of range: {action}")
            if not 0 <= int(previous) < ACTION_COUNT:
                result["errors"].append(f"previous action out of range: {previous}")
            if int(action) != int(previous) and not segment:
                result["errors"].append(f"action change without persistence segment at frame {frame_id}")

            state, error = _decode_checked(state_blob, {}, f"state frame {frame_id}")
            if error:
                corrupted_blobs.append(error)
            input_value, error = _decode_checked(input_blob, {}, f"input frame {frame_id}")
            if error:
                corrupted_blobs.append(error)
            controller, error = _decode_checked(controller_blob, {}, f"controller frame {frame_id}")
            if error:
                corrupted_blobs.append(error)
            derived, error = _decode_checked(derived_blob, {}, f"derived frame {frame_id}")
            if error:
                corrupted_blobs.append(error)
            if not isinstance(state, Mapping) or not isinstance(input_value, Mapping):
                result["errors"].append(f"invalid state/input shape at frame {frame_id}")
            else:
                if isinstance(input_value.get("processed_stick"), Mapping):
                    processed_input += 1
                else:
                    result["capture_errors"].append(f"processed input missing at frame {frame_id}")
                raw_available += int(bool(input_value.get("raw_available")))
            if not isinstance(controller, Mapping) or not isinstance(derived, Mapping):
                result["errors"].append(f"invalid diagnostics shape at frame {frame_id}")
            features, error = _decode_checked(feature_blob, [], f"features frame {frame_id}")
            if error:
                result["capture_errors"].append(error)
            elif isinstance(features, list) and len(features) == 832:
                feature_rows += 1
            else:
                result["capture_errors"].append(f"semantic feature width is not 832 at frame {frame_id}")
            outcomes, error = _decode_checked(outcome_blob, {}, f"outcomes frame {frame_id}")
            if error:
                result["capture_errors"].append(error)
            elif isinstance(outcomes, Mapping) and all(str(horizon) in outcomes for horizon in (50, 100, 250, 500, 1000)):
                outcome_rows += 1
            reward, error = _decode_checked(reward_blob, {}, f"reward frame {frame_id}")
            if error:
                result["capture_errors"].append(error)
            elif isinstance(reward, Mapping) and isinstance(reward.get("components"), Mapping):
                reward_rows += 1
            if isinstance(derived, Mapping) and "reward_components" not in derived:
                result["capture_errors"].append(f"derived reward components missing at frame {frame_id}")

        if corrupted_blobs:
            result["errors"].extend(corrupted_blobs)

        raw_gaps: list[float] = []
        raw_drifts: list[float] = []
        raw_last: dict[str, tuple[int, float | None]] = {}
        raw_decoded = 0
        raw_unassigned = 0
        raw_bridge_timestamp_rows = 0
        for row in raw_rows:
            sample_id, raw_episode, raw_bridge_session, timestamp_ns, bridge_ms, _tick, state_blob, input_blob = row
            state, error = _decode_checked(state_blob, {}, f"raw state sample {sample_id}")
            if error:
                result["errors"].append(error)
            input_value, error = _decode_checked(input_blob, {}, f"raw input sample {sample_id}")
            if error:
                result["errors"].append(error)
            if isinstance(state, Mapping) and isinstance(input_value, Mapping):
                raw_decoded += 1
            if raw_episode is None:
                raw_unassigned += 1
            source_key = str(raw_bridge_session or _mapping(state).get("session") or "unknown")
            try:
                bridge_value = float(bridge_ms) if bridge_ms is not None else None
            except (TypeError, ValueError):
                bridge_value = None
            if bridge_value is not None and bridge_value >= 0.0:
                raw_bridge_timestamp_rows += 1
            if raw_episode is not None and str(raw_episode) not in frame_counts:
                result["capture_errors"].append(
                    f"raw sample {sample_id} references unknown episode {raw_episode}"
                )
            previous_raw = raw_last.get(source_key)
            if previous_raw is not None:
                local_delta = (int(timestamp_ns) - previous_raw[0]) / 1e6
                if local_delta < 0.0:
                    result["errors"].append(f"non-monotonic raw timestamp in stream {source_key}")
                else:
                    raw_gaps.append(local_delta)
                if bridge_value is not None and previous_raw[1] is not None:
                    source_delta = bridge_value - previous_raw[1]
                    if source_delta < 0.0:
                        result["errors"].append(f"non-monotonic raw bridge timestamp in stream {source_key}")
                    raw_drifts.append(local_delta - max(0.0, source_delta))
            raw_last[source_key] = (int(timestamp_ns), bridge_value)

        closed_episodes = 0
        boundary_errors = []
        episode_ids = set(frame_counts)
        for row in episode_rows:
            episode_id, _session_id, started_ns, ended_ns, outcome, _start_phase, _end_phase, *boundaries = row
            episode_id = str(episode_id)
            if frame_counts.get(episode_id, 0) == 0:
                boundary_errors.append(f"episode has no frames: {episode_id}")
            if ended_ns is not None:
                closed_episodes += 1
            else:
                boundary_errors.append(f"episode is not closed: {episode_id}")
            if episode_id not in episode_ids:
                boundary_errors.append(f"episode frame index missing: {episode_id}")
            if has_episode_boundaries:
                first_frame_id, last_frame_id, first_timestamp_ns, last_timestamp_ns, *_ = boundaries
                if frame_counts.get(episode_id, 0) and (
                    first_frame_id is None or last_frame_id is None
                    or first_timestamp_ns is None or last_timestamp_ns is None
                ):
                    boundary_errors.append(f"explicit frame boundary missing: {episode_id}")
            if require_capture and str(outcome or "unknown") not in {"death", "victory"}:
                result["capture_errors"].append(
                    f"episode outcome is not death/victory: {episode_id}={outcome!r}"
                )
        result["errors"].extend(boundary_errors)

        result["raw_input_coverage"] = raw_available / len(frames) if frames else 0.0
        result["processed_input_coverage"] = processed_input / len(frames) if frames else 0.0
        result["feature_coverage"] = feature_rows / len(frames) if frames else 0.0
        result["reward_coverage"] = reward_rows / len(frames) if frames else 0.0
        result["outcome_coverage"] = outcome_rows / len(frames) if frames else 0.0
        result["bridge_timestamp_coverage"] = bridge_timestamp_rows / len(frames) if frames else 0.0
        result["raw_bridge_timestamp_coverage"] = raw_bridge_timestamp_rows / len(raw_rows) if raw_rows else 0.0
        result["raw_decoded_coverage"] = raw_decoded / len(raw_rows) if raw_rows else 0.0
        result["raw_unassigned_samples"] = raw_unassigned
        result["mean_frame_interval_ms"] = sum(frame_gaps) / len(frame_gaps) if frame_gaps else 0.0
        result["p90_frame_interval_ms"] = _percentile(frame_gaps, 0.90) or 0.0
        result["timestamp_drift_ms"] = {
            "interval_count": len(timestamp_drifts),
            "mean_signed": sum(timestamp_drifts) / len(timestamp_drifts) if timestamp_drifts else None,
            "p90_abs": _percentile((abs(value) for value in timestamp_drifts), 0.90),
            "max_abs": max((abs(value) for value in timestamp_drifts), default=None),
            "definition": "local monotonic frame interval minus bridge published_at_ms interval",
        }
        result["raw_timestamp_drift_ms"] = {
            "interval_count": len(raw_drifts),
            "mean_signed": sum(raw_drifts) / len(raw_drifts) if raw_drifts else None,
            "p90_abs": _percentile((abs(value) for value in raw_drifts), 0.90),
            "max_abs": max((abs(value) for value in raw_drifts), default=None),
        }
        result["stream_status"] = {
            "rich_frames": len(frames) > 0,
            "raw_samples": len(raw_rows) > 0,
            "raw_samples_decodable": raw_decoded == len(raw_rows) if raw_rows else False,
            "processed_input_present": processed_input == len(frames) if frames else False,
            "semantic_features_present": feature_rows == len(frames) if frames else False,
            "reward_components_present": reward_rows == len(frames) if frames else False,
            "fixed_horizon_outcomes_present": outcome_rows == len(frames) if frames else False,
            "episode_boundaries_closed": closed_episodes == len(episode_rows) if episode_rows else False,
        }
        if not frames:
            result["capture_errors"].append("no rich frames recorded")
        if not raw_rows:
            result["capture_errors"].append("no raw samples recorded")
        if bridge_timestamp_rows != len(frames):
            result["capture_errors"].append("one or more rich frames lack a valid bridge timestamp")
        if raw_rows and raw_bridge_timestamp_rows != len(raw_rows):
            result["capture_errors"].append("one or more raw samples lack a valid bridge timestamp")
        if raw_unassigned:
            result["warnings"].append(f"{raw_unassigned} raw samples fall outside a rich episode boundary")
        if result["timestamp_drift_ms"]["p90_abs"] is not None and result["timestamp_drift_ms"]["p90_abs"] > 250.0:
            result["warnings"].append("rich/source timestamp interval drift p90 exceeds 250 ms")
        result["metrics"] = summarize_dataset(path, connection=connection)
        if require_capture:
            result["errors"].extend(result["capture_errors"])
        result["ok"] = not result["errors"]
        result["capture_ready"] = not result["errors"] and not result["capture_errors"]
        result["require_capture"] = bool(require_capture)
        return result
    finally:
        connection.close()


def summarize_dataset(
    path: Path, *, connection: sqlite3.Connection | None = None
) -> dict[str, Any]:
    """Return the common outcome metrics used when comparing demonstrations."""

    owns_connection = connection is None
    connection = connection or sqlite3.connect(str(path))
    rows = connection.execute(
        "SELECT action,controller_blob,derived_blob,outcome_blob FROM frames ORDER BY episode_id,timestamp_ns,frame_id"
    ).fetchall()
    total = max(1, len(rows))
    health_loss = deaths = hazard_failures = projectile_hits = 0
    safest = in_spacing = no_safe = escape_frames = 0
    escape_entries = escape_reversals = post_escape_reentries = 0
    previous_escape = False
    previous_action: int | None = None
    separations: list[float] = []
    for action, controller_blob, derived_blob, outcome_blob in rows:
        controller = _from_blob(controller_blob, {})
        derived = _from_blob(derived_blob, {})
        outcomes = _from_blob(outcome_blob, {})
        one_second = _mapping(outcomes.get("1000"))
        quarter_second = _mapping(outcomes.get("250"))
        lost = _number(one_second.get("health_loss")) > 0.0
        health_loss += int(lost)
        deaths += int(bool(one_second.get("dead")))
        actionable = bool(derived.get("hazard_actionable"))
        hazard_failures += int(actionable and (_number(quarter_second.get("health_loss")) > 0.0 or bool(quarter_second.get("dead"))))
        tti = _number(derived.get("nearest_projectile_tti_ms"), float("inf"))
        projectile_hits += int(lost and math.isfinite(tti) and tti <= 250.0)
        safest += int(bool(controller.get("safest_action_selected")))
        in_spacing += int(bool(derived.get("inside_desired_ranged_spacing")))
        no_safe += int(bool(controller.get("no_safe_action")))
        escape = bool(derived.get("escape"))
        escape_frames += int(escape)
        escape_entries += int(escape and not previous_escape)
        if escape and not previous_escape:
            post_escape_reentries += int(
                _number(derived.get("nearest_enemy_distance"), float("inf"))
                < 0.70 * max(1.0, _number(derived.get("ranged_spacing_target"), 170.0))
            )
        if previous_action is not None:
            old = ACTION_VECTORS[MoveAction(int(previous_action))]
            new = ACTION_VECTORS[MoveAction(int(action))]
            escape_reversals += int(old[0] * new[0] + old[1] * new[1] <= -0.5)
        previous_action, previous_escape = int(action), escape
        separation = derived.get("nearest_enemy_distance")
        if separation is not None and math.isfinite(_number(separation, float("inf"))):
            separations.append(float(separation))
    if owns_connection:
        connection.close()
    return {
        "frames": len(rows),
        "health_loss_rate": health_loss / total,
        "death_or_collision_rate": deaths / total,
        "hazard_window_failure_rate": hazard_failures / total,
        "projectile_hit_rate": projectile_hits / total,
        "post_escape_reentry_rate": post_escape_reentries / max(1, escape_entries),
        "escape_entries": escape_entries,
        "escape_direction_reversals": escape_reversals,
        "mean_enemy_separation": sum(separations) / len(separations) if separations else None,
        "minimum_enemy_separation": min(separations) if separations else None,
        "safest_action_selection_rate": safest / total,
        "time_in_escape_fraction": escape_frames / total,
        "desired_ranged_spacing_fraction": in_spacing / total,
        "no_safe_action_by_next_tick_rate": no_safe / total,
        "definitions": {
            "health_loss": "positive observed health loss by the 1000 ms horizon",
            "hazard_window_failure": "actionable frame followed by health loss or death by 250 ms",
            "projectile_hit": "health-loss frame with modeled projectile TTI <= 250 ms",
            "post_escape_reentry": "escape entry whose nearest enemy is already inside 70% of the ranged target",
        },
    }


def replay_frame(path: Path, frame_id: int) -> dict[str, Any]:
    connection = sqlite3.connect(str(path))
    reward_column = "reward_blob" if "reward_blob" in _table_columns(connection, "frames") else "NULL"
    row = connection.execute(
        "SELECT frame_id,episode_id,frame_number,timestamp_ns,phase,wave,action,previous_action,state_blob,input_blob,controller_blob,derived_blob,outcome_blob," + reward_column + " FROM frames WHERE frame_id=?",
        (int(frame_id),),
    ).fetchone()
    connection.close()
    if row is None:
        raise KeyError(f"frame not found: {frame_id}")
    return {
        "frame_id": row[0], "episode_id": row[1], "frame_number": row[2],
        "timestamp_ns": row[3], "phase": row[4], "wave": row[5],
        "action": row[6], "previous_action": row[7],
        "state": _from_blob(row[8], {}), "input": _from_blob(row[9], {}),
        "controller": _from_blob(row[10], {}), "derived": _from_blob(row[11], {}),
        "outcomes": _from_blob(row[12], {}), "reward": _from_blob(row[13], {}),
    }


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(str(path))
    rows = connection.execute(
        "SELECT episode_id,feature_blob,action,input_blob FROM frames WHERE feature_blob IS NOT NULL ORDER BY episode_id,timestamp_ns,frame_id"
    ).fetchall()
    connection.close()
    return [
        {"episode_id": episode, "features": _from_blob(features, []), "action": int(action),
         "input": _from_blob(input_blob, {})}
        for episode, features, action, input_blob in rows
    ]


__all__ = [
    "DATASET_NAME", "DATASET_SCHEMA_VERSION", "HumanDemoWriter", "load_training_rows",
    "normalize_human_input", "quantize_stick", "derive_frame", "replay_frame", "summarize_dataset", "validate_dataset",
]
