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
        self._frame_number = 0
        self._last_build_snapshot: Any = None
        self._last_ui_actions: list[Any] = []
        self._arbiter: FinalActionArbiter | None = None
        self._create_schema()
        self._meta("dataset", DATASET_NAME)
        self._meta("schema_version", DATASET_SCHEMA_VERSION)
        self._meta("session_id", self.session_id)
        self._meta("created_wall_time_ns", time.time_ns())

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, started_ns INTEGER NOT NULL,
                    ended_ns INTEGER, outcome TEXT, start_phase TEXT, end_phase TEXT
                );
                CREATE TABLE IF NOT EXISTS frames (
                    frame_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
                    frame_number INTEGER NOT NULL, timestamp_ns INTEGER NOT NULL,
                    wall_time_ns INTEGER NOT NULL, bridge_timestamp_ms REAL, tick INTEGER,
                    phase TEXT NOT NULL, wave INTEGER, action INTEGER NOT NULL,
                    previous_action INTEGER NOT NULL, action_segment_id TEXT,
                    state_blob BLOB NOT NULL, input_blob BLOB NOT NULL,
                    controller_blob BLOB NOT NULL, derived_blob BLOB NOT NULL,
                    feature_blob BLOB, outcome_blob BLOB
                );
                CREATE TABLE IF NOT EXISTS raw_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
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

    def _meta(self, key: str, value: Any) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                (str(key), json.dumps(value, separators=(",", ":"))),
            )

    def start_episode(self, *, phase: str = "unknown", episode_id: str | None = None) -> str:
        with self.lock:
            self.episode_id = episode_id or f"{self.session_id}:{uuid.uuid4().hex[:12]}"
            self._last_action = int(MoveAction.IDLE)
            self._last_timestamp_ms = None
            self._segment_id = None
            self._frame_number = 0
            self._last_build_snapshot = None
            self._last_ui_actions = []
            self._arbiter = None
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
        with self.lock:
            self.connection.execute(
                "INSERT INTO raw_samples(session_id,timestamp_ns,bridge_timestamp_ms,tick,state_blob,input_blob) VALUES(?,?,?,?,?,?)",
                (self.session_id, time.monotonic_ns(), _number(payload.get("published_at_ms"), None),
                 int(_number(payload.get("tick"), -1)), _json_bytes(payload), _json_bytes(input_value)),
            )
            self.connection.commit()

    def record_frame(self, state: Mapping[str, Any], *, received_ns: int | None = None) -> int:
        with self.lock:
            if self.episode_id is None:
                self.start_episode(phase=str(state.get("phase", "unknown")))
            assert self.episode_id is not None
            action = int(state.get("human_action", state.get("action", 0)))
            if not 0 <= action < ACTION_COUNT:
                action = int(MoveAction.IDLE)
            timestamp_ms = _number(state.get("published_at_ms"), time.monotonic_ns() / 1e6)
            input_value = normalize_human_input(_mapping(state.get("human_input")), action)
            if self._segment_id is None or action != self._last_action:
                if self._segment_id is not None:
                    self.connection.execute(
                        "UPDATE action_segments SET ended_ns=?,duration_ms=? WHERE segment_id=?",
                        (received_ns or time.monotonic_ns(),
                         max(0.0, timestamp_ms - (_number(self._last_timestamp_ms))), self._segment_id),
                    )
                self._segment_id = uuid.uuid4().hex
                self.connection.execute(
                    "INSERT INTO action_segments(segment_id,episode_id,action,started_ns) VALUES(?,?,?,?)",
                    (self._segment_id, self.episode_id, action, received_ns or time.monotonic_ns()),
                )
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
            try:
                from v3.combat_policy import SemanticCombatVectorizer
                features = [round(float(item), 6) for item in SemanticCombatVectorizer().build(state, self._last_action)]
            except Exception:
                features = None
            now_ns = int(received_ns or time.monotonic_ns())
            cursor = self.connection.execute(
                "INSERT INTO frames(episode_id,frame_number,timestamp_ns,wall_time_ns,bridge_timestamp_ms,tick,phase,wave,action,previous_action,action_segment_id,state_blob,input_blob,controller_blob,derived_blob,feature_blob) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.episode_id, self._frame_number, now_ns, time.time_ns(), timestamp_ms,
                 int(_number(state.get("tick"), -1)), str(state.get("phase", "unknown")),
                 int(_number(_mapping(state.get("wave")).get("number"))), action, self._last_action,
                 self._segment_id, _json_bytes(dict(state)), _json_bytes(input_value),
                 _json_bytes(controller), _json_bytes(derived), _json_bytes(features) if features else None),
            )
            frame_id = int(cursor.lastrowid)
            self._record_build_decision(frame_id, state, now_ns)
            self._frame_number += 1
            self._last_action = action
            self._last_timestamp_ms = timestamp_ms
            self.connection.commit()
            return frame_id

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
            if self._segment_id is not None:
                self.connection.execute(
                    "UPDATE action_segments SET ended_ns=?,duration_ms=? WHERE segment_id=? AND ended_ns IS NULL",
                    (time.monotonic_ns(), None, self._segment_id),
                )
            self.connection.execute(
                "UPDATE episodes SET ended_ns=?,outcome=?,end_phase=? WHERE episode_id=?",
                (time.monotonic_ns(), str(outcome), str(end_phase), self.episode_id),
            )
            self.connection.commit()
            self.episode_id = None
            self._segment_id = None

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
                "SELECT frame_id,timestamp_ns,state_blob,derived_blob FROM frames WHERE episode_id=? ORDER BY timestamp_ns",
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


def validate_dataset(path: Path) -> dict[str, Any]:
    """Validate synchronization, input coverage, action alignment, and continuity."""

    connection = sqlite3.connect(str(path))
    frames = connection.execute(
        "SELECT episode_id,timestamp_ns,bridge_timestamp_ms,action,previous_action,action_segment_id,input_blob,derived_blob,outcome_blob FROM frames ORDER BY episode_id,timestamp_ns"
    ).fetchall()
    result: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "schema_version": DATASET_SCHEMA_VERSION,
        "frames": len(frames),
        "raw_samples": int(connection.execute("SELECT COUNT(*) FROM raw_samples").fetchone()[0]),
        "episodes": int(connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]),
        "errors": [], "warnings": [],
    }
    last_by_episode: dict[str, tuple[int, float]] = {}
    raw_available = 0
    gaps = []
    for row in frames:
        episode, timestamp_ns, bridge_ms, action, previous, segment, input_blob, _derived, _outcome = row
        if not 0 <= int(action) < ACTION_COUNT:
            result["errors"].append(f"frame action out of range: {action}")
        if episode in last_by_episode and int(timestamp_ns) <= last_by_episode[episode][0]:
            result["errors"].append(f"non-monotonic timestamp in episode {episode}")
        if episode in last_by_episode:
            delta = int(timestamp_ns) - last_by_episode[episode][0]
            gaps.append(delta / 1e6)
            if delta > 250_000_000:
                result["warnings"].append(f"frame gap {delta / 1e6:.1f} ms in episode {episode}")
        last_by_episode[episode] = (int(timestamp_ns), _number(bridge_ms, 0.0))
        input_value = _from_blob(input_blob, {})
        raw_available += int(bool(input_value.get("raw_available")))
        if int(action) != int(previous) and not segment:
            result["errors"].append("action change without persistence segment")
    result["raw_input_coverage"] = raw_available / len(frames) if frames else 0.0
    result["mean_frame_interval_ms"] = sum(gaps) / len(gaps) if gaps else 0.0
    result["p90_frame_interval_ms"] = sorted(gaps)[min(len(gaps) - 1, int(len(gaps) * 0.9))] if gaps else 0.0
    result["metrics"] = summarize_dataset(path, connection=connection)
    result["ok"] = not result["errors"]
    connection.close()
    return result


def summarize_dataset(
    path: Path, *, connection: sqlite3.Connection | None = None
) -> dict[str, Any]:
    """Return the common outcome metrics used when comparing demonstrations."""

    owns_connection = connection is None
    connection = connection or sqlite3.connect(str(path))
    rows = connection.execute(
        "SELECT action,controller_blob,derived_blob,outcome_blob FROM frames ORDER BY episode_id,timestamp_ns"
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
    row = connection.execute(
        "SELECT frame_id,episode_id,frame_number,timestamp_ns,phase,wave,action,previous_action,state_blob,input_blob,controller_blob,derived_blob,outcome_blob FROM frames WHERE frame_id=?",
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
        "outcomes": _from_blob(row[12], {}),
    }


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(str(path))
    rows = connection.execute(
        "SELECT episode_id,feature_blob,action,input_blob FROM frames WHERE feature_blob IS NOT NULL ORDER BY episode_id,timestamp_ns"
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
