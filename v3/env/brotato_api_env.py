"""Gymnasium compatibility entrypoint backed by the active v4 runtime."""

import time
from collections import deque
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from brotato_ai.bridge.client import BridgeClient
from brotato_ai.control import (
    CombatDecisionPipeline,
    CombatSafetyShield,
    CrowdRecoveryGuard,
    FinalActionWriter,
)
from brotato_ai.data.recorder import DecisionTraceLogger
from brotato_ai.domain.actions import ACTION_VECTORS, MoveAction
from brotato_ai.domain.state import normalize_state
from v3.combat_policy import (
    center_stagnation_signal,
    movement_transition_metrics,
    projectile_time_to_impact,
)
from v3.config import V3Config
from v3.protocol import (
    configure_message,
    reset_message,
    training_pause_message,
)
from v3.reward import ApiRewardEngine
from v3.ui_automation import AutoUiController
from v3.vectorizer import ApiStateVectorizer
from brotato_ai.telemetry import percentile, risk_diagnostics, reward_time_scale


def _state_wave(state: Mapping[str, Any]) -> float:
    wave = state.get("wave", {}) if isinstance(state, Mapping) else {}
    try:
        return float(wave.get("number", 0.0)) if isinstance(wave, Mapping) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _state_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _state_risks(value: Any) -> list[float]:
    result = [0.0] * len(MoveAction)
    if not isinstance(value, (list, tuple)):
        return result
    for index, raw_value in enumerate(value[:len(result)]):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            result[index] = float(np.clip(value, 0.0, 1.0))
    return result


def _projectile_diagnostics(
    state: Mapping[str, Any],
    requested_action: int,
    applied_action: int,
) -> dict[str, Any]:
    """Summarize projectile visibility, forecast danger, and chosen lane."""

    player = state.get("player", {}) if isinstance(state, Mapping) else {}
    player = player if isinstance(player, Mapping) else {}
    position = player.get("position", {})
    position = position if isinstance(position, Mapping) else {}
    px = float(position.get("x", 0.0) or 0.0)
    py = float(position.get("y", 0.0) or 0.0)
    combat = state.get("combat", {}) if isinstance(state, Mapping) else {}
    combat = combat if isinstance(combat, Mapping) else {}
    try:
        player_speed = max(150.0, float(combat.get("move_speed", 300.0)))
    except (TypeError, ValueError):
        player_speed = 300.0
    all_projectiles = _state_items(state.get("projectiles"))
    projectiles = [
        projectile
        for projectile in all_projectiles
        if bool(projectile.get("hostile", True))
    ]
    enemies = _state_items(state.get("enemies"))
    owner_ids = {
        str(enemy.get("runtime_id"))
        for enemy in enemies
        if str(enemy.get("runtime_id", ""))
    }
    paths = state.get("projectile_paths", {})
    paths = paths if isinstance(paths, Mapping) else {}
    risks = _state_risks(paths.get("action_risk"))
    requested_risk = risks[int(requested_action)]
    applied_risk = risks[int(applied_action)]
    safe_action = int(np.argmin(risks)) if risks else int(MoveAction.IDLE)
    hazard_count = 0
    nearest_tti = None
    nearest_miss = None
    for projectile in projectiles:
        tti, miss_distance = projectile_time_to_impact(
            projectile,
            (px, py),
            ACTION_VECTORS[MoveAction(int(applied_action))],
            player_speed,
        )
        try:
            radius = max(8.0, float(projectile.get("radius", 12.0))) + 42.0
        except (TypeError, ValueError):
            radius = 54.0
        if tti <= 0.8 and miss_distance <= radius:
            hazard_count += 1
        if nearest_tti is None or tti < nearest_tti:
            nearest_tti = float(tti)
            nearest_miss = float(miss_distance)
    return {
        "projectile_visible": bool(projectiles),
        "projectile_count": len(projectiles),
        "projectile_total_count": len(all_projectiles),
        "projectile_hostile_count": len(projectiles),
        "projectile_owner_known_count": sum(
            str(projectile.get("owner_runtime_id", "")) in owner_ids
            for projectile in projectiles
        ),
        "projectile_path_present": bool(paths),
        "projectile_path_count": int(paths.get("count", 0) or 0),
        "projectile_path_requested_risk": requested_risk,
        "projectile_path_applied_risk": applied_risk,
        "projectile_path_safe_action": safe_action,
        "projectile_path_risk_margin": requested_risk - applied_risk,
        "projectile_path_action_improved": applied_risk + 1e-6 < requested_risk,
        "projectile_predicted_hazard_count": hazard_count,
        "projectile_nearest_tti": nearest_tti if nearest_tti is not None else -1.0,
        "projectile_nearest_miss_distance": nearest_miss if nearest_miss is not None else -1.0,
    }


class BrotatoApiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: V3Config,
        server: Optional[BridgeClient] = None,
        vectorizer=None,
        state_hz: Optional[float] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.server = server or BridgeClient(cfg.host, cfg.port)
        self.server.start()
        self.vectorizer = vectorizer or ApiStateVectorizer()
        self.state_hz = float(state_hz) if state_hz is not None else None
        self._configured_session = ""
        self.reward_engine = ApiRewardEngine(late_wave_focus=cfg.late_wave_focus)
        self.action_space = spaces.Discrete(len(MoveAction))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.vectorizer.observation_size,),
            dtype=np.float32,
        )
        self.sequence = 0
        self.previous_action = int(MoveAction.IDLE)
        self.last_state = None
        self.safety_shield = CombatSafetyShield(enabled=cfg.safety_shield)
        self.crowd_recovery_guard = CrowdRecoveryGuard(
            enabled=True,
            shield=self.safety_shield,
        )
        self.decision_pipeline = CombatDecisionPipeline(
            safety_shield=self.safety_shield,
            crowd_recovery_guard=self.crowd_recovery_guard,
        )
        self.action_writer = FinalActionWriter(
            self.server, timeout_sec=self.cfg.state_timeout_sec
        )
        self.combat_logger = DecisionTraceLogger(cfg.combat_decision_log)
        self.ui_controller = AutoUiController(
            max_shop_buys=cfg.max_shop_buys,
            max_shop_rerolls=cfg.max_shop_rerolls,
            build_profile=cfg.ui_build_profile,
            ui_model_path=cfg.ui_model_path,
            decision_log_path=cfg.ui_decision_log,
        )
        self.training_paused = False
        self._last_published_ms: Optional[int] = None
        self._effective_state_hz = 0.0
        self._last_state_interval_ms = 0.0
        self._last_control_started: Optional[float] = None
        self._state_intervals_ms = deque(maxlen=512)
        self._control_intervals_ms = deque(maxlen=512)
        self._cached_projectile_paths: dict[str, Any] = {}
        self._cached_arena_grid: dict[str, Any] = {}

    def _merge_cached_spatial_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Reuse large bridge payloads when the bridge sends a lightweight tick."""
        merged = dict(state)
        if state.get("phase") != "combat":
            self._cached_projectile_paths = {}
            self._cached_arena_grid = {}
            return merged

        projectile_paths = state.get("projectile_paths")
        if isinstance(projectile_paths, Mapping) and projectile_paths:
            self._cached_projectile_paths = dict(projectile_paths)
        elif self._cached_projectile_paths:
            merged["projectile_paths"] = self._cached_projectile_paths

        arena_grid = state.get("arena_grid")
        if isinstance(arena_grid, Mapping) and arena_grid.get("enemy"):
            self._cached_arena_grid = dict(arena_grid)
        elif self._cached_arena_grid:
            merged["arena_grid"] = self._cached_arena_grid
        return merged

    def _observe_state_rate(self, state: Mapping[str, Any]) -> float:
        try:
            published_ms = int(state.get("published_at_ms", -1))
        except (TypeError, ValueError):
            published_ms = -1
        if published_ms < 0:
            return self._effective_state_hz
        previous = self._last_published_ms
        self._last_published_ms = published_ms
        if previous is None or published_ms <= previous:
            return self._effective_state_hz
        self._last_state_interval_ms = max(1.0, float(published_ms - previous))
        if not hasattr(self, "_state_intervals_ms"):
            self._state_intervals_ms = deque(maxlen=512)
        self._state_intervals_ms.append(self._last_state_interval_ms)
        instantaneous = 1000.0 / self._last_state_interval_ms
        if self._effective_state_hz <= 0.0:
            self._effective_state_hz = instantaneous
        else:
            self._effective_state_hz = 0.90 * self._effective_state_hz + 0.10 * instantaneous
        return self._effective_state_hz

    def set_training_paused(self, paused: bool) -> None:
        """Freeze game simulation while PPO updates its network weights."""

        normalized = bool(paused)
        if normalized == self.training_paused:
            return
        self.server.send(
            training_pause_message(normalized),
            timeout_sec=self.cfg.state_timeout_sec,
        )
        self.training_paused = normalized

    def _configure_state_rate(self, state) -> None:
        if self.state_hz is None:
            return
        session = str(state.get("session", ""))
        if session == self._configured_session:
            return
        hello = self.server.last_hello or {}
        if "configurable_state_rate" not in hello.get("capabilities", []):
            raise RuntimeError("Bridge 0.3.3+ is required for configured PPO state rate")
        self.server.send(configure_message(state_hz=self.state_hz))
        self._configured_session = session
        print(f"[v3-env] bridge state rate configured={self.state_hz:g} Hz")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence += 1
        if self.last_state is not None:
            try:
                self.server.send(reset_message(self.sequence))
            except Exception as exc:
                print(f"[v3-env] reset request not delivered: {exc}")
        self.ui_controller.reset_episode()
        self.decision_pipeline.reset()
        print("[v3-env] waiting for combat; structured menu automation is active")
        state = self.server.wait_for_state(
            timeout_sec=self.cfg.reset_timeout_sec,
        )
        self._configure_state_rate(state)
        ui_sent = []
        ui_confirmed = []
        if self.cfg.automate_menus and state.get("phase") != "combat":
            result = self.ui_controller.advance(
                self.server,
                state,
                self.sequence,
                self.cfg.reset_timeout_sec,
                allow_restart=True,
            )
            state = result.state
            self.sequence = result.sequence
            ui_sent = result.sent_roles
            ui_confirmed = result.confirmed_roles
        elif state.get("phase") != "combat":
            state = self.server.wait_for_state(
                timeout_sec=self.cfg.reset_timeout_sec,
                after_tick=int(state.get("tick", -1)),
                combat_only=True,
            )
        self._cached_projectile_paths = {}
        self._cached_arena_grid = {}
        state = self._merge_cached_spatial_state(state)
        state = normalize_state(state)
        self.last_state = state
        self._last_published_ms = None
        self._effective_state_hz = 0.0
        self._last_state_interval_ms = 0.0
        self._last_control_started = None
        self._state_intervals_ms.clear()
        self._control_intervals_ms.clear()
        self._observe_state_rate(state)
        self.previous_action = int(MoveAction.IDLE)
        self.reward_engine.reset(state)
        reset_vectorizer = getattr(self.vectorizer, "reset", None)
        if callable(reset_vectorizer):
            reset_vectorizer(state)
        observation = self.vectorizer.build(state, self.previous_action)
        return observation, {
            "tick": int(state.get("tick", -1)),
            "phase": state.get("phase"),
            "wave": state.get("wave", {}).get("number", 0),
            "ui_sent": ui_sent,
            "ui_confirmed": ui_confirmed,
            "effective_state_hz": self._effective_state_hz,
        }

    def step(self, action):
        previous_state = self.last_state or {}
        previous_action = self.previous_action
        requested = int(MoveAction(int(action)))
        control_started = time.monotonic()
        control_interval_ms = (
            0.0
            if self._last_control_started is None
            else max(0.0, (control_started - self._last_control_started) * 1000.0)
        )
        self._last_control_started = control_started
        if control_interval_ms > 0.0:
            self._control_intervals_ms.append(control_interval_ms)
        decision_trace = self.decision_pipeline.apply(
            previous_state,
            requested,
            previous_action=previous_action,
            state_interval_ms=self._last_state_interval_ms,
            control_interval_ms=control_interval_ms,
        )
        decision = decision_trace.decision
        normalized = decision.applied_action
        projectile_diagnostics = _projectile_diagnostics(
            previous_state,
            requested,
            normalized,
        )
        previous_paths = (self.last_state or {}).get("projectile_paths", {})
        projectile_risks = (
            previous_paths.get("action_risk", [])
            if isinstance(previous_paths, dict)
            else []
        )
        enemy_risks = (
            previous_paths.get("enemy_action_risk", [])
            if isinstance(previous_paths, dict)
            else []
        )
        boundary_risks = (
            previous_paths.get("boundary_action_risk", [])
            if isinstance(previous_paths, dict)
            else []
        )

        def _risk_vector(values) -> list[float]:
            result = [0.0] * len(MoveAction)
            if not isinstance(values, (list, tuple)):
                return result
            for index, raw_value in enumerate(values[:len(result)]):
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    result[index] = float(np.clip(value, 0.0, 1.0))
            return result

        def _selected_risk(values) -> float:
            return _risk_vector(values)[normalized]

        selected_projectile_risk = _selected_risk(projectile_risks)
        selected_enemy_risk = _selected_risk(enemy_risks)
        selected_boundary_risk = _selected_risk(boundary_risks)
        selected_path_risk = min(
            1.0,
            selected_projectile_risk + selected_enemy_risk + selected_boundary_risk,
        )
        self.combat_logger.record(decision_trace)
        self.sequence += 1
        self.action_writer.write(decision_trace, self.sequence)
        previous_tick = int(self.last_state.get("tick", -1)) if self.last_state else None
        state = self.server.wait_for_state(
            timeout_sec=self.cfg.state_timeout_sec,
            after_tick=previous_tick,
            minimum_sequence=self.sequence,
        )
        state = self._merge_cached_spatial_state(state)
        state = normalize_state(state)
        effective_state_hz = self._observe_state_rate(state)
        movement = movement_transition_metrics(
            previous_state,
            state,
            previous_action,
            normalized,
            state_hz=self.state_hz or 12.0,
        )
        threat_max_risk = max(
            _risk_vector(projectile_risks) + _risk_vector(enemy_risks),
            default=0.0,
        )
        center_threat_risk = max(
            threat_max_risk,
            float(decision_trace.requested_risk.total),
        )
        center_stagnation = center_stagnation_signal(
            previous_state,
            state,
            threat_risk=center_threat_risk,
            radius=float(
                getattr(self.vectorizer, "center_stagnation_radius", 0.0)
            ),
            threat_exemption=float(
                getattr(self.vectorizer, "center_stagnation_threat_exemption", 0.0)
            ),
        ) and not decision_trace.recovery_active
        dense_scale = reward_time_scale(
            self._last_state_interval_ms,
            self.state_hz or 24.0,
        )
        idle_penalty = (
            float(getattr(self.vectorizer, "idle_reward_scale", 0.0)) * dense_scale
            if movement["active"] and normalized == int(MoveAction.IDLE)
            else 0.0
        )
        reversal_penalty = (
            float(getattr(self.vectorizer, "reversal_reward_scale", 0.0)) * dense_scale
            if movement["reversal"] and threat_max_risk < 0.35
            else 0.0
        )
        low_motion_penalty = (
            float(getattr(self.vectorizer, "low_motion_reward_scale", 0.0)) * dense_scale
            if movement["low_motion"]
            else 0.0
        )
        contact_override_penalty = (
            float(getattr(self.vectorizer, "enemy_contact_override_penalty", 0.0)) * dense_scale
            if decision_trace.enemy_contact_overridden
            else 0.0
        )
        center_stagnation_penalty = (
            float(getattr(self.vectorizer, "center_stagnation_reward_scale", 0.0)) * dense_scale
            if center_stagnation
            else 0.0
        )
        reward = self.reward_engine.step(state, dense_scale=dense_scale)
        reward_components = dict(self.reward_engine.last_components)
        late_focus = bool(self.cfg.late_wave_focus and _state_wave(previous_state) >= 18)
        threat_scale = 3.0 if late_focus else 1.0
        path_risk_penalty = float(
            getattr(self.vectorizer, "path_risk_reward_scale", 0.0)
        ) * selected_path_risk * threat_scale * dense_scale
        boundary_risk_penalty = float(
            getattr(self.vectorizer, "boundary_risk_reward_scale", 0.0)
        ) * selected_boundary_risk * threat_scale * dense_scale
        reward -= path_risk_penalty + boundary_risk_penalty
        reward_components["path_risk"] = -path_risk_penalty
        reward_components["boundary_risk"] = -boundary_risk_penalty
        reward -= (
            idle_penalty
            + reversal_penalty
            + low_motion_penalty
            + contact_override_penalty
            + center_stagnation_penalty
        )
        reward_components["idle"] = -idle_penalty
        reward_components["reversal"] = -reversal_penalty
        reward_components["low_motion"] = -low_motion_penalty
        reward_components["contact_override"] = -contact_override_penalty
        reward_components["center_stagnation"] = -center_stagnation_penalty
        terminated = bool(state.get("dead") or state.get("victory"))
        ui_sent = []
        ui_confirmed = []
        if self.cfg.automate_menus and not terminated and state.get("phase") != "combat":
            result = self.ui_controller.advance(
                self.server,
                state,
                self.sequence,
                self.cfg.reset_timeout_sec,
            )
            self.sequence = result.sequence
            for menu_state in result.states:
                reward += self.reward_engine.step(menu_state)
                for key, value in self.reward_engine.last_components.items():
                    reward_components[key] = reward_components.get(key, 0.0) + float(value)
            state = result.state
            ui_sent = result.sent_roles
            ui_confirmed = result.confirmed_roles
            state = self._merge_cached_spatial_state(state)
            state = normalize_state(state)
            terminated = bool(state.get("dead") or state.get("victory"))
        previous_player = previous_state.get("player", {})
        current_player = state.get("player", {})
        try:
            health_before = float(previous_player.get("health", 0.0))
        except (AttributeError, TypeError, ValueError):
            health_before = 0.0
        try:
            health_after = float(current_player.get("health", 0.0))
        except (AttributeError, TypeError, ValueError):
            health_after = health_before
        damage_taken = max(0.0, health_before - health_after)
        projectile_visible = bool(projectile_diagnostics["projectile_visible"])
        projectile_hazard = projectile_diagnostics["projectile_predicted_hazard_count"] > 0
        projectile_diagnostics.update(
            {
                "damage_taken": damage_taken,
                "damage_after_projectile_visible": damage_taken if projectile_visible else 0.0,
                "damage_after_projectile_hazard": damage_taken if projectile_hazard else 0.0,
                # Victory is terminal but must never be counted as a death.
                "death_after_projectile_visible": bool(state.get("dead") and projectile_visible),
                "death_after_projectile_hazard": bool(state.get("dead") and projectile_hazard),
                "projectile_tti_exposed": bool(
                    projectile_diagnostics["projectile_nearest_tti"] >= 0.0
                    and projectile_diagnostics["projectile_nearest_tti"] <= 0.8
                ),
                "projectile_hazard_exposed": bool(projectile_hazard),
            }
        )
        truncated = not terminated and state.get("phase") != "combat"
        self.last_state = state
        self.previous_action = normalized
        observation = self.vectorizer.build(state, self.previous_action)
        projectile_paths = state.get("projectile_paths", {})
        path_risks = (
            projectile_paths.get("action_risk", [])
            if isinstance(projectile_paths, dict)
            else []
        )
        finite_risks = _risk_vector(path_risks)
        enemy_path_risks = (
            projectile_paths.get("enemy_action_risk", [])
            if isinstance(projectile_paths, dict)
            else []
        )
        finite_enemy_risks = _risk_vector(enemy_path_risks)
        boundary_path_risks = (
            projectile_paths.get("boundary_action_risk", [])
            if isinstance(projectile_paths, dict)
            else []
        )
        finite_boundary_risks = _risk_vector(boundary_path_risks)
        def _path_stats(values):
            finite = _risk_vector(values)
            unsafe = sum(value >= 0.65 for value in finite)
            return (
                min(finite, default=0.0),
                unsafe,
                unsafe / max(1, len(finite)),
            )
        projectile_path_min, projectile_path_unsafe, projectile_path_fraction = _path_stats(path_risks)
        enemy_path_min, enemy_path_unsafe, enemy_path_fraction = _path_stats(enemy_path_risks)
        boundary_path_min, boundary_path_unsafe, boundary_path_fraction = _path_stats(boundary_path_risks)
        projectile_path_pre_min, projectile_path_pre_unsafe, projectile_path_pre_fraction = _path_stats(projectile_risks)
        enemy_path_pre_min, enemy_path_pre_unsafe, enemy_path_pre_fraction = _path_stats(enemy_risks)
        boundary_path_pre_min, boundary_path_pre_unsafe, boundary_path_pre_fraction = _path_stats(boundary_risks)
        risk_stats = risk_diagnostics(
            decision_trace.all_risks,
            requested,
        )
        state_interval_values = list(self._state_intervals_ms)
        control_interval_values = list(self._control_intervals_ms)
        wave_clear = float(reward_components.get("wave_clear", 0.0)) > 0.0
        dead = bool(state.get("dead"))
        victory = bool(state.get("victory"))
        info = {
            "tick": int(state.get("tick", -1)),
            "phase": state.get("phase"),
            "wave": state.get("wave", {}).get("number", 0),
            "ui_sent": ui_sent,
            "ui_confirmed": ui_confirmed,
            "requested_action": requested,
            "applied_action": normalized,
            "safety_overridden": decision.overridden,
            "hazard_overridden": decision_trace.hazard_overridden,
            "hazard_source": decision_trace.source,
            "hazard_action": decision_trace.hazard_decision.applied_action,
            "hazard_requested_risk": decision_trace.requested_risk.total,
            "hazard_stage_applied_risk": decision_trace.hazard_risk.total,
            "hazard_applied_risk": decision_trace.applied_risk.total,
            "hazard_risk_reduction": (
                decision_trace.requested_risk.total
                - decision_trace.applied_risk.total
            ),
            "hazard_enemy_risk": decision_trace.requested_risk.enemy_total,
            "hazard_projectile_risk": decision_trace.requested_risk.projectile_total,
            "hazard_indicator_risk": decision_trace.requested_risk.indicator,
            "hazard_boundary_risk": decision_trace.requested_risk.boundary_total,
            "hazard_applied_enemy_risk": decision_trace.applied_risk.enemy_total,
            "hazard_applied_projectile_risk": decision_trace.applied_risk.projectile_total,
            "hazard_applied_indicator_risk": decision_trace.applied_risk.indicator,
            "hazard_applied_boundary_risk": decision_trace.applied_risk.boundary_total,
            "enemy_contact_overridden": decision_trace.enemy_contact_overridden,
            "enemy_contact_requested_risk": decision_trace.requested_risk.enemy_total,
            "enemy_contact_applied_risk": decision_trace.applied_risk.enemy_total,
            "enemy_contact_override_penalty": contact_override_penalty,
            "crowd_recovery_overridden": decision_trace.recovery_overridden,
            "crowd_recovery_active": decision_trace.recovery_active,
            "hazard_state_interval_ms": decision_trace.state_interval_ms,
            "hazard_control_interval_ms": decision_trace.control_interval_ms,
            "requested_risk": decision.requested_risk,
            "applied_risk": decision.applied_risk,
            "materials": int(state.get("counters", {}).get("materials", 0)),
            "health_fraction": float(state.get("player", {}).get("health", 0.0))
            / max(1.0, float(state.get("player", {}).get("max_health", 1.0))),
            "enemy_count": len(state.get("enemies", [])),
            "projectile_count": len(state.get("projectiles", [])),
            "attack_indicator_count": len(state.get("attack_indicators", [])),
            "projectile_path_count": int(projectile_paths.get("count", 0))
            if isinstance(projectile_paths, dict)
            else 0,
            "projectile_path_max_risk": max(finite_risks, default=0.0),
            "projectile_path_action_risk": (
                finite_risks[normalized] if normalized < len(finite_risks) else 0.0
            ),
            "enemy_path_max_risk": max(finite_enemy_risks, default=0.0),
            "enemy_path_action_risk": (
                finite_enemy_risks[normalized]
                if normalized < len(finite_enemy_risks)
                else 0.0
            ),
            "boundary_path_max_risk": max(finite_boundary_risks, default=0.0),
            "boundary_path_action_risk": (
                finite_boundary_risks[normalized]
                if normalized < len(finite_boundary_risks)
                else 0.0
            ),
            "projectile_path_min_risk": projectile_path_min,
            "projectile_path_unsafe_action_count": projectile_path_unsafe,
            "projectile_path_unsafe_action_fraction": projectile_path_fraction,
            "enemy_path_min_risk": enemy_path_min,
            "enemy_path_unsafe_action_count": enemy_path_unsafe,
            "enemy_path_unsafe_action_fraction": enemy_path_fraction,
            "boundary_path_min_risk": boundary_path_min,
            "boundary_path_unsafe_action_count": boundary_path_unsafe,
            "boundary_path_unsafe_action_fraction": boundary_path_fraction,
            "projectile_path_min_risk_before_action": projectile_path_pre_min,
            "projectile_path_unsafe_action_count_before_action": projectile_path_pre_unsafe,
            "projectile_path_unsafe_action_fraction_before_action": projectile_path_pre_fraction,
            "enemy_path_min_risk_before_action": enemy_path_pre_min,
            "enemy_path_unsafe_action_count_before_action": enemy_path_pre_unsafe,
            "enemy_path_unsafe_action_fraction_before_action": enemy_path_pre_fraction,
            "boundary_path_min_risk_before_action": boundary_path_pre_min,
            "boundary_path_unsafe_action_count_before_action": boundary_path_pre_unsafe,
            "boundary_path_unsafe_action_fraction_before_action": boundary_path_pre_fraction,
            "selected_path_risk_penalty": selected_path_risk,
            "minimum_action_risk": risk_stats["minimum_action_risk"],
            "unsafe_action_count": risk_stats["unsafe_action_count"],
            "unsafe_action_fraction": risk_stats["unsafe_action_fraction"],
            "requested_to_minimum_regret": risk_stats["requested_to_minimum_regret"],
            "late_wave_focus": late_focus,
            "late_threat_scale": threat_scale,
            "effective_state_hz": effective_state_hz,
            "state_interval_p50_ms": percentile(state_interval_values, 0.50),
            "state_interval_p95_ms": percentile(state_interval_values, 0.95),
            "state_interval_p99_ms": percentile(state_interval_values, 0.99),
            "control_interval_p50_ms": percentile(control_interval_values, 0.50),
            "control_interval_p95_ms": percentile(control_interval_values, 0.95),
            "control_interval_p99_ms": percentile(control_interval_values, 0.99),
            "stale_state_count": int(getattr(self.server, "stale_state_count", 0)),
            "dropped_state_count": int(getattr(self.server, "dropped_state_count", 0)),
            "state_tick_gap": int(getattr(self.server, "last_tick_gap", 0)),
            "reward_time_scale": dense_scale,
            "movement_distance": float(movement["distance"]),
            "movement_efficiency": float(movement["efficiency"]),
            "movement_reversal": bool(movement["reversal"]),
            "movement_low_motion": bool(movement["low_motion"]),
            "movement_idle_penalty": idle_penalty,
            "movement_reversal_penalty": reversal_penalty,
            "movement_low_motion_penalty": low_motion_penalty,
            "movement_center_stagnation": bool(center_stagnation),
            "movement_center_stagnation_penalty": center_stagnation_penalty,
            "reward_total": float(reward),
            "reward_components": reward_components,
        }
        info.update(
            {
                "projectile_visible_before_action": projectile_diagnostics["projectile_visible"],
                "projectile_count_before_action": projectile_diagnostics["projectile_count"],
                "projectile_total_count_before_action": projectile_diagnostics["projectile_total_count"],
                "projectile_hostile_count_before_action": projectile_diagnostics["projectile_hostile_count"],
                "projectile_owner_known_count": projectile_diagnostics["projectile_owner_known_count"],
                "projectile_path_present_before_action": projectile_diagnostics["projectile_path_present"],
                "projectile_path_count_before_action": projectile_diagnostics["projectile_path_count"],
                "projectile_path_requested_risk": projectile_diagnostics["projectile_path_requested_risk"],
                "projectile_path_applied_risk": projectile_diagnostics["projectile_path_applied_risk"],
                "projectile_path_safe_action": projectile_diagnostics["projectile_path_safe_action"],
                "projectile_path_risk_margin": projectile_diagnostics["projectile_path_risk_margin"],
                "projectile_path_action_improved": projectile_diagnostics["projectile_path_action_improved"],
                "projectile_tti_exposed": projectile_diagnostics["projectile_tti_exposed"],
                "projectile_hazard_exposed": projectile_diagnostics["projectile_hazard_exposed"],
                "hazard_overridden": decision_trace.hazard_overridden,
                "hazard_source": decision_trace.source,
                "hazard_action": decision_trace.hazard_decision.applied_action,
                "hazard_requested_risk": decision_trace.requested_risk.total,
                "hazard_stage_applied_risk": decision_trace.hazard_risk.total,
                "hazard_applied_risk": decision_trace.applied_risk.total,
                "hazard_risk_reduction": (
                    decision_trace.requested_risk.total
                    - decision_trace.applied_risk.total
                ),
                "hazard_enemy_risk": decision_trace.requested_risk.enemy_total,
                "hazard_projectile_risk": decision_trace.requested_risk.projectile_total,
                "hazard_indicator_risk": decision_trace.requested_risk.indicator,
                "hazard_boundary_risk": decision_trace.requested_risk.boundary_total,
                "hazard_applied_enemy_risk": decision_trace.applied_risk.enemy_total,
                "hazard_applied_projectile_risk": decision_trace.applied_risk.projectile_total,
                "hazard_applied_indicator_risk": decision_trace.applied_risk.indicator,
                "hazard_applied_boundary_risk": decision_trace.applied_risk.boundary_total,
                "projectile_predicted_hazard_count": projectile_diagnostics["projectile_predicted_hazard_count"],
                "projectile_nearest_tti": projectile_diagnostics["projectile_nearest_tti"],
                "projectile_nearest_miss_distance": projectile_diagnostics["projectile_nearest_miss_distance"],
                "damage_taken": projectile_diagnostics["damage_taken"],
                "damage_after_projectile_visible": projectile_diagnostics["damage_after_projectile_visible"],
                "damage_after_projectile_hazard": projectile_diagnostics["damage_after_projectile_hazard"],
                "death_after_projectile_visible": projectile_diagnostics["death_after_projectile_visible"],
                "death_after_projectile_hazard": projectile_diagnostics["death_after_projectile_hazard"],
                "projectile_tti_exposed": projectile_diagnostics["projectile_tti_exposed"],
                "projectile_hazard_exposed": projectile_diagnostics["projectile_hazard_exposed"],
            }
        )
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.training_paused:
            try:
                self.set_training_paused(False)
            except Exception:
                pass
        self.server.close()
        super().close()
