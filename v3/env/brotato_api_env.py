"""Gymnasium environment backed by the local Brotato mod API."""

from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from v3.bridge_server import BridgeServer
from v3.combat_policy import (
    ACTION_VECTORS,
    CombatDecisionLogger,
    CombatSafetyShield,
    CrowdRecoveryGuard,
    EnemyContactGuard,
    SafetyDecision,
    movement_transition_metrics,
    projectile_time_to_impact,
)
from v3.config import V3Config
from v3.protocol import (
    MoveAction,
    action_message,
    configure_message,
    reset_message,
    training_pause_message,
)
from v3.reward import ApiRewardEngine
from v3.ui_automation import AutoUiController
from v3.vectorizer import ApiStateVectorizer


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
    projectiles = _state_items(state.get("projectiles"))
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
        server: Optional[BridgeServer] = None,
        vectorizer=None,
        state_hz: Optional[float] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.server = server or BridgeServer(cfg.host, cfg.port)
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
        self.enemy_contact_guard = EnemyContactGuard(
            enabled=bool(getattr(self.vectorizer, "enemy_contact_guard", False)),
            risk_threshold=float(
                getattr(self.vectorizer, "enemy_contact_guard_threshold", 0.22)
            ),
            improvement_margin=float(
                getattr(self.vectorizer, "enemy_contact_guard_margin", 0.08)
            ),
        )
        self.crowd_recovery_guard = CrowdRecoveryGuard(enabled=True)
        self.combat_logger = CombatDecisionLogger(cfg.combat_decision_log)
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
        instantaneous = 1000.0 / max(1.0, float(published_ms - previous))
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
        self.crowd_recovery_guard.reset()
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
        self.last_state = state
        self._last_published_ms = None
        self._effective_state_hz = 0.0
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
        contact_decision = self.enemy_contact_guard.apply(previous_state, requested)
        shield_decision = self.safety_shield.apply(
            previous_state, contact_decision.applied_action
        )
        crowd_decision = self.crowd_recovery_guard.apply(
            previous_state, shield_decision.applied_action
        )
        if contact_decision.overridden:
            decision = SafetyDecision(
                requested,
                crowd_decision.applied_action,
                contact_decision.requested_risk,
                crowd_decision.applied_risk,
            )
        else:
            decision = SafetyDecision(
                requested,
                crowd_decision.applied_action,
                shield_decision.requested_risk,
                crowd_decision.applied_risk,
            )
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
        self.combat_logger.record(
            self.last_state or {},
            decision,
            source="policy_with_safety" if decision.overridden else "policy",
            previous_action=self.previous_action,
        )
        self.sequence += 1
        self.server.send(action_message(normalized, self.sequence), timeout_sec=self.cfg.state_timeout_sec)
        previous_tick = int(self.last_state.get("tick", -1)) if self.last_state else None
        state = self.server.wait_for_state(
            timeout_sec=self.cfg.state_timeout_sec,
            after_tick=previous_tick,
            minimum_sequence=self.sequence,
        )
        state = self._merge_cached_spatial_state(state)
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
        idle_penalty = (
            float(getattr(self.vectorizer, "idle_reward_scale", 0.0))
            if movement["active"] and normalized == int(MoveAction.IDLE)
            else 0.0
        )
        reversal_penalty = (
            float(getattr(self.vectorizer, "reversal_reward_scale", 0.0))
            if movement["reversal"] and threat_max_risk < 0.35
            else 0.0
        )
        low_motion_penalty = (
            float(getattr(self.vectorizer, "low_motion_reward_scale", 0.0))
            if movement["low_motion"]
            else 0.0
        )
        contact_override_penalty = (
            float(getattr(self.vectorizer, "enemy_contact_override_penalty", 0.0))
            if contact_decision.overridden
            else 0.0
        )
        reward = self.reward_engine.step(state)
        reward_components = dict(self.reward_engine.last_components)
        late_focus = bool(self.cfg.late_wave_focus and _state_wave(previous_state) >= 18)
        threat_scale = 3.0 if late_focus else 1.0
        path_risk_penalty = float(
            getattr(self.vectorizer, "path_risk_reward_scale", 0.0)
        ) * selected_path_risk * threat_scale
        boundary_risk_penalty = float(
            getattr(self.vectorizer, "boundary_risk_reward_scale", 0.0)
        ) * selected_boundary_risk * threat_scale
        reward -= path_risk_penalty + boundary_risk_penalty
        reward_components["path_risk"] = -path_risk_penalty
        reward_components["boundary_risk"] = -boundary_risk_penalty
        reward -= (
            idle_penalty
            + reversal_penalty
            + low_motion_penalty
            + contact_override_penalty
        )
        reward_components["idle"] = -idle_penalty
        reward_components["reversal"] = -reversal_penalty
        reward_components["low_motion"] = -low_motion_penalty
        reward_components["contact_override"] = -contact_override_penalty
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
                "death_after_projectile_visible": bool(terminated and projectile_visible),
                "death_after_projectile_hazard": bool(terminated and projectile_hazard),
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
        info = {
            "tick": int(state.get("tick", -1)),
            "phase": state.get("phase"),
            "wave": state.get("wave", {}).get("number", 0),
            "ui_sent": ui_sent,
            "ui_confirmed": ui_confirmed,
            "requested_action": requested,
            "applied_action": normalized,
            "safety_overridden": decision.overridden,
            "enemy_contact_overridden": contact_decision.overridden,
            "enemy_contact_requested_risk": contact_decision.requested_risk,
            "enemy_contact_applied_risk": contact_decision.applied_risk,
            "enemy_contact_override_penalty": contact_override_penalty,
            "crowd_recovery_overridden": crowd_decision.overridden,
            "crowd_recovery_active": self.crowd_recovery_guard.remaining > 0,
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
            "selected_path_risk_penalty": selected_path_risk,
            "late_wave_focus": late_focus,
            "late_threat_scale": threat_scale,
            "effective_state_hz": effective_state_hz,
            "movement_distance": float(movement["distance"]),
            "movement_efficiency": float(movement["efficiency"]),
            "movement_reversal": bool(movement["reversal"]),
            "movement_low_motion": bool(movement["low_motion"]),
            "movement_idle_penalty": idle_penalty,
            "movement_reversal_penalty": reversal_penalty,
            "movement_low_motion_penalty": low_motion_penalty,
            "reward_total": float(reward),
            "reward_components": reward_components,
        }
        info.update(
            {
                "projectile_visible_before_action": projectile_diagnostics["projectile_visible"],
                "projectile_count_before_action": projectile_diagnostics["projectile_count"],
                "projectile_owner_known_count": projectile_diagnostics["projectile_owner_known_count"],
                "projectile_path_present_before_action": projectile_diagnostics["projectile_path_present"],
                "projectile_path_count_before_action": projectile_diagnostics["projectile_path_count"],
                "projectile_path_requested_risk": projectile_diagnostics["projectile_path_requested_risk"],
                "projectile_path_applied_risk": projectile_diagnostics["projectile_path_applied_risk"],
                "projectile_path_safe_action": projectile_diagnostics["projectile_path_safe_action"],
                "projectile_path_risk_margin": projectile_diagnostics["projectile_path_risk_margin"],
                "projectile_path_action_improved": projectile_diagnostics["projectile_path_action_improved"],
                "projectile_predicted_hazard_count": projectile_diagnostics["projectile_predicted_hazard_count"],
                "projectile_nearest_tti": projectile_diagnostics["projectile_nearest_tti"],
                "projectile_nearest_miss_distance": projectile_diagnostics["projectile_nearest_miss_distance"],
                "damage_taken": projectile_diagnostics["damage_taken"],
                "damage_after_projectile_visible": projectile_diagnostics["damage_after_projectile_visible"],
                "damage_after_projectile_hazard": projectile_diagnostics["damage_after_projectile_hazard"],
                "death_after_projectile_visible": projectile_diagnostics["death_after_projectile_visible"],
                "death_after_projectile_hazard": projectile_diagnostics["death_after_projectile_hazard"],
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
