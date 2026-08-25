"""Gymnasium environment backed by the local Brotato mod API."""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from v3.bridge_server import BridgeServer
from v3.combat_policy import (
    CombatDecisionLogger,
    CombatSafetyShield,
    CrowdRecoveryGuard,
    EnemyContactGuard,
    SafetyDecision,
    movement_transition_metrics,
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
        self.reward_engine = ApiRewardEngine()
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
        self.last_state = state
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
        path_risk_penalty = float(
            getattr(self.vectorizer, "path_risk_reward_scale", 0.0)
        ) * selected_path_risk
        boundary_risk_penalty = float(
            getattr(self.vectorizer, "boundary_risk_reward_scale", 0.0)
        ) * selected_boundary_risk
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
            terminated = bool(state.get("dead") or state.get("victory"))
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
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.training_paused:
            try:
                self.set_training_paused(False)
            except Exception:
                pass
        self.server.close()
        super().close()
