"""Gymnasium environment backed by the local Brotato mod API."""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from v3.bridge_server import BridgeServer
from v3.combat_policy import CombatDecisionLogger, CombatSafetyShield
from v3.config import V3Config
from v3.protocol import MoveAction, action_message, configure_message, reset_message
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
        self.combat_logger = CombatDecisionLogger(cfg.combat_decision_log)
        self.ui_controller = AutoUiController(
            max_shop_buys=cfg.max_shop_buys,
            max_shop_rerolls=cfg.max_shop_rerolls,
            build_profile=cfg.ui_build_profile,
            ui_model_path=cfg.ui_model_path,
            decision_log_path=cfg.ui_decision_log,
        )

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
        observation = self.vectorizer.build(state, self.previous_action)
        return observation, {
            "tick": int(state.get("tick", -1)),
            "phase": state.get("phase"),
            "wave": state.get("wave", {}).get("number", 0),
            "ui_sent": ui_sent,
            "ui_confirmed": ui_confirmed,
        }

    def step(self, action):
        requested = int(MoveAction(int(action)))
        decision = self.safety_shield.apply(self.last_state or {}, requested)
        normalized = decision.applied_action
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
        reward = self.reward_engine.step(state)
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
            state = result.state
            ui_sent = result.sent_roles
            ui_confirmed = result.confirmed_roles
            terminated = bool(state.get("dead") or state.get("victory"))
        truncated = not terminated and state.get("phase") != "combat"
        self.last_state = state
        self.previous_action = normalized
        observation = self.vectorizer.build(state, self.previous_action)
        info = {
            "tick": int(state.get("tick", -1)),
            "phase": state.get("phase"),
            "wave": state.get("wave", {}).get("number", 0),
            "ui_sent": ui_sent,
            "ui_confirmed": ui_confirmed,
            "requested_action": requested,
            "applied_action": normalized,
            "safety_overridden": decision.overridden,
            "requested_risk": decision.requested_risk,
            "applied_risk": decision.applied_risk,
            "materials": int(state.get("counters", {}).get("materials", 0)),
            "health_fraction": float(state.get("player", {}).get("health", 0.0))
            / max(1.0, float(state.get("player", {}).get("max_health", 1.0))),
            "enemy_count": len(state.get("enemies", [])),
            "projectile_count": len(state.get("projectiles", [])),
            "attack_indicator_count": len(state.get("attack_indicators", [])),
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        self.server.close()
        super().close()
