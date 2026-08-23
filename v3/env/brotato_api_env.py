"""Gymnasium environment backed by the local Brotato mod API."""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from v3.bridge_server import BridgeServer
from v3.config import V3Config
from v3.protocol import MoveAction, action_message, reset_message
from v3.reward import ApiRewardEngine
from v3.vectorizer import ApiStateVectorizer


class BrotatoApiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: V3Config, server: Optional[BridgeServer] = None):
        super().__init__()
        self.cfg = cfg
        self.server = server or BridgeServer(cfg.host, cfg.port)
        self.server.start()
        self.vectorizer = ApiStateVectorizer()
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence += 1
        if self.last_state is not None:
            try:
                self.server.send(reset_message(self.sequence))
            except Exception as exc:
                print(f"[v3-env] reset request not delivered: {exc}")
        print("[v3-env] waiting for a combat state; start/resume a wave in Brotato")
        state = self.server.wait_for_state(
            timeout_sec=self.cfg.reset_timeout_sec,
            combat_only=True,
        )
        self.last_state = state
        self.previous_action = int(MoveAction.IDLE)
        self.reward_engine.reset(state)
        observation = self.vectorizer.build(state, self.previous_action)
        return observation, {"tick": int(state.get("tick", -1)), "phase": state.get("phase")}

    def step(self, action):
        normalized = int(MoveAction(int(action)))
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
        truncated = not terminated and state.get("phase") != "combat"
        self.last_state = state
        self.previous_action = normalized
        observation = self.vectorizer.build(state, self.previous_action)
        info = {
            "tick": int(state.get("tick", -1)),
            "phase": state.get("phase"),
            "wave": state.get("wave", {}).get("number", 0),
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        self.server.close()
        super().close()
