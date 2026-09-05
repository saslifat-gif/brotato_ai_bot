"""Gymnasium compatibility entrypoint backed by the active v4 runtime."""

import dataclasses
import time
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
from brotato_ai.performance import RuntimeProfiler
from brotato_ai.policy.human_action import HumanProposal
from brotato_ai.policy.modes import PolicyMode, parse_policy_mode
from v4.combat_base import (
    movement_transition_metrics,
    projectile_time_to_impact,
)
from v4.config import RuntimeConfig
from v4.protocol import (
    configure_message,
    reset_message,
    training_pause_message,
)
from v4.reward import ApiRewardEngine
from v4.ui_automation import AutoUiController
from v4.vectorizer import ApiStateVectorizer


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


def _build_human_policy_stack(cfg: RuntimeConfig, profiler: RuntimeProfiler):
    """Build the optional learned-human-policy stack for the configured mode.

    Returns ``(policy_mode, human_policy, human_builder, hybrid_controller)``.
    Any load failure degrades the mode to HANDCRAFTED with a printed warning;
    the production controller never depends on learned inference being alive.
    """

    mode = parse_policy_mode(getattr(cfg, "policy_mode", None))
    if mode is PolicyMode.HANDCRAFTED:
        return mode, None, None, None
    try:
        from brotato_ai.policy.features import HumanPolicyFeatureBuilder
        from brotato_ai.policy.human_action import EventHumanActionPolicy
        from brotato_ai.policy.hybrid import HumanHybridController

        builder = HumanPolicyFeatureBuilder()
        policy = EventHumanActionPolicy.load(cfg.human_model_path)
        hybrid = HumanHybridController(
            decision_interval_ms=float(getattr(cfg, "human_decision_interval_ms", 438.0)),
            hold_prior_ms=float(getattr(cfg, "human_hold_prior_ms", 438.0)),
            min_confidence=float(getattr(cfg, "human_confidence_threshold", 0.0)),
            full_learned=(mode is PolicyMode.EXPERIMENTAL_FULL_LEARNED),
        )
        print(
            f"[v4-env] human policy mode={mode.value} model={cfg.human_model_path} "
            f"schema={policy.feature_schema_version} "
            f"heldout_next_action={policy.metrics.get('teacher_forced', {}).get('next_action_accuracy_on_true_change')}"
        )
        return mode, policy, builder, hybrid
    except Exception as exc:
        profiler.count("human_policy_load_failure")
        print(
            f"[v4-env] WARNING: {mode.value} requested but the human policy failed to "
            f"load ({exc}); falling back to HANDCRAFTED"
        )
        return PolicyMode.HANDCRAFTED, None, None, None


class BrotatoApiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: RuntimeConfig,
        server: Optional[BridgeClient] = None,
        vectorizer=None,
        state_hz: Optional[float] = None,
        profiler: RuntimeProfiler | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.profiler = profiler or RuntimeProfiler(
            enabled=cfg.runtime_profile_path is not None,
            sample_limit=cfg.runtime_profile_sample_limit,
        )
        self.server = server or BridgeClient(cfg.host, cfg.port, profiler=self.profiler)
        set_profiler = getattr(self.server, "set_profiler", None)
        if callable(set_profiler):
            set_profiler(self.profiler)
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
        self.combat_logger = DecisionTraceLogger(
            cfg.combat_decision_log, profiler=self.profiler
        )
        self.ui_controller = AutoUiController(
            max_shop_buys=cfg.max_shop_buys,
            max_shop_rerolls=cfg.max_shop_rerolls,
            build_profile=cfg.ui_build_profile,
            ui_model_path=cfg.ui_model_path,
            decision_log_path=cfg.ui_decision_log,
        )
        self.policy_mode, self.human_policy, self.human_builder, self.hybrid_controller = (
            _build_human_policy_stack(cfg, self.profiler)
        )
        self.training_paused = False
        self._last_published_ms: Optional[int] = None
        self._effective_state_hz = 0.0
        self._last_state_interval_ms = 0.0
        self._last_control_started: Optional[float] = None
        self._control_ticks = 0
        self._control_overruns = 0
        self._control_missed_ticks = 0
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
        from brotato_ai.control.movement_calibration import calibrate_startup_speed
        return calibrate_startup_speed(merged)

    def _observe_state_rate(self, state: Mapping[str, Any]) -> float:
        try:
            published_ms = int(state.get("published_at_ms", -1))
        except (TypeError, ValueError):
            published_ms = -1
        if published_ms < 0:
            return self._effective_state_hz
        previous = self._last_published_ms
        self._last_published_ms = published_ms
        self.profiler.count("state_samples")
        if previous is None or published_ms <= previous:
            return self._effective_state_hz
        self._last_state_interval_ms = max(1.0, float(published_ms - previous))
        instantaneous = 1000.0 / self._last_state_interval_ms
        if self._effective_state_hz <= 0.0:
            self._effective_state_hz = instantaneous
        else:
            self._effective_state_hz = 0.90 * self._effective_state_hz + 0.10 * instantaneous
        self.profiler.value("source_interval_ms", self._last_state_interval_ms)
        self.profiler.value("source_effective_hz", instantaneous)
        return self._effective_state_hz

    def _human_proposal(self, state: Mapping[str, Any]) -> Optional[HumanProposal]:
        """Run the learned human policy silently; never raises."""

        if self.human_policy is None or self.human_builder is None:
            return None
        try:
            held_action = int(self.previous_action)
            self.human_builder.observe(
                state,
                held_action,
                timestamp_ms=getattr(self, "_last_published_ms", 0),
            )
            model_input = self.human_builder.build_input(held_action)
            return self.human_policy.propose(model_input, held_action)
        except Exception:
            self.profiler.count("human_policy_propose_failure")
            return None

    def _apply_human_policy(
        self,
        requested: int,
        *,
        escape_active: bool,
    ) -> tuple[dict[str, Any], int]:
        """Return (trace fields, effective requested action) for this step.

        HANDCRAFTED is a no-op.  SHADOW_HUMAN only records fields.  HYBRID_HUMAN
        and EXPERIMENTAL_FULL_LEARNED may replace the requested action before
        the safety arbiter, which remains the single override authority.
        """

        if self.policy_mode is PolicyMode.HANDCRAFTED:
            return {}, requested
        proposal = self._human_proposal(self.last_state or {})
        if (
            self.policy_mode is PolicyMode.SHADOW_HUMAN
            or self.hybrid_controller is None
        ):
            # SHADOW records; a hybrid/full mode whose controller was never
            # attached (e.g. a bare env in tests) degrades to recording too —
            # never to changing the requested action.
            return {
                "human_proposed_action": proposal.action if proposal else None,
                "human_confidence": proposal.probability if proposal else None,
                "human_change_probability": proposal.change_probability if proposal else None,
                "human_duration_ms": proposal.duration_ms if proposal else None,
                "human_source": "shadow",
                "human_used": False,
                "human_agrees": (proposal.action == requested) if proposal else None,
                "human_fallback_reason": "" if proposal else "no_proposal",
            }, requested
        resolution = self.hybrid_controller.resolve(
            requested_action=requested,
            escape_active=escape_active,
            proposal=proposal,
        )
        return {
            "human_proposed_action": proposal.action if proposal else None,
            "human_confidence": proposal.probability if proposal else None,
            "human_change_probability": proposal.change_probability if proposal else None,
            "human_duration_ms": proposal.duration_ms if proposal else None,
            "human_source": resolution.source,
            "human_used": resolution.used_human,
            "human_agrees": (proposal.action == requested) if proposal else None,
            "human_fallback_reason": "" if resolution.used_human else resolution.reason,
        }, resolution.requested_action

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
        print(f"[v4-env] bridge state rate configured={self.state_hz:g} Hz")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence += 1
        if self.last_state is not None:
            try:
                self.server.send(reset_message(self.sequence))
            except Exception as exc:
                print(f"[v4-env] reset request not delivered: {exc}")
        self.ui_controller.reset_episode()
        self.decision_pipeline.reset()
        if self.human_builder is not None:
            self.human_builder.reset()
        if self.hybrid_controller is not None:
            self.hybrid_controller.reset()
        print("[v4-env] waiting for combat; structured menu automation is active")
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
        self.server.mark_state_processing_start()
        state = self._merge_cached_spatial_state(state)
        state = normalize_state(state)
        self.server.mark_state_processing_end()
        self.last_state = state
        self._last_published_ms = None
        self._effective_state_hz = 0.0
        self._last_state_interval_ms = 0.0
        self._last_control_started = None
        self._control_ticks = 0
        self._control_overruns = 0
        self._control_missed_ticks = 0
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
            "control_ticks": self._control_ticks,
            "control_overruns": self._control_overruns,
            "control_missed_ticks": self._control_missed_ticks,
        }

    def step(self, action):
        loop_started = self.profiler.begin("control_loop_total")
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
        self._control_ticks += 1
        if self.state_hz is not None and self.state_hz > 0.0 and control_interval_ms > 0.0:
            budget_ms = 1000.0 / self.state_hz
            # Diagnostics only: the policy and action semantics are unchanged.
            if control_interval_ms > budget_ms * 1.10:
                self._control_overruns += 1
                self.profiler.count("control_tick_overruns")
            missed = max(0, int(control_interval_ms // budget_ms) - 1)
            if missed:
                self._control_missed_ticks += missed
                self.profiler.count("control_ticks_missed", missed)
            self.profiler.value("control_budget_ms", budget_ms)
        self.profiler.count("control_ticks")
        self.profiler.value("control_interval_ms", control_interval_ms)
        if control_interval_ms > 0.0:
            self.profiler.value("control_effective_hz", 1000.0 / control_interval_ms)
        started = self.profiler.begin("human_policy")
        human_fields, requested = self._apply_human_policy(
            requested,
            escape_active=self.crowd_recovery_guard.active,
        )
        self.profiler.end("human_policy", started)
        started = self.profiler.begin("decision_pipeline")
        decision_trace = self.decision_pipeline.apply(
            previous_state,
            requested,
            previous_action=previous_action,
            state_interval_ms=self._last_state_interval_ms,
            control_interval_ms=control_interval_ms,
        )
        self.profiler.end("decision_pipeline", started)
        if human_fields:
            decision_trace = dataclasses.replace(decision_trace, **human_fields)
        decision = decision_trace.decision
        self.server.mark_action_decision()
        normalized = decision.applied_action
        started = self.profiler.begin("projectile_diagnostics")
        projectile_diagnostics = _projectile_diagnostics(
            previous_state,
            requested,
            normalized,
        )
        self.profiler.end("projectile_diagnostics", started)
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
        started = self.profiler.begin("recording_enqueue")
        self.combat_logger.record(decision_trace)
        self.profiler.end("recording_enqueue", started)
        self.sequence += 1
        started = self.profiler.begin("action_send")
        self.action_writer.write(decision_trace, self.sequence)
        self.profiler.end("action_send", started)
        self.server.mark_action_sent()
        previous_tick = int(self.last_state.get("tick", -1)) if self.last_state else None
        started = self.profiler.begin("state_wait")
        state = self.server.wait_for_state(
            timeout_sec=self.cfg.state_timeout_sec,
            after_tick=previous_tick,
            minimum_sequence=self.sequence,
        )
        self.profiler.end("state_wait", started)
        self.server.mark_state_processing_start()
        started = self.profiler.begin("state_cache_merge")
        state = self._merge_cached_spatial_state(state)
        self.profiler.end("state_cache_merge", started)
        started = self.profiler.begin("state_normalization")
        state = normalize_state(state)
        self.profiler.end("state_normalization", started)
        self.server.mark_state_processing_end()
        effective_state_hz = self._observe_state_rate(state)
        started = self.profiler.begin("movement_transition_metrics")
        movement = movement_transition_metrics(
            previous_state,
            state,
            previous_action,
            normalized,
            state_hz=self.state_hz or 12.0,
        )
        self.profiler.end("movement_transition_metrics", started)
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
            if decision_trace.enemy_contact_overridden
            else 0.0
        )
        started = self.profiler.begin("reward_and_telemetry")
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
        self.profiler.end("reward_and_telemetry", started)
        terminated = bool(state.get("dead") or state.get("victory"))
        ui_sent = []
        ui_confirmed = []
        if self.cfg.automate_menus and not terminated and state.get("phase") != "combat":
            started = self.profiler.begin("ui_automation")
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
            self.profiler.end("ui_automation", started)
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
        started = self.profiler.begin("observation_vectorizer")
        observation = self.vectorizer.build(state, self.previous_action)
        self.profiler.end("observation_vectorizer", started)
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
            "dead": bool(state.get("dead")),
            "victory": bool(state.get("victory")),
            "wave_clear": bool(
                reward_components.get("wave_clear", 0.0) > 0.0
                or reward_components.get("wave_advance", 0.0) > 0.0
                or (state.get("victory") and previous_state.get("phase") == "combat")
            ),
            "terminated": terminated,
            "truncated": truncated,
            "control_overruns": self._control_overruns,
            "control_missed_ticks": self._control_missed_ticks,
            "ui_sent": ui_sent,
            "ui_confirmed": ui_confirmed,
            "requested_action": requested,
            "applied_action": normalized,
            "safety_overridden": decision.overridden,
            "human_proposed_action": decision_trace.human_proposed_action,
            "human_confidence": decision_trace.human_confidence,
            "human_change_probability": decision_trace.human_change_probability,
            "human_source": decision_trace.human_source,
            "human_used": decision_trace.human_used,
            "human_agrees": decision_trace.human_agrees,
            "policy_mode": self.policy_mode.value,
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
            }
        )
        self.profiler.end("control_loop_total", loop_started)
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.training_paused:
            try:
                self.set_training_paused(False)
            except Exception:
                pass
        self.combat_logger.close()
        self.server.close()
        if self.cfg.runtime_profile_path is not None:
            self.profiler.write(self.cfg.runtime_profile_path)
        super().close()
