"""Bounded menu automation using UI actions advertised by the game bridge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from v3.bridge_server import BridgeServer
from v3.protocol import ui_action_message
from v3.ui_build_policy import (
    LearnedUiBuildPolicy,
    RankedUiAction,
    StickMeleeTeacher,
    UiDecisionLogger,
)

MAX_NO_ACTION_STATES = 30
MAX_TRANSITION_WAIT_STATES = 300
MAX_UPGRADE_CLICKS_PER_WAVE = 32
MAX_ITEM_CLAIMS_PER_WAVE = 32
UPGRADE_FALLBACK_WAIT_STATES = 4


def shop_budget_limits(
    wave: int,
    materials: int,
    *,
    base_buys: int,
    base_rerolls: int,
) -> tuple[int, int, int]:
    """Scale bounded shop work so rich late runs do not hoard materials."""

    wave = max(0, int(wave))
    materials = max(0, int(materials))
    buys = max(0, int(base_buys))
    rerolls = max(0, int(base_rerolls))
    late_or_rich = wave >= 8 or materials >= 300
    if late_or_rich and buys > 0:
        # Four visible offers and ten rerolls were not enough to drain the
        # 700-1,000 material balances produced by strong mid-game runs.
        buys = max(buys, min(48, 8 + materials // 25))
    if late_or_rich and rerolls > 0:
        rerolls = max(rerolls, min(20, 3 + materials // 50))
    reserve = max(20, min(100, 160 - wave * 10)) if late_or_rich else 0
    return buys, rerolls, reserve


def available_actions(state: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    ui = state.get("ui", {})
    actions = ui.get("actions", []) if isinstance(ui, Mapping) else []
    return [
        dict(action)
        for action in actions
        if isinstance(action, Mapping)
        and action.get("role") == role
        and bool(action.get("enabled"))
        and str(action.get("id", "")).startswith("/")
    ]


@dataclass
class UiAutomationResult:
    state: dict[str, Any]
    sequence: int
    states: list[dict[str, Any]]
    sent_roles: list[str]
    confirmed_roles: list[str]


class AutoUiController:
    def __init__(
        self,
        max_shop_buys: int = 4,
        max_shop_rerolls: int = 1,
        *,
        build_profile: str = "stick_melee",
        ui_model_path: Path | None = None,
        decision_log_path: Path | None = None,
    ):
        self.max_shop_buys = max(0, int(max_shop_buys))
        self.max_shop_rerolls = max(0, int(max_shop_rerolls))
        self._shop_wave = None
        self._shop_buys = 0
        self._shop_rerolls = 0
        self._shop_buy_limit = self.max_shop_buys
        self._shop_reroll_limit = self.max_shop_rerolls
        self._shop_reserve = 0
        self._attempted: set[tuple[str, int]] = set()
        self._upgrade_clicks: dict[int, int] = {}
        self._item_claims: dict[int, int] = {}
        self._teacher = StickMeleeTeacher() if build_profile == "stick_melee" else None
        self._learned = LearnedUiBuildPolicy(ui_model_path) if ui_model_path else None
        self._decision_logger = UiDecisionLogger(decision_log_path)

    def _rank_structured(
        self,
        state: Mapping[str, Any],
        actions: list[dict[str, Any]],
        *,
        allow_learned: bool = True,
    ) -> dict[str, Any] | None:
        ranked: RankedUiAction | None = None
        if allow_learned and self._learned is not None:
            ranked = self._learned.select(state, actions)
        if ranked is None and self._teacher is not None:
            ranked = self._teacher.select(state, actions)
        if ranked is None:
            return None
        action = dict(ranked.action)
        action["_policy_source"] = ranked.source
        action["_policy_score"] = ranked.score
        return action

    def _rank_shop(
        self,
        state: Mapping[str, Any],
        actions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Let the learned model rank purchases without buying harmful offers.

        The imitation dataset contains choice rankings but no calibrated
        no-purchase class.  Keep the rule teacher as an eligibility gate until
        that abstention target exists, then use the compact model to rank the
        safe candidates.
        """

        candidates = actions
        if self._learned is not None and self._teacher is not None:
            candidates = [
                action
                for action in actions
                if self._teacher.select(state, [action]) is not None
            ]
        return self._rank_structured(state, candidates)

    def reset_episode(self) -> None:
        self._shop_wave = None
        self._shop_buys = 0
        self._shop_rerolls = 0
        self._shop_buy_limit = self.max_shop_buys
        self._shop_reroll_limit = self.max_shop_rerolls
        self._shop_reserve = 0
        self._attempted.clear()
        self._upgrade_clicks.clear()
        self._item_claims.clear()

    def _enter_shop(self, state: Mapping[str, Any]) -> None:
        wave = state.get("wave", {})
        wave_number = wave.get("number") if isinstance(wave, Mapping) else None
        if wave_number == self._shop_wave:
            return
        self._shop_wave = wave_number
        self._shop_buys = 0
        self._shop_rerolls = 0
        self._attempted.clear()
        materials = int(state.get("counters", {}).get("materials", 0))
        self._shop_buy_limit, self._shop_reroll_limit, self._shop_reserve = (
            shop_budget_limits(
                int(wave_number or 0),
                materials,
                base_buys=self.max_shop_buys,
                base_rerolls=self.max_shop_rerolls,
            )
        )
        if self._shop_buy_limit > self.max_shop_buys or self._shop_reroll_limit > self.max_shop_rerolls:
            print(
                f"[v3-ui] rich-shop materials={materials} wave={wave_number} "
                f"buy_limit={self._shop_buy_limit} reroll_limit={self._shop_reroll_limit} "
                f"reserve={self._shop_reserve}"
            )

    def choose(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        phase = str(state.get("phase", "menu"))
        if phase == "upgrade":
            choices = available_actions(state, "upgrade_choice")
            wave = int(state.get("wave", {}).get("number", -1))
            if self._upgrade_clicks.get(wave, 0) >= MAX_UPGRADE_CLICKS_PER_WAVE:
                return None
            return self._rank_structured(
                state, choices, allow_learned=False
            ) or (choices[0] if choices else None)
        if phase == "item_found":
            wave = int(state.get("wave", {}).get("number", -1))
            if self._item_claims.get(wave, 0) >= MAX_ITEM_CLAIMS_PER_WAVE:
                return None
            take = available_actions(state, "take_item")
            recycle = available_actions(state, "recycle_item")
            ranked = self._rank_structured(
                state, [*take, *recycle], allow_learned=False
            )
            if ranked is not None:
                return ranked
            if take:
                return take[0]
            return recycle[0] if recycle else None
        if phase == "shop":
            self._enter_shop(state)
            materials = int(state.get("counters", {}).get("materials", 0))
            if self._shop_buys < self._shop_buy_limit:
                buy_candidates = []
                for action in available_actions(state, "buy"):
                    key = (str(action["id"]), materials)
                    if key not in self._attempted:
                        buy_candidates.append(action)
                ranked = self._rank_shop(state, buy_candidates)
                if ranked is not None:
                    return ranked
                # Old bridge packages have no structured choice metadata. Keep
                # the verified first-affordable fallback during upgrades.
                if buy_candidates and not any(action.get("choice") for action in buy_candidates):
                    return buy_candidates[0]
            if (
                self._shop_rerolls < self._shop_reroll_limit
                and materials > self._shop_reserve
            ):
                rerolls = available_actions(state, "reroll")
                if rerolls:
                    return rerolls[0]
            next_wave = available_actions(state, "next_wave")
            return next_wave[0] if next_wave else None
        if phase == "game_over":
            restarts = available_actions(state, "restart")
            return restarts[0] if restarts else None
        if phase == "menu":
            starts = available_actions(state, "start")
            return starts[0] if len(starts) == 1 else None
        return None

    def mark_sent(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> None:
        self._decision_logger.record(
            state,
            action,
            source=str(action.get("_policy_source", "script_fallback")),
            score=(
                float(action["_policy_score"])
                if action.get("_policy_score") is not None
                else None
            ),
        )
        role = str(action.get("role", ""))
        if role == "upgrade_choice":
            wave = int(state.get("wave", {}).get("number", -1))
            self._upgrade_clicks[wave] = self._upgrade_clicks.get(wave, 0) + 1
        elif role in {"take_item", "recycle_item"}:
            wave = int(state.get("wave", {}).get("number", -1))
            self._item_claims[wave] = self._item_claims.get(wave, 0) + 1
        elif role == "buy":
            materials = int(state.get("counters", {}).get("materials", 0))
            self._attempted.add((str(action.get("id", "")), materials))
            self._shop_buys += 1
        elif role == "reroll":
            self._shop_rerolls += 1
            self._attempted.clear()

    def advance(
        self,
        server: BridgeServer,
        state: dict[str, Any],
        sequence: int,
        timeout_sec: float,
        *,
        allow_restart: bool = False,
    ) -> UiAutomationResult:
        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        observed: list[dict[str, Any]] = []
        sent_roles: list[str] = []
        confirmed_roles: list[str] = []
        no_action_states = 0
        pending_phase_change: tuple[str, str, int, str] | None = None
        while state.get("phase") != "combat":
            phase = str(state.get("phase", "menu"))
            ui = state.get("ui", {})
            last_result = ui.get("last_result", {}) if isinstance(ui, Mapping) else {}
            result_ok = (
                isinstance(last_result, Mapping)
                and int(last_result.get("sequence", -1)) == pending_phase_change[2]
                and bool(last_result.get("ok"))
            ) if pending_phase_change is not None else False
            result_changed = result_ok and bool(last_result.get("changed"))
            restart_stage_changed = (
                pending_phase_change is not None
                and pending_phase_change[1] == "restart"
                and any(
                    str(action.get("id", "")) != pending_phase_change[3]
                    for action in available_actions(state, "restart")
                )
            )
            repeatable_choice = (
                pending_phase_change is not None
                and pending_phase_change[1]
                in {"upgrade_choice", "take_item", "recycle_item"}
            )
            choice_fallback_ready = (
                repeatable_choice
                and not result_ok
                and no_action_states >= UPGRADE_FALLBACK_WAIT_STATES
                and bool(available_actions(state, pending_phase_change[1]))
            )
            if pending_phase_change is not None and (
                phase != pending_phase_change[0]
                or (repeatable_choice and result_changed)
                or choice_fallback_ready
                or restart_stage_changed
            ):
                print(
                    f"[v3-ui] confirmed role={pending_phase_change[1]} "
                    f"phase={pending_phase_change[0]}->{phase} "
                    f"ui_changed={result_changed}"
                )
                confirmed_roles.append(pending_phase_change[1])
                pending_phase_change = None
            if phase == "victory" or (phase == "game_over" and not allow_restart):
                break
            # Once a transition action is sent, wait for the advertised phase
            # to disappear.  Shop counters can change immediately after the Go
            # button is pressed while the old controls remain visible for a
            # frame; choosing again there can activate a stale shop button.
            action = None if pending_phase_change is not None else self.choose(state)
            previous_tick = int(state.get("tick", -1))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out automating Brotato phase={phase}")
            if action is not None:
                if action.get("role") == "restart" and not allow_restart:
                    break
                sequence += 1
                server.send(
                    ui_action_message(str(action["id"]), sequence),
                    timeout_sec=min(remaining, 10.0),
                )
                self.mark_sent(state, action)
                sent_roles.append(str(action.get("role", "")))
                print(
                    f"[v3-ui] sent role={action.get('role')} "
                    f"name={action.get('name', '')} target={action.get('id')} "
                    f"choice={action.get('choice', {}).get('id', '-')} "
                    f"policy={action.get('_policy_source', 'script_fallback')} "
                    f"score={action.get('_policy_score', '-')}"
                )
                if action.get("role") in {
                    "upgrade_choice",
                    "take_item",
                    "recycle_item",
                    "next_wave",
                    "restart",
                }:
                    pending_phase_change = (
                        phase,
                        str(action.get("role")),
                        sequence,
                        str(action.get("id", "")),
                    )
                minimum_sequence = sequence
                no_action_states = 0
            else:
                minimum_sequence = None
                no_action_states += 1
                if (
                    pending_phase_change is not None
                    and no_action_states >= MAX_TRANSITION_WAIT_STATES
                ):
                    raise RuntimeError(
                        "UI transition did not complete "
                        f"role={pending_phase_change[1]} phase={phase}"
                    )
                if (
                    pending_phase_change is None
                    and phase not in {"wave_end", "menu"}
                    and no_action_states >= MAX_NO_ACTION_STATES
                ):
                    raise RuntimeError(f"no safe UI action advertised for phase={phase}")
            state = server.wait_for_state(
                timeout_sec=remaining,
                after_tick=previous_tick,
                minimum_sequence=minimum_sequence,
            )
            observed.append(state)
        if pending_phase_change is not None and state.get("phase") != pending_phase_change[0]:
            print(
                f"[v3-ui] confirmed role={pending_phase_change[1]} "
                f"phase={pending_phase_change[0]}->{state.get('phase')}"
            )
            confirmed_roles.append(pending_phase_change[1])
        return UiAutomationResult(
            state=state,
            sequence=sequence,
            states=observed,
            sent_roles=sent_roles,
            confirmed_roles=confirmed_roles,
        )
