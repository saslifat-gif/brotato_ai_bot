"""Small reusable policy base for structured Brotato UI decisions.

The real-time movement policy stays separate.  This module ranks the variable
number of actions advertised by shop, upgrade and found-item screens.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


STAT_KEYS = (
    "stat_max_hp",
    "stat_armor",
    "stat_crit_chance",
    "stat_luck",
    "stat_attack_speed",
    "stat_elemental_damage",
    "stat_hp_regeneration",
    "stat_lifesteal",
    "stat_melee_damage",
    "stat_percent_damage",
    "stat_dodge",
    "stat_engineering",
    "stat_range",
    "stat_ranged_damage",
    "stat_speed",
    "stat_harvesting",
)
ROLE_NAMES = (
    "buy",
    "upgrade_choice",
    "take_item",
    "recycle_item",
    "reroll",
    "next_wave",
    "lock",
    "other",
)
CATEGORY_NAMES = ("item", "weapon", "upgrade")
ID_BUCKETS = 1024
ID_EMBEDDING_SIZE = 8
STICK_MELEE_TEACHER_VERSION = 3
RANGED_SMG_TEACHER_VERSION = 2
CONTEXT_SIZE = 4 + len(STAT_KEYS) + 4
CHOICE_SIZE = 5 + len(CATEGORY_NAMES) + 3 + len(ROLE_NAMES) + len(STAT_KEYS)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _hash_bucket(value: Any) -> int:
    encoded = str(value or "unknown").encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little") % ID_BUCKETS


def choice_data(action: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(action.get("choice"))


def effect_totals(choice: Mapping[str, Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in STAT_KEYS}
    effects = choice.get("effects", [])
    if not isinstance(effects, Iterable) or isinstance(effects, (str, bytes, Mapping)):
        return totals
    for effect in effects:
        effect = _mapping(effect)
        key = str(effect.get("key", "")).lower()
        if key in totals:
            totals[key] += _number(effect.get("value"))
    return totals


@dataclass(frozen=True)
class RankedUiAction:
    action: dict[str, Any]
    score: float
    source: str


class StickMeleeTeacher:
    """Language-independent rule teacher for the first UI curriculum."""

    stat_weights = {
        "stat_melee_damage": 8.0,
        "stat_percent_damage": 3.0,
        "stat_attack_speed": 2.5,
        "stat_lifesteal": 2.0,
        "stat_max_hp": 1.2,
        "stat_armor": 3.0,
        "stat_dodge": 1.0,
        "stat_speed": 0.8,
        "stat_hp_regeneration": 0.8,
        "stat_range": 0.25,
        "stat_crit_chance": 0.5,
        "stat_luck": 0.25,
        "stat_harvesting": 0.3,
        "stat_ranged_damage": -5.0,
        "stat_elemental_damage": -3.0,
        "stat_engineering": -3.0,
    }

    @staticmethod
    def _is_stick(choice: Mapping[str, Any]) -> bool:
        item_id = str(choice.get("id", "")).lower()
        base_id = str(choice.get("base_id", "")).lower()
        return base_id == "weapon_stick" or item_id.startswith("weapon_stick_")

    def score_choice(self, choice: Mapping[str, Any], wave: int) -> float:
        if not choice:
            return float("-inf")
        category = str(choice.get("category", "item")).lower()
        base_id = str(choice.get("base_id", "")).lower()
        item_id = str(choice.get("id", "")).lower()
        score = _number(choice.get("tier")) * 1.5
        if self._is_stick(choice):
            score += 200.0
        elif category == "weapon":
            # This first curriculum is intentionally a pure Stick build.
            # Other melee weapons consume one of the six synergy slots.
            score += -20.0 if int(_number(choice.get("weapon_type"), -1)) == 0 else -40.0
        if base_id == "upgrade_melee_damage" or item_id.startswith("upgrade_melee_damage_"):
            score += 80.0
        elif base_id == "upgrade_attack_speed" or item_id.startswith("upgrade_attack_speed_"):
            score += 35.0
        elif base_id in {"upgrade_percent_damage", "upgrade_armor", "upgrade_health"}:
            score += 20.0
        totals = effect_totals(choice)
        for key, value in totals.items():
            weight = self.stat_weights.get(key, 0.0)
            if key == "stat_harvesting" and wave > 10:
                weight *= 0.25
            score += weight * value
        tags = " ".join(str(tag).lower() for tag in choice.get("tags", []))
        if "melee" in tags:
            score += 8.0
        return score

    def select(
        self,
        state: Mapping[str, Any],
        actions: Sequence[Mapping[str, Any]],
    ) -> RankedUiAction | None:
        candidates = [dict(action) for action in actions if choice_data(action)]
        if not candidates:
            return None
        wave = int(_number(_mapping(state.get("wave")).get("number"), 0))
        materials = _number(_mapping(state.get("counters")).get("materials"), 0)
        rich_shop = wave >= 8 or materials >= 300
        ranked: list[tuple[float, dict[str, Any]]] = []
        for action in candidates:
            role = str(action.get("role", ""))
            choice = choice_data(action)
            base_score = self.score_choice(choice, wave)
            if role == "buy":
                price = max(0.0, _number(choice.get("price")))
                if choice.get("affordable") is False or price > materials:
                    continue
                # Price matters, but never overwhelms the explicit Stick bonus.
                score = base_score - (price / max(20.0, materials)) * 8.0
                # Early shops stay selective. Once the run is rich, buying a
                # positively useful item is better than carrying hundreds of
                # unspent materials into harder waves.
                minimum_buy_score = 0.0 if rich_shop else 4.0
                if score < minimum_buy_score:
                    continue
            elif role == "take_item":
                score = base_score + 4.0
            elif role == "recycle_item":
                score = -base_score
            elif role == "upgrade_choice":
                score = base_score
            else:
                continue
            ranked.append((score, action))
        if not ranked:
            return None
        score, action = max(ranked, key=lambda item: item[0])
        return RankedUiAction(
            action=action,
            score=float(score),
            source=f"stick_melee_teacher_v{STICK_MELEE_TEACHER_VERSION}",
        )


class RangedSmgTeacher:
    """Rule teacher for a focused Well-Rounded ranged build.

    The policy fills the six weapon slots with SMGs whenever possible and
    uses a Shredder as a controlled crowd-clear fallback.  It deliberately
    keeps the legacy Stick teacher above so old datasets and experiments
    remain reproducible.
    """

    stat_weights = {
        "stat_ranged_damage": 8.0,
        "stat_attack_speed": 5.0,
        "stat_percent_damage": 4.0,
        "stat_lifesteal": 4.5,
        "stat_armor": 3.5,
        "stat_max_hp": 3.0,
        "stat_speed": 2.5,
        "stat_dodge": 2.2,
        "stat_hp_regeneration": 1.5,
        "stat_range": 1.5,
        "stat_crit_chance": 1.25,
        "stat_luck": 1.0,
        "stat_harvesting": 1.0,
        "stat_melee_damage": -6.0,
        "stat_elemental_damage": -5.0,
        "stat_engineering": -6.0,
    }

    # A ranged build still needs a stable early-game floor.  Filling all six
    # weapon slots immediately leaves Well-Rounded with too little armor,
    # health, speed, and sustain to survive the first dense waves.
    early_wave_end = 8
    early_survival_upgrade_bonus = {
        "upgrade_armor": 44.0,
        "upgrade_health": 44.0,
        "upgrade_max_hp": 44.0,
        "upgrade_lifesteal": 40.0,
        "upgrade_hp_regeneration": 36.0,
        "upgrade_dodge": 30.0,
        "upgrade_speed": 30.0,
    }
    early_survival_effect_weights = {
        "stat_armor": 22.0,
        "stat_max_hp": 18.0,
        "stat_lifesteal": 22.0,
        "stat_hp_regeneration": 16.0,
        "stat_speed": 18.0,
        "stat_dodge": 14.0,
    }

    @staticmethod
    def _choice_text(choice: Mapping[str, Any]) -> str:
        values = [
            choice.get("id", ""),
            choice.get("base_id", ""),
            choice.get("name_key", ""),
            choice.get("display_name", ""),
        ]
        for key in ("tags", "sets"):
            value = choice.get(key, [])
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                values.extend(value)
        return " ".join(str(value).lower() for value in values)

    @classmethod
    def _is_smg(cls, choice: Mapping[str, Any]) -> bool:
        text = cls._choice_text(choice)
        return (
            "weapon_smg" in text
            or " smg" in f" {text}"
            or "submachine" in text
        )

    @classmethod
    def _is_shredder(cls, choice: Mapping[str, Any]) -> bool:
        text = cls._choice_text(choice)
        return "weapon_shredder" in text or "shredder" in text

    @classmethod
    def _is_ranged_weapon(cls, choice: Mapping[str, Any]) -> bool:
        if str(choice.get("category", "")).lower() != "weapon":
            return False
        if int(_number(choice.get("weapon_type"), -1)) == 1:
            return True
        text = cls._choice_text(choice)
        return any(token in text for token in ("ranged", "gun", "pistol", "shotgun"))

    @classmethod
    def _weapon_counts(cls, state: Mapping[str, Any]) -> tuple[int, int, int]:
        build = _mapping(state.get("build"))
        weapons = build.get("weapons", [])
        if not isinstance(weapons, list):
            return 0, 0, 0
        smg = sum(1 for item in weapons if cls._is_smg(_mapping(item)))
        shredder = sum(1 for item in weapons if cls._is_shredder(_mapping(item)))
        return smg, shredder, len(weapons)

    @staticmethod
    def _early_weapon_cap(wave: int) -> int:
        """Increase the weapon allowance gradually instead of front-loading it."""

        if wave <= 2:
            return 1
        if wave <= 4:
            return 2
        if wave <= 6:
            return 3
        return 4

    @classmethod
    def _early_survival_bonus(cls, choice: Mapping[str, Any], wave: int) -> float:
        if wave >= cls.early_wave_end:
            return 0.0
        base_id = str(choice.get("base_id", "")).lower()
        phase = 1.0 if wave <= 4 else 0.55
        bonus = cls.early_survival_upgrade_bonus.get(base_id, 0.0) * phase
        totals = effect_totals(choice)
        for key, value in totals.items():
            if value > 0.0:
                bonus += cls.early_survival_effect_weights.get(key, 0.0) * value * phase
        return bonus

    @classmethod
    def _keyword_bonus(cls, choice: Mapping[str, Any], wave: int) -> float:
        text = cls._choice_text(choice)
        score = 0.0
        if any(token in text for token in ("sharp_bullet", "bandana", "pumpkin", "pierc")):
            score += 36.0
        if any(token in text for token in ("baby_with_a_beard", "cyberball", "baby_elephant")):
            score += 22.0
        if any(token in text for token in ("turret", "engineering", "mine")):
            score -= 28.0
        if any(token in text for token in ("elemental", "burn", "wand", "lightning")):
            score -= 18.0
        if wave >= 12 and "harvest" in text:
            score -= 8.0
        return score

    def score_choice(
        self,
        choice: Mapping[str, Any],
        wave: int,
        state: Mapping[str, Any] | None = None,
    ) -> float:
        if not choice:
            return float("-inf")
        category = str(choice.get("category", "item")).lower()
        base_id = str(choice.get("base_id", "")).lower()
        item_id = str(choice.get("id", "")).lower()
        score = _number(choice.get("tier")) * 1.5 + self._keyword_bonus(choice, wave)
        if category == "weapon":
            smg_count, shredder_count, weapon_count = self._weapon_counts(state or {})
            if weapon_count >= 6:
                return -120.0
            if self._is_smg(choice):
                score += 180.0 + (6 - smg_count) * 12.0
                if wave < self.early_wave_end and smg_count >= self._early_weapon_cap(wave):
                    # A third/fourth gun is useful later, but should not
                    # displace a meaningful defensive choice before wave 8.
                    score -= 185.0
            elif self._is_shredder(choice):
                score += 105.0 + (2 - min(shredder_count, 2)) * 8.0
                if wave < 8 and smg_count < 4:
                    score -= 28.0
                if wave < self.early_wave_end and weapon_count >= self._early_weapon_cap(wave):
                    score -= 145.0
            elif self._is_ranged_weapon(choice):
                # Keep a run alive when the preferred weapons do not appear,
                # but never let an unrelated gun displace the core plan.
                score += 28.0 if smg_count == 0 and shredder_count == 0 else -72.0
            else:
                return -120.0
        elif base_id in {"upgrade_ranged_damage", "upgrade_projectile_damage"} or item_id.startswith(
            "upgrade_ranged_damage_"
        ):
            score += 92.0
        elif base_id == "upgrade_attack_speed" or item_id.startswith("upgrade_attack_speed_"):
            score += 68.0
        elif base_id in {"upgrade_percent_damage", "upgrade_damage"}:
            score += 48.0
        elif base_id in {
            "upgrade_lifesteal",
            "upgrade_armor",
            "upgrade_health",
            "upgrade_max_hp",
            "upgrade_hp_regeneration",
        }:
            score += 44.0
        elif base_id in {"upgrade_dodge", "upgrade_speed", "upgrade_range"}:
            score += 30.0
        totals = effect_totals(choice)
        for key, value in totals.items():
            weight = self.stat_weights.get(key, 0.0)
            if key in {"stat_luck", "stat_harvesting"} and wave >= 12:
                weight *= 0.3
            if key in {"stat_armor", "stat_max_hp", "stat_lifesteal", "stat_speed"} and wave >= 12:
                weight *= 1.35
            score += weight * value
        score += self._early_survival_bonus(choice, wave)
        return score

    def select(
        self,
        state: Mapping[str, Any],
        actions: Sequence[Mapping[str, Any]],
    ) -> RankedUiAction | None:
        candidates = [dict(action) for action in actions if choice_data(action)]
        if not candidates:
            return None
        wave = int(_number(_mapping(state.get("wave")).get("number"), 0))
        materials = _number(_mapping(state.get("counters")).get("materials"), 0)
        rich_shop = wave >= 8 or materials >= 300
        ranked: list[tuple[float, dict[str, Any]]] = []
        for action in candidates:
            role = str(action.get("role", ""))
            choice = choice_data(action)
            base_score = self.score_choice(choice, wave, state)
            if role == "buy":
                price = max(0.0, _number(choice.get("price")))
                if choice.get("affordable") is False or price > materials:
                    continue
                score = base_score - (price / max(20.0, materials)) * 8.0
                minimum_buy_score = 0.0 if rich_shop else (8.0 if wave < self.early_wave_end else 4.0)
                if score < minimum_buy_score:
                    continue
            elif role == "take_item":
                score = base_score + 4.0
            elif role == "recycle_item":
                score = -base_score
            elif role == "upgrade_choice":
                score = base_score
            else:
                continue
            ranked.append((score, action))
        if not ranked:
            return None
        score, action = max(ranked, key=lambda item: item[0])
        return RankedUiAction(
            action=action,
            score=float(score),
            source=f"ranged_smg_teacher_v{RANGED_SMG_TEACHER_VERSION}",
        )


@dataclass(frozen=True)
class UiFeatureRow:
    context: np.ndarray
    choice: np.ndarray
    item_bucket: int
    base_bucket: int


class RangedSustainTeacher(RangedSmgTeacher):
    """User preference: ranged damage, then life steal, then percent damage."""

    priority_bonuses = (
        ("stat_ranged_damage", ("upgrade_ranged_damage", "upgrade_projectile_damage"), 180.0),
        ("stat_lifesteal", ("upgrade_lifesteal",), 140.0),
        ("stat_percent_damage", ("upgrade_percent_damage", "upgrade_damage"), 100.0),
    )

    def score_choice(self, choice, wave, state=None):
        score = super().score_choice(choice, wave, state)
        if str(choice.get("category", "item")).lower() == "weapon":
            return score
        totals = effect_totals(choice)
        base = str(choice.get("base_id", "")).lower()
        item_id = str(choice.get("id", "")).lower()
        for stat, upgrades, bonus in self.priority_bonuses:
            is_upgrade = base in upgrades or any(item_id.startswith(u + "_") for u in upgrades)
            if totals[stat] > 0 or (is_upgrade and totals[stat] >= 0):
                score += bonus
        return score

    def select(self, state, actions):
        ranked = super().select(state, actions)
        if ranked is None:
            return None
        return RankedUiAction(ranked.action, ranked.score, "ranged_sustain_teacher_v1")


class UiChoiceVectorizer:
    """Fixed features shared by teacher imitation and learned UI policies."""

    def build(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> UiFeatureRow:
        wave = _mapping(state.get("wave"))
        counters = _mapping(state.get("counters"))
        build = _mapping(state.get("build"))
        stats = _mapping(build.get("stats"))
        weapons = build.get("weapons", []) if isinstance(build.get("weapons"), list) else []
        items = build.get("items", []) if isinstance(build.get("items"), list) else []
        stick_count = sum(1 for item in weapons if StickMeleeTeacher._is_stick(_mapping(item)))
        smg_count = sum(1 for item in weapons if RangedSmgTeacher._is_smg(_mapping(item)))
        focus_weapon_count = max(stick_count, smg_count)
        context_values = [
            np.clip(_number(wave.get("number")) / 20.0, 0.0, 2.0),
            np.clip(_number(wave.get("time_left")) / 60.0, 0.0, 1.0),
            np.clip(_number(counters.get("materials")) / 500.0, 0.0, 4.0),
            1.0 if state.get("phase") != "combat" else 0.0,
            *[np.clip(_number(stats.get(key)) / 100.0, -5.0, 5.0) for key in STAT_KEYS],
            np.clip(len(weapons) / 6.0, 0.0, 2.0),
            np.clip(focus_weapon_count / 6.0, 0.0, 2.0),
            np.clip(len(items) / 100.0, 0.0, 2.0),
            1.0,
        ]
        choice = choice_data(action)
        role = str(action.get("role", "other"))
        category = str(choice.get("category", "item"))
        weapon_type = int(_number(choice.get("weapon_type"), -1))
        totals = effect_totals(choice)
        choice_values = [
            np.clip(_number(choice.get("tier")) / 4.0, 0.0, 2.0),
            np.clip(_number(choice.get("price")) / 500.0, -1.0, 4.0),
            np.clip(_number(choice.get("base_value")) / 500.0, -1.0, 4.0),
            1.0 if choice.get("affordable", True) else 0.0,
            1.0
            if StickMeleeTeacher._is_stick(choice) or RangedSmgTeacher._is_smg(choice)
            else 0.0,
            *[1.0 if category == name else 0.0 for name in CATEGORY_NAMES],
            *[1.0 if weapon_type == value else 0.0 for value in (-1, 0, 1)],
            *[1.0 if role == name else 0.0 for name in ROLE_NAMES],
            *[np.clip(totals[key] / 20.0, -5.0, 5.0) for key in STAT_KEYS],
        ]
        context_array = np.asarray(context_values, dtype=np.float32)
        choice_array = np.asarray(choice_values, dtype=np.float32)
        if context_array.shape != (CONTEXT_SIZE,) or choice_array.shape != (CHOICE_SIZE,):
            raise RuntimeError(
                f"UI feature shape mismatch context={context_array.shape} choice={choice_array.shape}"
            )
        return UiFeatureRow(
            context=context_array,
            choice=choice_array,
            item_bucket=_hash_bucket(choice.get("id")),
            base_bucket=_hash_bucket(choice.get("base_id")),
        )


class UiBuildBase(nn.Module):
    """Tiny candidate-scoring backbone reusable across later UI tasks."""

    def __init__(self) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(ID_BUCKETS, ID_EMBEDDING_SIZE)
        self.base_embedding = nn.Embedding(ID_BUCKETS, ID_EMBEDDING_SIZE)
        self.context_encoder = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, 48), nn.Tanh(), nn.Linear(48, 32), nn.Tanh()
        )
        self.choice_encoder = nn.Sequential(
            nn.Linear(CHOICE_SIZE + ID_EMBEDDING_SIZE * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 48),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(80, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(
        self,
        context: torch.Tensor,
        choice: torch.Tensor,
        item_bucket: torch.Tensor,
        base_bucket: torch.Tensor,
    ) -> torch.Tensor:
        embedded = torch.cat(
            [choice, self.item_embedding(item_bucket), self.base_embedding(base_bucket)], dim=-1
        )
        hidden = torch.cat(
            [self.context_encoder(context), self.choice_encoder(embedded)], dim=-1
        )
        return self.score_head(hidden).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LearnedUiBuildPolicy:
    def __init__(self, checkpoint: Path):
        payload = torch.load(checkpoint, map_location="cpu")
        self.model = UiBuildBase()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.vectorizer = UiChoiceVectorizer()

    def select(
        self,
        state: Mapping[str, Any],
        actions: Sequence[Mapping[str, Any]],
    ) -> RankedUiAction | None:
        candidates = [dict(action) for action in actions if choice_data(action)]
        if not candidates:
            return None
        rows = [self.vectorizer.build(state, action) for action in candidates]
        with torch.no_grad():
            scores = self.model(
                torch.from_numpy(np.stack([row.context for row in rows])),
                torch.from_numpy(np.stack([row.choice for row in rows])),
                torch.tensor([row.item_bucket for row in rows], dtype=torch.long),
                torch.tensor([row.base_bucket for row in rows], dtype=torch.long),
            )
        index = int(torch.argmax(scores).item())
        return RankedUiAction(
            action=candidates[index],
            score=float(scores[index].item()),
            source="ui_build_base",
        )


class UiDecisionLogger:
    def __init__(self, path: Path | None):
        self.path = path

    def record(
        self,
        state: Mapping[str, Any],
        selected: Mapping[str, Any],
        *,
        source: str,
        score: float | None,
    ) -> None:
        if self.path is None:
            return
        ui = _mapping(state.get("ui"))
        actions = [dict(action) for action in ui.get("actions", []) if isinstance(action, Mapping)]
        selected_id = str(selected.get("id", ""))
        selected_index = next(
            (index for index, action in enumerate(actions) if str(action.get("id", "")) == selected_id),
            -1,
        )
        record = {
            "schema": 1,
            "timestamp": time.time(),
            "phase": state.get("phase"),
            "wave": dict(_mapping(state.get("wave"))),
            "counters": dict(_mapping(state.get("counters"))),
            "build": dict(_mapping(state.get("build"))),
            "actions": actions,
            "selected_index": selected_index,
            "selected_id": selected_id,
            "selected_role": selected.get("role"),
            "policy_source": source,
            "policy_score": score,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
