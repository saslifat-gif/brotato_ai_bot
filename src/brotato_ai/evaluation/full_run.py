"""Full-run evaluation bookkeeping, independent of the game transport."""
from dataclasses import dataclass, field
from typing import Any, Mapping


def validate_start(state: Mapping[str, Any], *, character: str, weapon: str) -> dict:
    build = state.get("build", {})
    actual = str(build.get("character_id") or state.get("combat", {}).get("character_id", ""))
    weapons = [str(w.get("base_id") or w.get("id", "")) for w in build.get("weapons", [])]
    if state.get("phase") != "combat" or int(state.get("wave", {}).get("number", 0)) != 1:
        raise ValueError("Full-run evaluation must start in combat on wave 1; retries and partial runs are excluded")
    wave = state.get("wave", {})
    if "duration" in wave and "time_left" in wave and float(wave["duration"]) - float(wave["time_left"]) > 5:
        raise ValueError("Wave 1 is already in progress; start a fresh run within its first five seconds")
    if actual != character:
        raise ValueError(f"Expected character {character!r}, received {actual!r}")
    if weapon and not any(w == weapon or w.startswith(weapon + "_") for w in weapons):
        raise ValueError(f"Expected starting weapon {weapon!r}, received {weapons!r}")
    return {"character": actual, "starting_weapons": weapons, "difficulty": build.get("difficulty")}


@dataclass
class FullRunMetrics:
    steps: int = 0
    reward: float = 0.0
    max_wave: int = 1
    health_loss: float = 0.0
    overrides: int = 0
    shadow_proposals: int = 0
    shadow_disagreements: int = 0
    control_ms: list[float] = field(default_factory=list)
    wave_builds: dict[int, dict] = field(default_factory=dict)

    def observe(self, reward: float, info: Mapping[str, Any], state: Mapping[str, Any]) -> None:
        self.steps += 1
        self.reward += float(reward)
        wave = int(info.get("wave", 0))
        self.max_wave = max(self.max_wave, wave)
        self.health_loss += max(0.0, float(info.get("damage_taken", 0)))
        self.overrides += int(bool(info.get("safety_overridden")))
        if info.get("human_proposed_action") is not None:
            self.shadow_proposals += 1
            self.shadow_disagreements += int(not info.get("human_agrees", False))
        interval = float(info.get("hazard_control_interval_ms", 0))
        if interval > 0 and state.get("phase") == "combat":
            self.control_ms.append(interval)
        if wave not in self.wave_builds:
            self.wave_builds[wave] = dict(state.get("build", {}))

    def finish(self, info: Mapping[str, Any], *, terminated: bool, truncated: bool, elapsed: float) -> dict:
        victory, dead = bool(info.get("victory")), bool(info.get("dead"))
        valid = bool(terminated and not truncated and victory != dead)
        intervals = sorted(self.control_ms)
        p95 = intervals[min(len(intervals) - 1, int(len(intervals) * .95))] if intervals else None
        return {
            "valid_full_run": valid, "outcome": "victory" if valid and victory else "death" if valid else "incomplete",
            "victory": victory, "dead": dead, "terminated": terminated, "truncated": truncated,
            "max_wave": self.max_wave, "steps": self.steps, "reward": self.reward,
            "elapsed_seconds": elapsed, "health_loss": self.health_loss,
            "safety_override_fraction": self.overrides / max(1, self.steps),
            "shadow_proposals": self.shadow_proposals,
            "shadow_disagreement_fraction": self.shadow_disagreements / max(1, self.shadow_proposals),
            "control_interval_p95_ms": p95,
            "wave_builds": self.wave_builds,
        }


def summarize(runs: list[dict]) -> dict:
    valid = [r for r in runs if r["valid_full_run"]]
    wins = sum(r["outcome"] == "victory" for r in valid)
    return {"valid_runs": len(valid), "incomplete_runs": len(runs) - len(valid), "wins": wins,
            "win_rate": wins / len(valid) if valid else None,
            "mean_max_wave": sum(r["max_wave"] for r in valid) / len(valid) if valid else None}
