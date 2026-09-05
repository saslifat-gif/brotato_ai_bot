# Full-run evaluation

Use a frozen hierarchical PPO checkpoint with Well-Rounded, one starting SMG,
Danger 0. Start a fresh run and pause within the first five seconds of wave 1
before connecting. The evaluator rejects resumed shops, later waves, and
mismatched characters/weapons before its first gameplay action.

The bridge does not expose difficulty or seed control. Confirm Danger 0 manually
and record it in operator notes. Results retain `difficulty: null`; do not describe
these runs as seed matched. `--difficulty` requires bridge difficulty telemetry.

From a checkout with dependencies installed (PowerShell):

```powershell
$env:PYTHONPATH = "src;."
python -m v4.evaluate_full_run --model "C:\ml\brotato\models\version_3\ranged_smg_v2\v4_temporal_best\best_training_agent.zip" --human-model "C:\ml\brotato\models\version_3\human_demos\human_event_bc_manual_set.pt" --output "C:\ml\brotato\reports\full-run-baseline-01" --episodes 3 --variant baseline --operator-notes "Danger 0 confirmed manually"
```

Also available as `brotato-ai evaluate-full-run`. Use a new output directory each
time. Loading supports the former V3 bullet extractor import in the trusted local
checkpoint without modifying its weights. Human imitation runs in shadow mode
and never supplies the applied action.

A death triggers the existing full-restart UI path. If manual setup is required,
the next start validation fails rather than counting a partial run. After victory,
manual fresh-run setup may be necessary. Use one episode per invocation if the
game cannot restart automatically. Disconnections, timeouts, and conflicting
terminal flags do not count as deaths or enter the win-rate denominator.

Results record model hashes, source commit, starting build, outcomes, maximum
wave, damage, safety overrides, shadow disagreement, control timing and per-wave
builds. Shop and combat decisions are separate JSONL logs.

After baseline runs, repeat with `--variant shop`, then `--variant movement`,
using identical checkpoint, character, starting weapon, difficulty and control
rate. Alternate variants to reduce ordering bias. Three runs per variant are an
initial check, not reliable win-rate evidence. Compare complete-run wins and
reached waves, then timing and damage; retain baseline if results are inconclusive.

The shop experiment increases early weapon allowance and adjusts defensive stat
priorities from the current build. The movement experiment evaluates future escape
directions using constant-velocity forecasts. These are approximate forecasts,
not a game simulator, and add computation. The existing safety arbiter still
owns applied actions. Neither experiment changes normal runner defaults or has
yet demonstrated better survival.
