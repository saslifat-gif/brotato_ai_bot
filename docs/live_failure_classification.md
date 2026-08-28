# Live death extra-decision classification

Date: 2026-08-28  
Recordings: `models/version_3/raw_records/live_rate_{10,15,20,24,30,60}hz.jsonl`  
Method: last combat frame before `EndRun`, plus one extra controller call
about 80–150 ms earlier. Geometry is scored on the impact frame. This does
not re-simulate physics.

Machine-readable: `reports/live_failure_classification.json`

## Totals (14 deaths)

| Category | Count | Meaning |
|---|---:|---|
| wrong_action | 6 | A safer lane existed; the held action was worse |
| already_best | 6 | Held action was already the safest scored lane, then death |
| too_late | 1 | Projectile TTI under 50 ms at the lookback frame |
| no_safe_action | 1 | Every action still high-risk at impact |

One extra arbiter call would have been materially safer on **5 / 14** deaths.
Only **1 / 14** is a timing/frame-rate death.

Most lookbacks had **TTI = none** (no projectile on a collision course) and
**14–48 enemies**. These are contact/crowd deaths, not missed dodges.

## What to change in the policy

1. **Crowd/contact scoring** — six `already_best` deaths had scored risk 0.0
   with 17–48 enemies and 0–2 projectiles. The scorer is treating a lethal
   surround as safe. Raise contact/crowd risk and fire ESCAPE earlier at low
   HP.
2. **Action switching** — five extra calls would have moved to a safer lane
   (example: risk 2.43 → 0.0). The bot is holding a bad action. Loosen
   persistence / recovery hold when a much safer lane appears.
3. **Do not spend more effort on Hz** — too-late is 1 of 14.

Do not connect the framewise or event BC models. They do not address these
two failure modes.

## Controller changes (2026-08-28)

- Packed-enemy **crowd density** is scored inside 240 units, with extra
  weight when moving toward the cluster. This is a lane signal, not an HP
  rule.
- If a lane is at least `override_margin` safer, the arbiter switches even
  with previous-action stickiness.
- ESCAPE no longer uses HP. Early-wave packs are left to the policy lane
  chooser. ESCAPE pack-count only starts at **wave 8+**, or when every lane
  is already high-risk.

## 24 Hz live check after those changes

Three full-restart episodes, same checkpoint, 8 CPU threads.

| Trial | Waves | Mean wave | Effective Hz |
|---|---|---:|---:|
| Before crowd/ESCAPE change | 8, 6, 3 | 5.7 | ~9.5 |
| Early-wave ESCAPE on pack | 3, 2, 6 | 3.7 | ~13.0 |
| Policy-lane pack (no HP) | 7, 5, 6 | 6.0 | ~13.1 |

n = 3. Policy-lane pack removed the wave-2 panic. Mean wave is in line
with the original, not a clear survival jump. Production rate stays 24 Hz.
