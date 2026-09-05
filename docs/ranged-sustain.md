# Ranged damage and sustain priorities

The default build profile is now `ranged_sustain`, following the requested
preference for ranged damage, life steal, and percent damage, in that order.
It adds 180, 140, and 100 points respectively for positive stat effects or
matching upgrades. Existing tier, cost, negative-stat scoring, and weapon-plan
rules still apply, so these are strong preferences rather than unconditional
purchases. Equal-sized positive stat choices follow the requested order.
Negative effects receive no priority bonus. Unaffordable items remain excluded.

`ranged_smg` remains available explicitly for reproducing the earlier baseline.
The new profile emits `ranged_sustain_teacher_v1` in UI decision logs.
The movement model and combat safety rules are unchanged. Training resumed
under this profile is a different build configuration from earlier evaluations.

Validation: 220 regression tests passed, including priority order across early
and late waves, negative effects, and unaffordable preferred items.
