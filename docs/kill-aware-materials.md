# Kill-aware material routes

Material targeting can relax ranged spacing when a normal enemy is taking sustained damage and is predicted to die before the player can contact it. Enemy collision/path, projectile, telegraph, and boundary risks are unchanged. Other enemies still contribute ranged spacing.

The bridge exposes enemy runtime IDs and health, but does not expose weapon damage or the weapon's current target. This implementation estimates short-term clearance from observed health loss rather than inventing weapon DPS. It requires at least three recent damage events, uses half the weakest-hit/longest-interval damage rate, predicts at most 0.45 seconds ahead, and requires a 0.20-second margin before earliest contact. Evidence expires after damage stops; healing, observation gaps, wave/session changes, and resets invalidate it. Bosses, elites, charging enemies, out-of-range targets, and missing IDs do not qualify.

The decision trace records `clearable_enemy_count`. This is a forecast, not a confirmed kill or a claim of improved collection. The original full hazard risks remain in traces, even when material routing uses a reduced spacing preference. Unit tests cover unchanged physical hazards, stale evidence, missing data, and imminent contact. Live collection and survival improvement still need evaluation after the current baseline run.
