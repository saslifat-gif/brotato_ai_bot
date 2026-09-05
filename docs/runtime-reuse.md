# Reuse state and geometry during control (bridge 0.3.26)

The hazard scorer and recovery controller now read the immutable state payload
instead of separately thawing the entire snapshot. Recovery caches separation
by action and its nearest-enemy frame only inside one apply call. A finally
block clears the cache, including on errors, so equal ticks or mutated input
objects cannot reuse stale geometry on subsequent decisions.

The bridge exports per-stage milliseconds in `bridge_profile_ms` and includes
these in sampled slow-state logs. A 600-step probe identified projectile export
as the largest measured stage (8.47 ms mean). Its property lookup repeatedly
resolved the same schema for alternative property names. Resolving that schema
once per candidate list preserves ordered/null-skipping lookup while reducing
this stage to 6.34 ms in a follow-up 600-step probe. Maximum enemy counts were
7 and 12 respectively; these are live timing probes, not deterministic replays.
No scan coverage or cadence was reduced.

Validation: 217 Python tests passed, including cache lifetime and immutable
snapshot checks. A 1,631-state offline replay produced identical comparison
outputs before/after; Mac processing time fell from 4.59 to 2.12 seconds.
Both instrumented bridge builds loaded in the live game and completed their
600-step probes. Model weights and training settings are unchanged.
