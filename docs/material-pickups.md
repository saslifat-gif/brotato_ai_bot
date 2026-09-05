# Material pickup preference

The active controller now favors nearby material pickups, without retraining the
PPO model. Normal movement can steer toward money within 450 game units when its
estimated total risk is at most 0.20 and no more than 0.03 above the current lane.
Each hazard component is also limited to a 0.02 increase. These are estimated
risks, not guarantees that a route is safe.

Attraction rewards progress toward nearby clusters and higher-value materials,
with diminishing value weighting and a maximum of 24 nearby targets. Targets
within 12 units are ignored to avoid circling pickups already under the player.
A direction must offer a meaningful gain before it replaces the current action.
Normal pickup changes appear as `material_pickup` in decision traces.

During crowd recovery, a smaller 0.12 score preference breaks close choices among
low-risk directions. Existing separation penalties, escape hold timing and the
final safer-lane fallback remain in place. Below 35% health, the new preference
turns off to leave healing and escape choices alone. Healing pickups and crates
are not treated as money.

Validation: 208 unit tests passed on Windows, including nearby collection,
hazard rejection, low-health behavior, direction stability and escape scoring.
The change has not yet been measured for money gained per wave or survival in a
new live run. Compare those metrics before claiming an economic or win-rate gain.
