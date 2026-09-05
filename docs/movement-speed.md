# Movement speed telemetry

Bridge 0.3.23 estimates movement speed from the player's `linear_velocity`
instead of treating the first speed-named property as world units per second.
A live Brotato 1.1.15.4 wave-one probe reported `combat.move_speed = 1500`
while linear velocity was 472.001923 and position changes agreed with that
velocity. This inflated movement predictions used by both bridge path risks
and the Python safety shield.

Only nonzero velocity aligned with a current movement command is sampled.
The estimate is smoothed and retained during idle. A different player instance
resets it; speed-stat changes rescale an existing estimate. Before the first
sample, the existing stat-based 300-unit fallback is used. The combat field
`move_speed_source` identifies measured velocity versus that fallback.

This is a measured estimate, not a new authoritative game speed property.
Acceleration, aligned knockback, and temporary speed effects can affect it.
Stationary, opposite, and sideways samples are excluded. A full-run win-rate
comparison remains necessary before claiming that this improves survival.

Validation: all 215 Python regression tests passed on Windows. The bridge
also needs live verification after installing the mod and restarting Brotato
with mods enabled; Python tests do not validate GDScript execution.
