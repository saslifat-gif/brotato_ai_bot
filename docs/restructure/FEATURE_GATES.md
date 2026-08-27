# Controller Feature Gates

Every future movement/controller feature must provide:

1. one named owner and one call site;
2. typed input and output contracts;
3. normal, missing-data, and conflict tests;
4. a fixed-recording comparison against the current baseline;
5. override, risk-reduction, switching, observed-damage-sample, and unique-damage-window metrics;
6. a safe-default feature flag;
7. short, stable telemetry tags;
8. a documented rollback;
9. a smoke test after a normal restart.

The maintained comparison set is `policy_only`, `projectile_only`, `enemy_only`, `unified`, `unified_stable`, and `noop_analyzer_control`. The no-op result must exactly match policy-only modeled risk or the analyzer has drifted.

Replay output is a geometric counterfactual. It must never describe repeated health-loss samples as unique hits or present modeled risk as proof of alternate game outcomes.

