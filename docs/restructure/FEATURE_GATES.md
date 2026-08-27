# Controller Feature Gates

Every future movement/controller feature must provide:

1. one named owner and one call site;
2. typed input and output contracts;
3. normal, missing-data, and conflict tests;
4. a fixed-recording comparison against the current baseline;
5. override, risk-reduction, switching, observed-damage-sample, and unique-damage-window metrics;
6. minimum-risk, unsafe-action, and requested-to-minimum-regret metrics when it changes arbitration;
7. pre/post exposure metrics when it changes hazard interpretation;
8. a safe-default feature flag;
9. short, stable telemetry tags;
10. a documented rollback;
11. a smoke test after a normal restart;
12. temporal zero-history/shuffled-history and BC held-out checks when it changes the model.

The maintained comparison set is policy_only, projectile_only, enemy_only, unified,
unified_stable, and noop_analyzer_control. In this set, policy_only is the
shield-off baseline and unified is the shield-on baseline. The no-op result must
exactly match policy-only modeled risk or the analyzer has drifted.

Replay output is geometric counterfactual analysis. It must never describe repeated
health-loss samples as unique hits or present modeled risk as proof of alternate
game outcomes.
