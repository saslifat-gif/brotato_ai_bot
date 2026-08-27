# V4 Restructure Baseline

Captured on 2026-08-27 before structural changes.

## Source snapshot

- Windows repository: `C:\ml\brotato`
- Original branch: `feat/v3-api-agent`
- Dedicated work branch: `feat/v4-restructure`
- Original HEAD: `84e3cd9 fix: distinguish hostile projectile telemetry`
- Remote relation: 19 commits ahead of `origin/feat/v3-api-agent`
- Original unstaged binary-diff object: `2518ce457f4efe21a237519985058bc2ab48c099`
- Untracked contract: `V4_RESTRUCTURE_HANDOFF.md`
- Protected unrelated path: `v1/shop/ocr_winmedia.py`; never stage, edit, revert, or overwrite it.

The original working tree also contained active v3/v4 changes in:

- `test/v1/unit/test_v3_api.py`
- `v3/README.md`
- `v3/combat_policy.py`
- `v3/config.py`
- `v3/env/brotato_api_env.py`
- `v3/train_combat_finetune.py`

## Runtime snapshot

- Action bridge: port 4242; free at capture.
- Raw recorder: port 4243; free at capture.
- TensorBoard: port 6007; active, PID 27944.
- TensorBoard log root: `models/version_3/ranged_smg_v2/logs`.
- A pre-existing `python -m pytest -q` process, PID 64756, was left untouched.
- Training environment: `C:\Users\lifat\miniconda3\envs\bota_ai`.
- Active model directory: `models/version_3/ranged_smg_v2`.
- Scheduled resume order: newest `v4_temporal_checkpoints/v4_temporal_ppo_*_steps.zip`, then `v4_temporal_bootstrap.zip`, otherwise fresh transfer.
- Transfer source: `models/version_3/bullet_hell_finetune_best/best_training_agent.zip`.
- Semantic anchors: `models/version_3/human_semantic_combat_v2.jsonl`.

## Baseline measurements

These are evidence labels, not claims about counterfactual game outcomes.

- Previously observed live control stream: approximately 14.65 Hz.
- Latest persisted TensorBoard control sample at step 101,200: 15.39997 Hz.
- Requested control rate: 24 Hz.
- Raw recorder target: approximately 60 Hz.
- Raw library at capture: 20 files, 4,211,665,994 bytes (3.922 GiB).
- Previous stable unified selector backtest: mean modeled risk 0.216, minimum-risk action 84.1%, override rate 34.5%.
- Previous unified selector without switch penalty: mean modeled risk 0.215, minimum-risk action 87.7%, override rate 34.7%.
- Switch penalty reduced direction changes by approximately 13.5% in the prior fixed-recording comparison.
- Latest persisted TensorBoard run reached wave 7, recorded 128 cumulative death events, mean damage sample 0.11816, and reversal rate 0.20996 at step 101,200.

Wave survival, deaths, damage windows, and victory rate must be recomputed from a bounded live run after a normal restart. They were not safely inferable from the process table at capture.
