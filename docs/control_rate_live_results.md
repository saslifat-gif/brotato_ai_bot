# Live control-rate A/B results

Date: 2026-08-28  
Host: `lifat@192.168.1.3` (`C:\ml\brotato`)  
Bridge handshake: `mod=0.3.20`  
Frozen checkpoint: `models/version_3/ranged_smg_v2/v4_temporal_best/best_training_agent.zip` (160k)  
Launcher: `v4/run_frozen.py` with `BROTATO_V4_FULL_RESTART=1`  
Controller: unchanged V4 pipeline; one final action writer; hazard arbiter on

Each rate used three full EndRun restarts (Cancel RetryWave, then RestartButton).
Actions were held between control ticks by the bridge. Independent 60 Hz raw
recordings were written beside each rate when the recorder connected.

These are live physics outcomes. They are not the earlier passive replay.

## Rate sweep

| Rate | Waves | Mean wave | Effective Hz | HP loss | Projectile hits | All dead |
|---:|---|---:|---:|---:|---:|---|
| 10 Hz | 3, 8, 8 | 6.3 | 6.3 | 25, 90, 122 | 13, 15, 20 | yes |
| 15 Hz | 3, 8, 5 | 5.3 | 7.0 | 20, 114, 44 | 11, 25, 15 | yes |
| 20 Hz | 7, 10, 6 | 7.7 | 8.3 | 44, 224, 73 | 10, 39, 11 | yes |
| 24 Hz | 8, 6, 3 | 5.7 | 9.5 | 141, 49, 38 | 24, 11, 20 | yes |
| 30 Hz | 3, 8, 7 | 6.0 | 11.9 | 24, 94, 120 | 12, 12, 28 | yes |
| 60 Hz | 3, 7, 6 | 5.3 | 13.3 | 21, 145, 36 | 10, 23, 9 | yes |

Machine-readable files: `reports/live_rate_{10,15,20,24,30,60}hz.json`.

A first pass that clicked RetryWave Confirm was discarded for survival
comparison. Those retry-wave files remain as `reports/live_rate_*hz_retrywave.json`.

## What this does and does not show

The configure request is honored in direction: higher requested rates produce
higher processed rates. The live loop does not reach the request. 24 Hz
configure yields about 9.5 Hz; 60 Hz configure yields about 13 Hz.

Wave and health-loss ranges overlap across every rate. n = 3 is too small to
claim a causal survival improvement. There is no monotonic gain from 10 Hz to
60 Hz. 20 Hz had the highest mean wave (7.7) and also the largest health-loss
outlier (224).

Production control rate remains 24 Hz.

15 Hz live phase-offset trials were not run in this pass. The frozen runner
has no phase-offset schedule, and this rate sweep already shows no material
repeatable survival benefit from frequency alone.
