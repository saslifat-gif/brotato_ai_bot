# Brotato AI Bot (v1)

Windows-only reinforcement learning project for **Brotato**.

A reinforcement learning bot for Brotato running on **Windows**. Uses PPO to train a policy network, with modules for the game environment, reward system, shop strategy, and runtime control.

## Features

- PPO training pipeline: `v1/train.py`
- Custom game environment: `v1/env/brotato_env.py`
- Reward engine: `v1/reward/reward_engine.py`
- Shop strategy & OCR: `v1/shop/`
- Utility scripts: HP annotation, YOLO classification data prep, etc.

## Requirements

- OS: Windows 10/11
- Python: 3.11 recommended
- Game window: Brotato (window title/process name configurable via environment variables)

## Quick Start

1. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables (optional)
   - Copy `.env.example` and edit as needed.
   - If using the Roboflow detector, provide a `ROBOFLOW_API_KEY`.

4. Start training
```bash
python v1/train.py
```
Or use the batch script:
```bat
train_mask.bat
```

## Testing

Default unit test directory is `test/v1/unit`:
```bash
pytest
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ROBOFLOW_API_KEY` | Optional | Enables the Roboflow detection path |
| `BROTATO_OUTPUT_DIR` | Optional | Output directory for models and checkpoints |
| `BROTATO_WINDOW_TITLE` | Optional | Game window title |
| `BROTATO_EXE_NAME` | Optional | Game executable name (e.g. `Brotato.exe`) |

Full configuration options: `v1/config/runtime_config.py`

## Training Hotkeys

| Key | Action |
|-----|--------|
| `F7` | Start/pause automation |
| `F8` | Request training stop and save |
| `F6` | Show/hide debug window |

## Project Structure

```
.
├─v1/                 # Core training and environment code
│  ├─config/          # Runtime configuration
│  ├─env/             # Game environment and detection adapters
│  ├─reward/          # Reward calculation
│  ├─runtime/         # Input, stop control, debug window
│  └─shop/            # Shop strategy and OCR
├─test/               # Tests
├─raw_models/         # Raw models, assets, experimental scripts
├─train_mask.bat      # Windows launch script
└─requirements.txt    # Dependencies
```

## License

MIT — see `LICENSE`.
