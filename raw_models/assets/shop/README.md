Shop policy now supports OCR-only mode by default (no templates required).

Template folders (optional only if you enable template mode):

- `melee_damage/`
- `percent_damage/`
- `melee_weapon/`

How to use:

1. Put small cropped template images in the matching folder.
2. Templates should be from your current resolution/UI scale.
3. Keep each template tight around text/icon (avoid large background).
4. Add multiple variants (different rarity colors, language variants).

Runtime:

- The shop policy is enabled by default in `train.py`.
- Default mode is OCR-only + duplicate-item bonus.
- Template matching is optional and disabled by default.
- OCR keywords are used to detect:
  - melee damage (`近战伤害` / `melee damage`)
  - percent damage (`%伤害`)
  - melee weapon related words (`刀/剑/矛/...`)

PaddleOCR (optional):

1. Install:
   - `pip install paddleocr`
2. If OCR is unavailable, the policy falls back to template-only scoring.

Useful env overrides:

- `BROTATO_SHOP_POLICY_ENABLE=1`
- `BROTATO_SHOP_POLICY_MIN_STATE_SCORE=0.70`
- `BROTATO_SHOP_POLICY_BUY_MIN_SCORE=0.40`
- `BROTATO_SHOP_POLICY_REFRESH_MAX=2`
- `BROTATO_SHOP_POLICY_MAX_BUYS=4`
- `BROTATO_SHOP_TEMPLATE_DIR=<custom path>`
- `BROTATO_SHOP_OCR_ENABLE=1`
- `BROTATO_SHOP_OCR_MIN_CONF=0.45`
- `BROTATO_SHOP_OCR_LANG=ch`
- `BROTATO_SHOP_OCR_USE_GPU=0`
- `BROTATO_DEBUG_SHOW_SHOP_POLICY=1` (show live slot scores/actions window)
- `BROTATO_SHOP_POLICY_USE_TEMPLATES=0` (default OCR-only; set 1 to enable templates)
- `BROTATO_SHOP_POLICY_ALLOW_PIXEL_FALLBACK=0` (default: if OCR/template unavailable, do NOT click blindly)
