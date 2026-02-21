import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


_DEFAULT_SHOP_POINTS = [
    (195, 583),
    (560, 577),
    (912, 583),
    (1291, 574),
]


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _parse_point_list(raw: str, default: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    s = str(raw or "").strip()
    if not s:
        return list(default)
    out: List[Tuple[int, int]] = []
    for token in s.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(",")]
        if len(parts) != 2:
            continue
        try:
            out.append((int(parts[0]), int(parts[1])))
        except Exception:
            continue
    return out if out else list(default)


def _parse_rect(raw: str, default: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    s = str(raw or "").strip()
    if not s:
        return default
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return default
    try:
        x1, y1, x2, y2 = [int(v) for v in parts]
    except Exception:
        return default
    lx, rx = sorted((x1, x2))
    ty, by = sorted((y1, y2))
    if rx <= lx or by <= ty:
        return default
    return (lx, ty, rx, by)


def _parse_optional_rect(raw: str) -> Optional[Tuple[int, int, int, int]]:
    s = str(raw or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in parts]
    except Exception:
        return None
    lx, rx = sorted((x1, x2))
    ty, by = sorted((y1, y2))
    if rx <= lx or by <= ty:
        return None
    return (lx, ty, rx, by)


def _parse_point(raw: str, default: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    s = str(raw or "").strip()
    if not s:
        return default
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        return default
    try:
        x, y = int(parts[0]), int(parts[1])
    except Exception:
        return default
    if x < 0 or y < 0:
        return default
    return (x, y)


def _parse_lock_rects(
    raw: str,
    shop_points: List[Tuple[int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Parse explicit lock rects or auto-generate from shop_points.

    In Brotato, each shop item card has a small padlock icon at the
    bottom-left.  The default rects are generated 60px below each
    shop_point and 80px to the left, covering a 50x40 region.
    """
    s = str(raw or "").strip()
    if s:
        rects: List[Tuple[int, int, int, int]] = []
        for token in s.split(";"):
            token = token.strip()
            if not token:
                continue
            parts = [p.strip() for p in token.split(",")]
            if len(parts) != 4:
                continue
            try:
                rects.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
            except Exception:
                continue
        if rects:
            return rects
    # Auto-generate default lock rects from shop_points
    out: List[Tuple[int, int, int, int]] = []
    for px, py in shop_points:
        # Lock icon is roughly 80px left, 40-80px below the item center
        lx = max(0, px - 120)
        ly = py + 30
        out.append((lx, ly, lx + 70, ly + 50))
    return out


def _parse_rect_list(raw: str) -> List[Tuple[int, int, int, int]]:
    s = str(raw or "").strip()
    if not s:
        return []
    out: List[Tuple[int, int, int, int]] = []
    for token in s.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(",")]
        if len(parts) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in parts]
        except Exception:
            continue
        lx, rx = sorted((x1, x2))
        ty, by = sorted((y1, y2))
        if rx <= lx or by <= ty:
            continue
        out.append((lx, ty, rx, by))
    return out


@dataclass
class RuntimeConfig:
    window_title: str
    exe_name: str
    force_resize: bool
    window_w: int
    window_h: int
    raw_models_dir: str

    capture_backend: str
    output_idx: int
    align_wait_sec: float

    stop_mode: str
    input_mode: str
    input_physical_fallback: bool
    input_move_physical: bool

    debug_windows_enabled: bool
    debug_windows_set: str
    debug_render_fps: int

    require_arm_hotkey: bool
    enable_quit_hotkey: bool
    hotkey_debounce_sec: float

    hp_rect: Tuple[int, int, int, int]

    use_state_model: bool
    state_model_path: str
    state_image_size: int
    state_device: str
    state_shop_score: float
    state_upgrade_score: float
    state_gameover_score: float
    state_non_battle_min_score: float
    state_infer_interval: int
    state_non_battle_hold_sec: float

    shop_policy_enable: bool
    shop_target: str
    shop_buy_min_score: float
    shop_refresh_max: int
    shop_max_buys: int
    shop_points: List[Tuple[int, int]]
    shop_refresh_rect: Tuple[int, int, int, int]
    shop_go_rect: Tuple[int, int, int, int]
    shop_card_w: int
    shop_card_h: int
    shop_card_down: int
    shop_card_rects: List[Tuple[int, int, int, int]]
    shop_lock_rects: List[Tuple[int, int, int, int]]
    shop_buy_offset_y: int
    shop_click_confirm_frames: int
    shop_click_confirm_hamming: int
    shop_buy_retry_max: int
    shop_action_cooldown_sec: float
    shop_entry_lock_sec: float
    menu_action_cooldown_sec: float
    item_pick_take_rect: Tuple[int, int, int, int]
    item_pick_recycle_rect: Tuple[int, int, int, int]
    upgrade_select_rect: Optional[Tuple[int, int, int, int]]
    upgrade_select_points: List[Tuple[int, int]]
    upgrade_card_w: int
    upgrade_card_h: int
    upgrade_ocr_enable: bool
    upgrade_refresh_avoid_rect: Optional[Tuple[int, int, int, int]]
    upgrade_press_enter_confirm: bool
    gameover_click_pos: Optional[Tuple[int, int]]
    gameover_restart_rect: Tuple[int, int, int, int]
    gameover_new_game_rect: Tuple[int, int, int, int]

    ocr_backend: str
    ocr_lang: str
    ocr_min_conf: float
    ocr_async_max_age_sec: float

    obs_size: int
    obs_channels: int
    obs_danger_enable: bool
    obs_mask_mode: bool
    obs_stack: int
    frame_skip: int
    action_sleep_sec: float
    anti_stuck_enable: bool
    anti_stuck_same_action_steps: int
    anti_stuck_low_motion_steps: int
    anti_stuck_break_steps: int
    anti_stuck_motion_threshold: float

    reward_alive: float
    reward_non_battle_penalty: float
    reward_damage_scale: float
    reward_idle_penalty: float
    reward_activity_bonus: float
    reward_loot_bonus: float
    reward_kill_bonus: float           # per-kill reward (loot-spawn proxy)
    reward_kill_spawn_trigger: float   # min loot_spawn ratio to count as kill
    reward_death_penalty: float
    reward_time_penalty: float     # per-step hunger-games drain in battle
    idle_diff_threshold: float
    loot_delta_trigger: float
    death_confirm_frames: int

    ppo_learning_rate: float
    ppo_lr_end: float              # linear schedule end LR (1.0 → 0.0 progress)
    ppo_ent_coef: float
    ppo_n_steps: int
    ppo_batch_size: int
    ppo_n_epochs: int
    ppo_total_timesteps: int
    ppo_progress_bar: bool
    reset_num_timesteps: bool
    resume_model: str
    torch_threads: int
    torch_interop_threads: int

    # Savage Protocol shop overrides
    shop_stick_priority: bool      # force-buy stick/primitive-weapon slots first
    shop_stick_min_score: float    # OCR confidence threshold to recognise a stick
    shop_max_weapons: int          # max weapon slots before treating as "full"
    shop_refresh_settle_sec: float # seconds to wait after refresh click before trusting OCR

    def __post_init__(self):
        # PPO hyperparameter validation
        if self.ppo_learning_rate <= 0:
            raise ValueError(f"ppo_learning_rate must be positive, got {self.ppo_learning_rate}")
        if self.ppo_n_steps < self.ppo_batch_size:
            raise ValueError(
                f"ppo_n_steps ({self.ppo_n_steps}) must be >= ppo_batch_size ({self.ppo_batch_size})"
            )
        if self.ppo_n_steps % self.ppo_batch_size != 0:
            raise ValueError(
                f"ppo_n_steps ({self.ppo_n_steps}) must be divisible by ppo_batch_size ({self.ppo_batch_size})"
            )
        if self.ppo_total_timesteps <= 0:
            raise ValueError(f"ppo_total_timesteps must be positive, got {self.ppo_total_timesteps}")

        # HP rect validation
        x1, y1, x2, y2 = self.hp_rect
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid hp_rect: {self.hp_rect}")

        # Reward parameter validation
        if self.reward_damage_scale < 0:
            raise ValueError(f"reward_damage_scale must be >= 0, got {self.reward_damage_scale}")
        if self.reward_death_penalty < 0:
            raise ValueError(f"reward_death_penalty must be >= 0, got {self.reward_death_penalty}")

        # Observation validation
        if self.obs_size < 32 or self.obs_size > 512:
            print(f"[config] WARNING: obs_size={self.obs_size} is unusual (expected 32-512)")


def _get_env(
    key: str,
    default,
    cast: Callable[[str], object],
    compat: Optional[List[str]] = None,
):
    compat = compat or []
    if key in os.environ:
        return cast(os.environ.get(key, ""))
    for old in compat:
        if old in os.environ:
            val = os.environ.get(old, "")
            print(f"[config] deprecated env {old} -> {key}")
            return cast(val)
    return default


def load_runtime_config() -> RuntimeConfig:
    window_title = str(_get_env("BROTATO_WINDOW_TITLE", "Brotato", str))
    exe_name = str(_get_env("BROTATO_EXE_NAME", "Brotato.exe", str))
    force_resize = bool(_get_env("BROTATO_FORCE_WINDOW_RESIZE", False, _to_bool, ["BROTATO_FORCE_WINDOW_RESIZE"]))
    window_w = int(_get_env("BROTATO_WINDOW_W", 1920, int))
    window_h = int(_get_env("BROTATO_WINDOW_H", 1080, int))
    raw_models_dir = str(_get_env("BROTATO_RAW_MODELS_DIR", "raw_models", str)).strip()

    capture_backend = str(_get_env("BROTATO_CAPTURE_BACKEND", "windows-capture", str)).strip().lower()
    output_idx = int(_get_env("BROTATO_OUTPUT_IDX", -1, int))
    align_wait_sec = float(_get_env("BROTATO_ALIGN_WAIT", 3.0, float, ["BROTATO_ALIGN_WAIT"]))

    stop_mode = str(_get_env("BROTATO_STOP_MODE", "safe", str)).strip().lower()
    input_mode = str(_get_env("BROTATO_INPUT_MODE", "safe_background", str)).strip().lower()
    input_physical_fallback = bool(_get_env("BROTATO_INPUT_PHYSICAL_FALLBACK", False, _to_bool))
    input_move_physical = bool(_get_env("BROTATO_INPUT_MOVE_PHYSICAL", True, _to_bool))

    debug_windows_enabled = bool(_get_env("BROTATO_DEBUG_WINDOWS", True, _to_bool))
    debug_windows_set = str(_get_env("BROTATO_DEBUG_WINDOWS_SET", "core4", str)).strip().lower()
    debug_render_fps = max(1, int(_get_env("BROTATO_DEBUG_RENDER_FPS", 12, int, ["BROTATO_DEBUG_RENDER_FPS"])))

    require_arm_hotkey = bool(_get_env("BROTATO_REQUIRE_ARM", True, _to_bool))
    enable_quit_hotkey = bool(_get_env("BROTATO_ENABLE_QUIT_HOTKEY", True, _to_bool))
    hotkey_debounce_sec = max(0.05, float(_get_env("BROTATO_HOTKEY_DEBOUNCE_SEC", 0.25, float)))

    hp_rect = _parse_rect(
        str(_get_env("BROTATO_HP_RECT", "23,22,342,70", str, ["BROTATO_HP_RECT"])),
        (23, 22, 342, 70),
    )

    use_state_model = bool(_get_env("BROTATO_USE_STATE_MODEL", True, _to_bool, ["BROTATO_USE_STATE_MODEL"]))
    state_model_path = str(_get_env("BROTATO_STATE_YOLO_MODEL_PATH", "auto", str, ["BROTATO_STATE_YOLO_MODEL_PATH"]))
    state_image_size = max(64, int(_get_env("BROTATO_STATE_IMAGE_SIZE", 256, int)))
    state_device = str(_get_env("BROTATO_STATE_YOLO_DEVICE", "0", str))
    state_shop_score = float(_get_env("BROTATO_STATE_SHOP_SCORE", 0.62, float))
    state_upgrade_score = float(_get_env("BROTATO_STATE_UPGRADE_SCORE", 0.62, float))
    state_gameover_score = float(_get_env("BROTATO_STATE_GAMEOVER_SCORE", 0.94, float))
    state_non_battle_min_score = float(_get_env("BROTATO_STATE_NON_BATTLE_MIN_SCORE", 0.72, float))
    state_infer_interval = max(1, int(_get_env("BROTATO_STATE_INFER_INTERVAL", 4, int)))
    state_non_battle_hold_sec = max(0.0, float(_get_env("BROTATO_STATE_NON_BATTLE_HOLD_SEC", 1.20, float)))

    shop_policy_enable = bool(_get_env("BROTATO_SHOP_POLICY_ENABLE", True, _to_bool))
    shop_target = str(_get_env("BROTATO_SHOP_TARGET", "primitive_only", str)).strip().lower()
    shop_buy_min_score = float(_get_env("BROTATO_SHOP_POLICY_BUY_MIN_SCORE", 0.40, float))
    shop_refresh_max = max(0, int(_get_env("BROTATO_SHOP_POLICY_REFRESH_MAX", 6, int)))
    shop_max_buys = max(1, int(_get_env("BROTATO_SHOP_POLICY_MAX_BUYS", 4, int)))
    shop_points = _parse_point_list(
        str(_get_env("BROTATO_SHOP_POINTS", "", str)),
        _DEFAULT_SHOP_POINTS,
    )
    shop_refresh_rect = _parse_rect(
        str(_get_env("BROTATO_SHOP_REFRESH_RECT", "1324,63,1457,104", str)),
        (1324, 63, 1457, 104),
    )
    shop_go_rect = _parse_rect(
        str(_get_env("BROTATO_SHOP_GO_RECT", "1486,821,1887,890", str)),
        (1486, 821, 1887, 890),
    )
    shop_card_w = max(160, int(_get_env("BROTATO_SHOP_POLICY_CARD_W", 320, int)))
    shop_card_h = max(120, int(_get_env("BROTATO_SHOP_POLICY_CARD_H", 420, int)))
    shop_card_down = max(10, int(_get_env("BROTATO_SHOP_POLICY_CARD_DOWN", 80, int)))
    shop_card_rects = _parse_rect_list(str(_get_env("BROTATO_SHOP_CARD_RECTS", "", str)))
    shop_lock_rects = _parse_lock_rects(
        str(_get_env("BROTATO_SHOP_LOCK_RECTS", "", str)),
        shop_points,
    )
    shop_buy_offset_y = int(_get_env("BROTATO_SHOP_BUY_OFFSET_Y", -40, int))
    shop_click_confirm_frames = max(1, int(_get_env("BROTATO_SHOP_CLICK_CONFIRM_FRAMES", 6, int)))
    shop_click_confirm_hamming = max(1, int(_get_env("BROTATO_SHOP_CLICK_CONFIRM_HAMMING", 8, int)))
    shop_buy_retry_max = max(0, int(_get_env("BROTATO_SHOP_BUY_RETRY_MAX", 2, int)))
    shop_action_cooldown_sec = max(0.05, float(_get_env("BROTATO_PIXEL_UI_ACTION_COOLDOWN", 0.08, float)))
    shop_entry_lock_sec = max(0.0, float(_get_env("BROTATO_SHOP_ENTRY_LOCK_SEC", 0.20, float)))
    menu_action_cooldown_sec = max(0.05, float(_get_env("BROTATO_MENU_ACTION_COOLDOWN", 0.25, float)))
    item_pick_take_rect = _parse_rect(
        str(_get_env("BROTATO_ITEM_PICK_TAKE_RECT", "435,664,1037,737", str)),
        (435, 664, 1037, 737),
    )
    item_pick_recycle_rect = _parse_rect(
        str(_get_env("BROTATO_ITEM_PICK_RECYCLE_RECT", "436,737,1038,811", str)),
        (436, 737, 1038, 811),
    )
    upgrade_select_rect = _parse_optional_rect(
        str(_get_env("BROTATO_UPGRADE_SELECT_RECT", "", str)),
    )
    upgrade_select_points = _parse_point_list(
        str(_get_env("BROTATO_UPGRADE_SELECT_POINTS", "245,615;555,615;865,615;1175,615", str)),
        [(245, 615), (555, 615), (865, 615), (1175, 615)],
    )
    upgrade_card_w = max(100, int(_get_env("BROTATO_UPGRADE_CARD_W", 280, int)))
    upgrade_card_h = max(60, int(_get_env("BROTATO_UPGRADE_CARD_H", 120, int)))
    upgrade_ocr_enable = bool(_get_env("BROTATO_UPGRADE_OCR_ENABLE", True, _to_bool))
    upgrade_refresh_avoid_rect = _parse_optional_rect(
        str(_get_env("BROTATO_UPGRADE_REFRESH_RECT", "", str)),
    )
    upgrade_press_enter_confirm = bool(_get_env("BROTATO_UPGRADE_PRESS_ENTER_CONFIRM", False, _to_bool))
    gameover_click_pos = _parse_point(
        str(_get_env("BROTATO_GAMEOVER_CLICK_POS", "", str)),
        None,
    )
    gameover_restart_rect = _parse_rect(
        str(_get_env("BROTATO_GAMEOVER_RESTART_RECT", "337,979,636,1045", str)),
        (337, 979, 636, 1045),
    )
    gameover_new_game_rect = _parse_rect(
        str(_get_env("BROTATO_GAMEOVER_NEW_GAME_RECT", "662,978,959,1048", str)),
        (662, 978, 959, 1048),
    )

    ocr_backend = str(_get_env("BROTATO_OCR_BACKEND", "winmedia", str, ["BROTATO_SHOP_OCR_BACKEND"])).strip().lower()
    ocr_lang = str(_get_env("BROTATO_SHOP_OCR_LANG", "zh-Hans", str))
    ocr_min_conf = float(_get_env("BROTATO_SHOP_OCR_MIN_CONF", 0.30, float))
    ocr_async_max_age_sec = max(0.05, float(_get_env("BROTATO_SHOP_POLICY_ASYNC_MAX_AGE_SEC", 0.45, float)))

    obs_size = max(64, int(_get_env("BROTATO_OBS_SIZE", 160, int)))
    obs_danger_enable = bool(_get_env("BROTATO_OBS_DANGER_ENABLE", True, _to_bool))
    obs_mask_mode = bool(_get_env("BROTATO_OBS_MASK_MODE", False, _to_bool))
    obs_channels = 3 if (obs_danger_enable or obs_mask_mode) else 1
    obs_stack = max(1, int(_get_env("BROTATO_OBS_STACK", 4, int)))
    frame_skip = max(1, int(_get_env("BROTATO_FRAME_SKIP", 1, int)))
    action_sleep_sec = max(0.0, float(_get_env("BROTATO_ACTION_SLEEP_SEC", 0.0, float)))
    anti_stuck_enable = bool(_get_env("BROTATO_ANTI_STUCK_ENABLE", True, _to_bool))
    anti_stuck_same_action_steps = max(5, int(_get_env("BROTATO_ANTI_STUCK_SAME_ACTION_STEPS", 28, int)))
    anti_stuck_low_motion_steps = max(5, int(_get_env("BROTATO_ANTI_STUCK_LOW_MOTION_STEPS", 20, int)))
    anti_stuck_break_steps = max(1, int(_get_env("BROTATO_ANTI_STUCK_BREAK_STEPS", 6, int)))
    anti_stuck_motion_threshold = float(_get_env("BROTATO_ANTI_STUCK_MOTION_THRESHOLD", 1.1, float))

    reward_alive = float(_get_env("BROTATO_REWARD_ALIVE", 0.03, float))
    reward_non_battle_penalty = float(_get_env("BROTATO_REWARD_NON_BATTLE_PENALTY", 0.20, float))
    reward_damage_scale = float(_get_env("BROTATO_REWARD_DAMAGE_SCALE", 2.0, float))
    reward_idle_penalty = float(_get_env("BROTATO_REWARD_IDLE_PENALTY", 0.12, float))
    reward_activity_bonus = float(_get_env("BROTATO_REWARD_ACTIVITY_BONUS", 0.05, float))
    reward_loot_bonus = float(_get_env("BROTATO_REWARD_LOOT_COLLECT_BONUS", 2.0, float))
    reward_kill_bonus = float(_get_env("BROTATO_REWARD_KILL_BONUS", 1.5, float))
    reward_kill_spawn_trigger = float(_get_env("BROTATO_REWARD_KILL_SPAWN_TRIGGER", 0.015, float))
    reward_death_penalty = float(_get_env("BROTATO_REWARD_DEATH_PENALTY", 100.0, float))
    # Hunger-games per-step drain. 0.005 ≈ −5 pts/1000 steps; offset by loot/activity.
    reward_time_penalty = float(_get_env("BROTATO_REWARD_TIME_PENALTY", 0.005, float))
    idle_diff_threshold = float(_get_env("BROTATO_IDLE_DIFF_THRESHOLD", 1.5, float))
    loot_delta_trigger = float(_get_env("BROTATO_LOOT_DELTA_TRIGGER", 0.001, float))
    death_confirm_frames = max(1, int(_get_env("BROTATO_DEATH_CONFIRM_FRAMES", 3, int)))

    ppo_learning_rate = float(_get_env("BROTATO_PPO_LR", 3e-4, float))
    ppo_lr_end = float(_get_env("BROTATO_PPO_LR_END", 1e-5, float))
    # Raised from 0.01 → 0.02 to force more exploration (PPO_13 reset)
    ppo_ent_coef = float(_get_env("BROTATO_PPO_ENT_COEF", 0.02, float))
    ppo_n_steps = int(_get_env("BROTATO_PPO_N_STEPS", 4096, int))
    ppo_batch_size = int(_get_env("BROTATO_PPO_BATCH_SIZE", 512, int))
    ppo_n_epochs = int(_get_env("BROTATO_PPO_N_EPOCHS", 3, int))
    ppo_total_timesteps = int(_get_env("BROTATO_TOTAL_TIMESTEPS", 1_000_000, int))
    ppo_progress_bar = bool(_get_env("BROTATO_PPO_PROGRESS_BAR", True, _to_bool))
    reset_num_timesteps = bool(_get_env("BROTATO_RESET_NUM_TIMESTEPS", True, _to_bool))
    resume_model = str(_get_env("BROTATO_RESUME_MODEL", "auto", str))
    torch_threads = int(_get_env("BROTATO_TORCH_THREADS", 0, int))
    torch_interop_threads = int(_get_env("BROTATO_TORCH_INTEROP_THREADS", 0, int))

    shop_stick_priority = bool(_get_env("BROTATO_SHOP_STICK_PRIORITY", True, _to_bool))
    shop_stick_min_score = float(_get_env("BROTATO_SHOP_STICK_MIN_SCORE", 0.25, float))
    shop_max_weapons = max(1, int(_get_env("BROTATO_SHOP_MAX_WEAPONS", 6, int)))
    shop_refresh_settle_sec = max(0.0, float(_get_env("BROTATO_SHOP_REFRESH_SETTLE_SEC", 0.40, float)))

    if ocr_backend != "winmedia":
        print(f"[config] forcing OCR backend to winmedia (requested={ocr_backend})")
        ocr_backend = "winmedia"

    return RuntimeConfig(
        window_title=window_title,
        exe_name=exe_name,
        force_resize=force_resize,
        window_w=window_w,
        window_h=window_h,
        raw_models_dir=raw_models_dir,
        capture_backend=capture_backend,
        output_idx=output_idx,
        align_wait_sec=align_wait_sec,
        stop_mode=stop_mode,
        input_mode=input_mode,
        input_physical_fallback=input_physical_fallback,
        input_move_physical=input_move_physical,
        debug_windows_enabled=debug_windows_enabled,
        debug_windows_set=debug_windows_set,
        debug_render_fps=debug_render_fps,
        require_arm_hotkey=require_arm_hotkey,
        enable_quit_hotkey=enable_quit_hotkey,
        hotkey_debounce_sec=hotkey_debounce_sec,
        hp_rect=hp_rect,
        use_state_model=use_state_model,
        state_model_path=state_model_path,
        state_image_size=state_image_size,
        state_device=state_device,
        state_shop_score=state_shop_score,
        state_upgrade_score=state_upgrade_score,
        state_gameover_score=state_gameover_score,
        state_non_battle_min_score=state_non_battle_min_score,
        state_infer_interval=state_infer_interval,
        state_non_battle_hold_sec=state_non_battle_hold_sec,
        shop_policy_enable=shop_policy_enable,
        shop_target=shop_target,
        shop_buy_min_score=shop_buy_min_score,
        shop_refresh_max=shop_refresh_max,
        shop_max_buys=shop_max_buys,
        shop_points=shop_points,
        shop_refresh_rect=shop_refresh_rect,
        shop_go_rect=shop_go_rect,
        shop_card_w=shop_card_w,
        shop_card_h=shop_card_h,
        shop_card_down=shop_card_down,
        shop_card_rects=shop_card_rects,
        shop_lock_rects=shop_lock_rects,
        shop_buy_offset_y=shop_buy_offset_y,
        shop_click_confirm_frames=shop_click_confirm_frames,
        shop_click_confirm_hamming=shop_click_confirm_hamming,
        shop_buy_retry_max=shop_buy_retry_max,
        shop_action_cooldown_sec=shop_action_cooldown_sec,
        shop_entry_lock_sec=shop_entry_lock_sec,
        menu_action_cooldown_sec=menu_action_cooldown_sec,
        item_pick_take_rect=item_pick_take_rect,
        item_pick_recycle_rect=item_pick_recycle_rect,
        upgrade_select_rect=upgrade_select_rect,
        upgrade_select_points=upgrade_select_points,
        upgrade_card_w=upgrade_card_w,
        upgrade_card_h=upgrade_card_h,
        upgrade_ocr_enable=upgrade_ocr_enable,
        upgrade_refresh_avoid_rect=upgrade_refresh_avoid_rect,
        upgrade_press_enter_confirm=upgrade_press_enter_confirm,
        gameover_click_pos=gameover_click_pos,
        gameover_restart_rect=gameover_restart_rect,
        gameover_new_game_rect=gameover_new_game_rect,
        ocr_backend=ocr_backend,
        ocr_lang=ocr_lang,
        ocr_min_conf=ocr_min_conf,
        ocr_async_max_age_sec=ocr_async_max_age_sec,
        obs_size=obs_size,
        obs_channels=obs_channels,
        obs_danger_enable=obs_danger_enable,
        obs_mask_mode=obs_mask_mode,
        obs_stack=obs_stack,
        frame_skip=frame_skip,
        action_sleep_sec=action_sleep_sec,
        anti_stuck_enable=anti_stuck_enable,
        anti_stuck_same_action_steps=anti_stuck_same_action_steps,
        anti_stuck_low_motion_steps=anti_stuck_low_motion_steps,
        anti_stuck_break_steps=anti_stuck_break_steps,
        anti_stuck_motion_threshold=anti_stuck_motion_threshold,
        reward_alive=reward_alive,
        reward_non_battle_penalty=reward_non_battle_penalty,
        reward_damage_scale=reward_damage_scale,
        reward_idle_penalty=reward_idle_penalty,
        reward_activity_bonus=reward_activity_bonus,
        reward_loot_bonus=reward_loot_bonus,
        reward_kill_bonus=reward_kill_bonus,
        reward_kill_spawn_trigger=reward_kill_spawn_trigger,
        reward_death_penalty=reward_death_penalty,
        reward_time_penalty=reward_time_penalty,
        idle_diff_threshold=idle_diff_threshold,
        loot_delta_trigger=loot_delta_trigger,
        death_confirm_frames=death_confirm_frames,
        ppo_learning_rate=ppo_learning_rate,
        ppo_lr_end=ppo_lr_end,
        ppo_ent_coef=ppo_ent_coef,
        ppo_n_steps=ppo_n_steps,
        ppo_batch_size=ppo_batch_size,
        ppo_n_epochs=ppo_n_epochs,
        ppo_total_timesteps=ppo_total_timesteps,
        ppo_progress_bar=ppo_progress_bar,
        reset_num_timesteps=reset_num_timesteps,
        resume_model=resume_model,
        torch_threads=torch_threads,
        torch_interop_threads=torch_interop_threads,
        shop_stick_priority=shop_stick_priority,
        shop_stick_min_score=shop_stick_min_score,
        shop_max_weapons=shop_max_weapons,
        shop_refresh_settle_sec=shop_refresh_settle_sec,
    )
