import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter
except Exception:
    Language = None
    BitmapPixelFormat = None
    SoftwareBitmap = None
    OcrEngine = None
    DataWriter = None


@dataclass
class SlotOcrScore:
    slot: int
    primitive_class: float
    primitive_weapon: float
    non_primitive: float
    melee_buff: float
    primary: float
    weapon_name: str
    lines: List[str]
    attack_buff: float = 0.0
    attack_speed_buff: float = 0.0


@dataclass
class OcrBatchResult:
    frame_id: int
    ts: float
    slots: List[SlotOcrScore]
    error: str = ""


class WinMediaOCR:
    def __init__(self, lang: str = "zh-Hans"):
        if OcrEngine is None or Language is None or SoftwareBitmap is None or DataWriter is None:
            raise RuntimeError(
                "winmedia deps missing; install: "
                "winrt-runtime winrt-Windows.Media.Ocr winrt-Windows.Graphics.Imaging "
                "winrt-Windows.Storage.Streams winrt-Windows.Globalization "
                "winrt-Windows.Foundation winrt-Windows.Foundation.Collections"
            )
        self.lang = str(lang or "zh-Hans")
        self.engine = OcrEngine.try_create_from_language(Language(self.lang))
        if self.engine is None:
            self.engine = OcrEngine.try_create_from_user_profile_languages()
        if self.engine is None:
            raise RuntimeError(f"cannot create Windows.Media.Ocr engine lang={self.lang}")
        self._loop = asyncio.new_event_loop()

    def close(self):
        try:
            self._loop.close()
        except Exception:
            pass

    def _to_software_bitmap(self, img_bgr: np.ndarray):
        if img_bgr is None or img_bgr.size == 0:
            raise RuntimeError("empty image")
        if len(img_bgr.shape) == 2:
            bgra = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGRA)
        elif img_bgr.shape[2] == 4:
            bgra = img_bgr
        else:
            bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        h, w = bgra.shape[:2]
        writer = DataWriter()
        writer.write_bytes(bgra.tobytes())
        buf = writer.detach_buffer()
        return SoftwareBitmap.create_copy_from_buffer(buf, BitmapPixelFormat.BGRA8, int(w), int(h))

    async def _recognize_async(self, sb):
        return await self.engine.recognize_async(sb)

    def predict(self, img_bgr: np.ndarray) -> List[Tuple[str, float]]:
        sb = self._to_software_bitmap(img_bgr)
        res = self._loop.run_until_complete(self._recognize_async(sb))
        out: List[Tuple[str, float]] = []
        try:
            lines = list(res.lines)
        except Exception:
            lines = []
        for ln in lines:
            txt = str(getattr(ln, "text", "")).strip()
            if txt:
                out.append((txt, 1.0))
        if not out:
            whole = str(getattr(res, "text", "")).strip()
            if whole:
                for line in whole.splitlines():
                    s = str(line).strip()
                    if s:
                        out.append((s, 1.0))
        return out


def normalize_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = (
        s.replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("％", "%")
        .replace("：", ":")
        .replace("，", ",")
        .replace("（", "(")
        .replace("）", ")")
    )
    return s


PRIMITIVE_CLASS_TOKENS = ("原始", "primitive")
# Strict branch whitelist tokens.
BRANCH_WEAPON_TOKENS = (
    "树枝",
    "樹枝",
    "branch",
)
# Primitive weapon tokens remain broad to identify weapon cards and keep
# them out of the accessory (attack/attack-speed) path.
PRIMITIVE_WEAPON_TOKENS = (
    *BRANCH_WEAPON_TOKENS,
    "木棍",
    "棍棒",
    "stick",
    "长矛",
    "弹弓",
    "火把",
    "石头",
    "spear",
    "slingshot",
    "torch",
    "rock",
)
NON_PRIMITIVE_TOKENS = (
    "枪",
    "霰弹",
    "散弹",
    "冲锋枪",
    "激光",
    "火箭",
    "手枪",
    "狙击",
    "魔杖",
    "法杖",
    "gun",
    "shotgun",
    "smg",
    "laser",
    "rocket",
    "wand",
)
MELEE_BUFF_TOKENS = (
    "近战", "melee",
    "攻击速度", "attackspeed",
    "生命偷取", "lifesteal", "吸血",
    "暴击", "crit",
    "闪避", "dodge", "evasion",
    "护甲", "armor",
    "最大生命", "maxhp",
    "速度", "speed",
)
ATTACK_BUFF_TOKENS = ("攻击", "伤害", "damage", "dmg")
ATTACK_SPEED_TOKENS = ("攻击速度", "攻速", "attackspeed", "aspd")

def _looks_like_stat_modifier_line(t: str) -> bool:
    """Heuristic: item stat lines usually have +/- modifiers, not weapon stat labels."""
    if not t:
        return False
    has_delta = ("+" in t) or ("-" in t) or ("%" in t)
    # Weapon cards usually look like '伤害:xx / 冷却:xx / 范围:xx'
    has_weapon_label_sep = ":" in t
    return bool(has_delta and (not has_weapon_label_sep))


def score_primitive_text(
    text: str, conf: float, min_conf: float
) -> Tuple[float, float, float, float, float, float, float, str]:
    """Returns (primitive_class, primitive_weapon, non_primitive, melee_buff, attack_buff, attack_speed_buff, primary, weapon_name)."""
    t = normalize_text(text)
    c = float(np.clip(conf, 0.0, 1.0))
    if c < min_conf or not t:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "")

    primitive_class = c if any(tok in t for tok in PRIMITIVE_CLASS_TOKENS) else 0.0
    primitive_weapon = c if any(tok in t for tok in PRIMITIVE_WEAPON_TOKENS) else 0.0
    non_primitive = c if any(tok in t for tok in NON_PRIMITIVE_TOKENS) else 0.0
    melee_buff = c if any(tok in t for tok in MELEE_BUFF_TOKENS) else 0.0
    is_modifier_line = _looks_like_stat_modifier_line(t)
    is_attack_speed = any(tok in t for tok in ATTACK_SPEED_TOKENS)
    is_attack = any(tok in t for tok in ATTACK_BUFF_TOKENS) and (not is_attack_speed)
    attack_speed_buff = c if (is_modifier_line and is_attack_speed) else 0.0
    attack_buff = c if (is_modifier_line and is_attack) else 0.0

    # Extract weapon name for merge tracking
    weapon_name = ""
    for tok in PRIMITIVE_WEAPON_TOKENS:
        if tok in t:
            weapon_name = tok
            break

    if non_primitive > 0.0 and primitive_class <= 0.0 and primitive_weapon <= 0.0:
        return (0.0, 0.0, non_primitive, 0.0, 0.0, 0.0, 0.0, "")

    primary = max(
        primitive_class,
        primitive_weapon * 0.95,
        melee_buff * 0.65,
        attack_buff * 0.88,
        attack_speed_buff * 0.95,
    )
    return (
        primitive_class,
        primitive_weapon,
        non_primitive,
        melee_buff,
        attack_buff,
        attack_speed_buff,
        primary,
        weapon_name,
    )


def score_upgrade_text(text: str, conf: float, min_conf: float) -> float:
    """Score an upgrade option for the savage/primitive stick build. Returns a float score."""
    t = normalize_text(text)
    c = float(np.clip(conf, 0.0, 1.0))
    if c < min_conf or not t:
        return 0.0

    high_tokens = ("近战", "melee", "攻击速度", "attackspeed", "生命偷取", "lifesteal", "吸血")
    mid_high_tokens = ("闪避", "dodge", "暴击", "crit", "护甲", "armor")
    mid_tokens = ("伤害", "damage", "速度", "speed", "最大生命", "maxhp")
    penalty_tokens = ("远程", "ranged", "范围", "range")

    score = 0.0
    for tok in high_tokens:
        if tok in t:
            score += 0.9 * c
            break
    for tok in mid_high_tokens:
        if tok in t:
            score += 0.7 * c
            break
    for tok in mid_tokens:
        if tok in t:
            score += 0.5 * c
            break
    for tok in penalty_tokens:
        if tok in t:
            score -= 0.3 * c
            break
    return score


class ShopOcrWorker:
    def __init__(self, lang: str = "zh-Hans", min_conf: float = 0.30):
        self.lang = str(lang)
        self.min_conf = float(min_conf)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._job_frame_id = -1
        self._job_cards: Optional[Dict[int, np.ndarray]] = None
        self._last_result: Optional[OcrBatchResult] = None
        self._last_error = ""
        self._last_restart_try_ts = 0.0

    def start(self):
        if self._running:
            return
        self._last_error = ""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.8)
            self._thread = None

    def submit(self, frame_id: int, cards: Dict[int, np.ndarray]):
        if not self._running:
            now = time.time()
            if now - float(self._last_restart_try_ts) >= 2.0:
                self._last_restart_try_ts = now
                try:
                    self.start()
                except Exception:
                    pass
            return
        job_cards: Dict[int, np.ndarray] = {}
        for idx, card in cards.items():
            if card is None or card.size == 0:
                continue
            job_cards[int(idx)] = card.copy()
        with self._lock:
            self._job_frame_id = int(frame_id)
            self._job_cards = job_cards

    def get_latest(self, max_age_sec: float) -> Optional[OcrBatchResult]:
        with self._lock:
            res = self._last_result
        if res is None:
            return None
        age = time.time() - float(res.ts)
        if age > float(max_age_sec):
            return None
        return res

    def is_running(self) -> bool:
        th = self._thread
        return bool(self._running and (th is not None) and th.is_alive())

    def last_error(self) -> str:
        return str(self._last_error or "")

    def _loop(self):
        engine: Optional[WinMediaOCR] = None
        try:
            engine = WinMediaOCR(lang=self.lang)
            print(f"[shop] ocr init ok backend=winmedia lang={self.lang}")
        except Exception as e:
            self._last_error = str(e)
            print(f"[shop] ocr init failed: {e}")
            self._running = False
            return

        while self._running:
            frame_id = -1
            cards: Optional[Dict[int, np.ndarray]] = None
            with self._lock:
                if self._job_cards is not None:
                    frame_id = int(self._job_frame_id)
                    cards = self._job_cards
                    self._job_cards = None
            if cards is None:
                time.sleep(0.005)
                continue

            slots: List[SlotOcrScore] = []
            batch_error = ""
            for slot_idx, card in cards.items():
                try:
                    lines_raw = engine.predict(card)
                    lines = [str(t) for t, _ in lines_raw][:10]
                    p_class = 0.0
                    p_weapon = 0.0
                    non_prim = 0.0
                    m_buff = 0.0
                    atk_buff = 0.0
                    atk_spd_buff = 0.0
                    primary = 0.0
                    w_name = ""
                    for txt, conf in lines_raw:
                        s1, s2, s3, s4, s5, s6, s7, wn = score_primitive_text(txt, conf, self.min_conf)
                        p_class = max(p_class, s1)
                        p_weapon = max(p_weapon, s2)
                        non_prim = max(non_prim, s3)
                        m_buff = max(m_buff, s4)
                        atk_buff = max(atk_buff, s5)
                        atk_spd_buff = max(atk_spd_buff, s6)
                        primary = max(primary, s7)
                        if wn and not w_name:
                            w_name = wn
                    slots.append(
                        SlotOcrScore(
                            slot=int(slot_idx),
                            primitive_class=float(p_class),
                            primitive_weapon=float(p_weapon),
                            non_primitive=float(non_prim),
                            melee_buff=float(m_buff),
                            primary=float(primary),
                            weapon_name=str(w_name),
                            lines=lines,
                            attack_buff=float(atk_buff),
                            attack_speed_buff=float(atk_spd_buff),
                        )
                    )
                except Exception as e:
                    batch_error = str(e)
                    slots.append(
                        SlotOcrScore(
                            slot=int(slot_idx),
                            primitive_class=0.0,
                            primitive_weapon=0.0,
                            non_primitive=0.0,
                            melee_buff=0.0,
                            primary=0.0,
                            weapon_name="",
                            lines=[f"ocr-error:{e}"],
                            attack_buff=0.0,
                            attack_speed_buff=0.0,
                        )
                    )

            result = OcrBatchResult(frame_id=int(frame_id), ts=time.time(), slots=slots, error=batch_error)
            with self._lock:
                self._last_result = result

        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
