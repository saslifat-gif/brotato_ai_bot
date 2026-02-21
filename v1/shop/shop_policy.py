import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from runtime.input_driver import InputDriver
from shop.ocr_winmedia import BRANCH_WEAPON_TOKENS, OcrBatchResult, ShopOcrWorker, SlotOcrScore

BRANCH_TOKEN_SET = {str(tok).strip().lower() for tok in BRANCH_WEAPON_TOKENS}


@dataclass
class ShopDecision:
    action: str
    slot: int
    reason: str
    best_score: float
    debug: Dict[str, object]


class ShopPolicy:
    def __init__(
        self,
        input_driver: InputDriver,
        ocr_worker: ShopOcrWorker,
        shop_points: List[Tuple[int, int]],
        refresh_rect: Tuple[int, int, int, int],
        go_rect: Tuple[int, int, int, int],
        card_w: int,
        card_h: int,
        card_down: int,
        buy_min_score: float,
        refresh_max: int,
        max_buys: int,
        action_cooldown_sec: float,
        confirm_frames: int,
        confirm_hamming: int,
        buy_retry_max: int,
        ocr_max_age_sec: float,
        card_rects: Optional[List[Tuple[int, int, int, int]]] = None,
        lock_rects: Optional[List[Tuple[int, int, int, int]]] = None,
        buy_offset_y: int = -40,
        # --- Savage Protocol ---
        stick_priority: bool = True,
        stick_min_score: float = 0.25,
        max_weapons: int = 6,
        # Seconds to wait after a refresh click before trusting OCR again.
        # Gives the game time to visually update new items.
        refresh_settle_sec: float = 0.40,
    ):
        self.input_driver = input_driver
        self.ocr_worker = ocr_worker

        self.shop_points = list(shop_points)
        self.refresh_rect = tuple(refresh_rect)
        self.go_rect = tuple(go_rect)
        self.card_w = int(card_w)
        self.card_h = int(card_h)
        self.card_down = int(card_down)
        self.card_rects = [tuple(rc) for rc in list(card_rects or [])]

        self.buy_min_score = float(buy_min_score)
        self.refresh_max = int(refresh_max)
        self.max_buys = int(max_buys)
        self.action_cooldown_sec = float(action_cooldown_sec)
        self.confirm_frames = int(confirm_frames)
        self.confirm_hamming = int(confirm_hamming)
        self.buy_retry_max = int(buy_retry_max)
        self.ocr_max_age_sec = float(ocr_max_age_sec)
        self.lock_rects = list(lock_rects) if lock_rects else []
        self.buy_offset_y = int(buy_offset_y)

        # Savage Protocol
        self.stick_priority = bool(stick_priority)
        self.stick_min_score = float(max(0.0, stick_min_score))
        self.max_weapons = int(max(1, max_weapons))

        self.frame_idx = 0
        self.last_action_ts = 0.0
        # After a refresh click the game needs time to visually update.
        # We ignore OCR results for this many seconds after any refresh.
        self.refresh_settle_sec = float(max(0.0, refresh_settle_sec))
        self.last_refresh_ts = 0.0

        self.buy_count = 0
        self.refresh_count = 0
        self.total_decisions = 0
        self.total_buy = 0
        self.total_refresh = 0
        self.total_go = 0
        self.total_buy_fail = 0

        self.pending_buy: Optional[Dict[str, int]] = None
        self.buy_fail_streak = 0
        self.no_change_fail_count = 0
        self.last_fail_reason = ""
        self.blocked_slots_until: Dict[int, int] = {}

        self.bought_weapons: List[str] = []
        self.merge_bonus = 0.3

        self.last_slots: List[Dict[str, object]] = []
        self.last_best_idx = -1
        self.last_best_score = 0.0
        self.last_action = "idle"
        self.last_ocr_error = ""
        self.ocr_submit_count = 0
        self.ocr_hit_count = 0
        self.ocr_miss_count = 0
        self.last_ocr_age_sec = 0.0
        self.last_ocr_frame_id = -1
        self.ocr_miss_streak = 0
        # If OCR keeps missing in shop, use this many misses before
        # triggering a recovery action (refresh/go) instead of waiting forever.
        self.ocr_recover_miss_steps = 6

    def reset_cycle(self):
        self.buy_count = 0
        self.refresh_count = 0
        self.pending_buy = None
        self.buy_fail_streak = 0
        self.no_change_fail_count = 0
        self.last_fail_reason = ""
        self.blocked_slots_until = {}
        self.ocr_miss_streak = 0

    def reset_episode(self):
        """Call at the start of each new game to clear weapon-slot tracking."""
        self.bought_weapons = []
        self.reset_cycle()

    @staticmethod
    def _dhash64(img_bgr: np.ndarray) -> int:
        if img_bgr is None or img_bgr.size == 0:
            return 0
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
        diff = g[:, 1:] > g[:, :-1]
        out = 0
        bit = 0
        for y in range(8):
            for x in range(8):
                if diff[y, x]:
                    out |= (1 << bit)
                bit += 1
        return int(out)

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return int((int(a) ^ int(b)).bit_count())

    def _slot_card_patch(self, frame_bgr: np.ndarray, pt: Tuple[int, int], slot_idx: int) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if 0 <= int(slot_idx) < len(self.card_rects):
            x1, y1, x2, y2 = [int(v) for v in self.card_rects[int(slot_idx)]]
            x1 = int(np.clip(x1, 0, max(0, w - 1)))
            y1 = int(np.clip(y1, 0, max(0, h - 1)))
            x2 = int(np.clip(x2, x1 + 1, max(x1 + 1, w)))
            y2 = int(np.clip(y2, y1 + 1, max(y1 + 1, h)))
            return frame_bgr[y1:y2, x1:x2]
        x, y = int(pt[0]), int(pt[1])
        x1 = int(np.clip(x - self.card_w // 2, 0, max(0, w - 1)))
        x2 = int(np.clip(x + self.card_w // 2, x1 + 1, max(x1 + 1, w)))
        y1 = int(np.clip(y - self.card_h, 0, max(0, h - 1)))
        y2 = int(np.clip(y + self.card_down, y1 + 1, max(y1 + 1, h)))
        return frame_bgr[y1:y2, x1:x2]

    def _build_cards(self, frame_bgr: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        cards: Dict[int, np.ndarray] = {}
        hashes: Dict[int, int] = {}
        for i, pt in enumerate(self.shop_points):
            slot = i + 1
            card = self._slot_card_patch(frame_bgr, pt, i)
            cards[slot] = card
            hashes[slot] = self._dhash64(card)
        return cards, hashes

    def _slots_from_ocr(self, ocr_res: Optional[OcrBatchResult]) -> List[Dict[str, object]]:
        slot_map: Dict[int, SlotOcrScore] = {}
        if ocr_res is not None:
            for s in ocr_res.slots:
                slot_map[int(s.slot)] = s

        infos: List[Dict[str, object]] = []
        for i in range(len(self.shop_points)):
            slot = i + 1
            s = slot_map.get(slot)
            if s is None:
                infos.append(
                    {
                        "slot": slot,
                        "score": 0.0,
                        "primitive_class": 0.0,
                        "primitive_weapon": 0.0,
                        "non_primitive": 0.0,
                        "melee_buff": 0.0,
                        "attack_buff": 0.0,
                        "attack_speed_buff": 0.0,
                        "weapon_name": "",
                        "merge_bonus": 0.0,
                        "lines": [],
                    }
                )
            else:
                # Apply merge bonus if this weapon was already bought
                w_name = str(s.weapon_name or "")
                m_bonus = self.merge_bonus if (w_name and w_name in self.bought_weapons) else 0.0
                effective_score = float(s.primary) + m_bonus
                infos.append(
                    {
                        "slot": slot,
                        "score": effective_score,
                        "primitive_class": float(s.primitive_class),
                        "primitive_weapon": float(s.primitive_weapon),
                        "non_primitive": float(s.non_primitive),
                        "melee_buff": float(s.melee_buff),
                        "attack_buff": float(s.attack_buff),
                        "attack_speed_buff": float(s.attack_speed_buff),
                        "weapon_name": w_name,
                        "merge_bonus": m_bonus,
                        "lines": list(s.lines),
                    }
                )
        return infos

    @staticmethod
    def _is_non_primitive_only(info: Dict[str, object]) -> bool:
        return (
            float(info.get("non_primitive", 0.0)) > 0.0
            and float(info.get("primitive_weapon", 0.0)) <= 0.0
            and float(info.get("primitive_class", 0.0)) <= 0.0
        )

    @staticmethod
    def _is_any_weapon(info: Dict[str, object]) -> bool:
        return (
            float(info.get("primitive_weapon", 0.0)) > 0.0
            or float(info.get("non_primitive", 0.0)) > 0.0
            or bool(str(info.get("weapon_name", "") or "").strip())
        )

    @staticmethod
    def _is_branch_weapon(info: Dict[str, object]) -> bool:
        w = str(info.get("weapon_name", "") or "").strip().lower()
        if not w:
            return False
        return w in BRANCH_TOKEN_SET

    @staticmethod
    def _is_attack_stat_item(info: Dict[str, object]) -> bool:
        return (
            float(info.get("attack_buff", 0.0)) > 0.0
            or float(info.get("attack_speed_buff", 0.0)) > 0.0
        )

    def _buy_whitelist_kind(self, info: Dict[str, object]) -> str:
        if self._is_non_primitive_only(info):
            return ""
        if self._is_branch_weapon(info):
            return "branch"
        if self._is_attack_stat_item(info) and (not self._is_any_weapon(info)):
            return "atk_item"
        return ""

    def _is_buy_whitelist_item(self, info: Dict[str, object]) -> bool:
        return bool(self._buy_whitelist_kind(info))

    def _point_in_lock_rect(self, x: int, y: int) -> bool:
        """Check if (x, y) falls inside any lock exclusion rect."""
        for lx1, ly1, lx2, ly2 in self.lock_rects:
            if lx1 <= x <= lx2 and ly1 <= y <= ly2:
                return True
        return False

    def _safe_buy_point(self, slot_idx: int) -> Tuple[int, int]:
        """Return a click point for buying that avoids lock rects.

        Applies buy_offset_y to shift the click upward (into the item
        image area) and verifies it does not overlap any lock rect.
        """
        px, py = self.shop_points[slot_idx]
        # Apply offset (negative = upward, away from lock buttons)
        candidate_y = py + self.buy_offset_y
        if not self._point_in_lock_rect(px, candidate_y):
            return (px, candidate_y)
        # If offset point still in lock rect, try the original point
        if not self._point_in_lock_rect(px, py):
            return (px, py)
        # Last resort: shift further up
        return (px, max(0, py - abs(self.buy_offset_y) * 2))

    def _cooldown_ready(self) -> bool:
        return (time.time() - float(self.last_action_ts)) >= self.action_cooldown_sec

    def _start_pending_buy(self, slot: int, before_hash: int, weapon_name: str = ""):
        self.pending_buy = {
            "slot": int(slot),
            "before": int(before_hash),
            "deadline_frame": int(self.frame_idx + self.confirm_frames),
            "retries": 0,
            "weapon_name": str(weapon_name or ""),
        }

    def _handle_pending_buy(self, current_hashes: Dict[int, int]) -> Optional[ShopDecision]:
        if self.pending_buy is None:
            return None

        slot = int(self.pending_buy.get("slot", -1))
        before_hash = int(self.pending_buy.get("before", 0))
        deadline = int(self.pending_buy.get("deadline_frame", self.frame_idx))
        retries = int(self.pending_buy.get("retries", 0))

        now_hash = int(current_hashes.get(slot, before_hash))
        dist = self._hamming(before_hash, now_hash)
        if dist >= self.confirm_hamming:
            self.buy_count += 1
            self.total_buy += 1
            self.buy_fail_streak = 0
            self.no_change_fail_count = 0
            # Track bought weapon for merge detection
            bought_name = str(self.pending_buy.get("weapon_name", "") or "")
            if bought_name:
                self.bought_weapons.append(bought_name)
            self.pending_buy = None
            self.last_action = f"buy_{slot}_confirmed"
            return ShopDecision(
                action="buy_confirmed",
                slot=slot,
                reason=f"hash_changed({dist})",
                best_score=self.last_best_score,
                debug={"hamming": dist, "buy_count": self.buy_count},
            )

        if self.frame_idx <= deadline:
            self.last_action = f"buy_{slot}_waiting_confirm"
            return ShopDecision(
                action="wait_confirm",
                slot=slot,
                reason=f"hamming={dist}",
                best_score=self.last_best_score,
                debug={"hamming": dist, "deadline": deadline},
            )

        if retries < self.buy_retry_max and self._cooldown_ready():
            buy_pt = self._safe_buy_point(slot - 1)
            click_res = self.input_driver.click_client_point(buy_pt)
            self.last_action_ts = time.time()
            self.pending_buy = {
                "slot": slot,
                "before": now_hash,
                "deadline_frame": int(self.frame_idx + self.confirm_frames),
                "retries": retries + 1,
            }
            self.last_action = f"buy_{slot}_retry"
            return ShopDecision(
                action="buy_retry",
                slot=slot,
                reason=f"retry={retries + 1} click={click_res.method}:{click_res.ok}",
                best_score=self.last_best_score,
                debug={"hamming": dist, "click_error": click_res.error},
            )

        self.total_buy_fail += 1
        self.buy_fail_streak += 1
        if dist <= 0:
            self.no_change_fail_count += 1
        self.blocked_slots_until[int(slot)] = int(self.frame_idx + max(8, self.confirm_frames * 2))
        self.last_fail_reason = (
            f"slot={slot} hamming={dist} retries={retries} "
            f"no_change_fail={self.no_change_fail_count}"
        )
        self.pending_buy = None
        self.last_action = "buy_failed"
        return ShopDecision(
            action="buy_failed",
            slot=slot,
            reason=self.last_fail_reason,
            best_score=self.last_best_score,
            debug={"hamming": dist, "buy_fail_streak": self.buy_fail_streak},
        )

    def evaluate(self, frame_rgb: np.ndarray, in_shop: bool, state_score: float, allow_action: bool = True) -> ShopDecision:
        self.frame_idx += 1
        self.total_decisions += 1

        if frame_rgb is None or frame_rgb.size == 0:
            return ShopDecision("wait", -1, "empty_frame", 0.0, {})

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cards, card_hashes = self._build_cards(frame_bgr)
        self.ocr_worker.submit(self.frame_idx, cards)
        self.ocr_submit_count += 1
        ocr_res = self.ocr_worker.get_latest(self.ocr_max_age_sec)

        # After a refresh click the game needs time to visually update its
        # item cards.  ocr_res.ts records when OCR *completed*, not when the
        # frame was captured — so comparing ts vs last_action_ts is wrong
        # (OCR may finish 0.1 s after the click but have processed a
        # pre-refresh frame).  Instead we simply discard all OCR during the
        # settle window after any refresh click.
        if ocr_res is not None and self.refresh_settle_sec > 0.0:
            if (time.time() - self.last_refresh_ts) < self.refresh_settle_sec:
                ocr_res = None

        worker_last_error = ""
        try:
            worker_last_error = str(getattr(self.ocr_worker, "last_error", lambda: "")() or "")
        except Exception:
            worker_last_error = ""
        if ocr_res is None:
            self.ocr_miss_count += 1
            self.ocr_miss_streak += 1
            self.last_ocr_age_sec = float(self.ocr_max_age_sec)
            self.last_ocr_frame_id = -1
            self.last_ocr_error = worker_last_error
        else:
            self.ocr_hit_count += 1
            self.ocr_miss_streak = 0
            self.last_ocr_age_sec = max(0.0, time.time() - float(ocr_res.ts))
            self.last_ocr_frame_id = int(ocr_res.frame_id)
            self.last_ocr_error = str(ocr_res.error or worker_last_error or "")
        self.last_slots = self._slots_from_ocr(ocr_res)

        if not in_shop:
            self.reset_cycle()
            self.last_action = "not_shop"
            return ShopDecision("wait", -1, "not_in_shop", 0.0, {})

        pending_decision = self._handle_pending_buy(card_hashes)
        if pending_decision is not None:
            return pending_decision

        # Buy whitelist:
        #   1) Branch weapon only (tree branch)
        #   2) Non-weapon items with attack / attack-speed stat lines
        # All other items are ignored.
        best_idx = -1
        best_score = -1.0
        best_kind = ""
        for i, info in enumerate(self.last_slots):
            slot = int(info.get("slot", i + 1))
            if int(self.blocked_slots_until.get(slot, -1)) > int(self.frame_idx):
                continue
            kind = self._buy_whitelist_kind(info)
            if not kind:
                continue
            score = float(info.get("score", 0.0))
            if score > best_score:
                best_score = score
                best_idx = i
                best_kind = kind

        # Savage Protocol priority within one shop cycle:
        #   1) If stick_priority and weapon slots are open: buy stick first
        #   2) Otherwise buy best whitelist item
        #   3) If nothing in whitelist: refresh / go
        effective_buy_threshold = self.buy_min_score
        force_refresh = False

        if self.stick_priority:
            weapon_count = len(self.bought_weapons)

            # Find best branch slot.
            stick_best_idx = -1
            stick_best_pw = -1.0
            for i, info in enumerate(self.last_slots):
                slot = int(info.get("slot", i + 1))
                if int(self.blocked_slots_until.get(slot, -1)) > int(self.frame_idx):
                    continue
                if self._is_non_primitive_only(info):
                    continue
                if not self._is_branch_weapon(info):
                    continue
                pw = float(info.get("primitive_weapon", 0.0))
                if pw >= self.stick_min_score and pw > stick_best_pw:
                    stick_best_pw = pw
                    stick_best_idx = i

            if stick_best_idx >= 0 and weapon_count < self.max_weapons:
                # CASE 1: stick available + slot open -> buy stick.
                best_idx = stick_best_idx
                best_score = max(float(self.last_slots[stick_best_idx].get("score", 0.0)), stick_best_pw)
                effective_buy_threshold = self.stick_min_score
                best_kind = "branch"
            elif weapon_count < self.max_weapons and best_idx < 0:
                # CASE 2: slots open but no whitelisted item -> keep rolling.
                force_refresh = True
        # ─────────────────────────────────────────────────────────────────────

        self.last_best_idx = int(best_idx)
        self.last_best_score = float(best_score)

        if not bool(allow_action):
            self.last_action = "wait_lock"
            return ShopDecision("wait", -1, "shop_entry_lock", best_score, {"state_score": float(state_score)})

        if not self._cooldown_ready():
            self.last_action = "wait_cooldown"
            return ShopDecision("wait", -1, "cooldown", best_score, {})

        # If OCR is stale/unavailable, do not random-click. Wait for reliable signal.
        if ocr_res is None:
            # Recovery: if OCR keeps missing for multiple frames in shop,
            # do one controlled refresh (or leave shop) to avoid deadlock.
            if self.ocr_miss_streak >= int(self.ocr_recover_miss_steps):
                if self.refresh_count < self.refresh_max:
                    click_res = self.input_driver.click_client_rect(self.refresh_rect)
                    self.last_action_ts = time.time()
                    self.last_refresh_ts = self.last_action_ts
                    self.refresh_count += 1
                    self.total_refresh += 1
                    self.blocked_slots_until = {}
                    self.last_action = "refresh_no_ocr"
                    self.ocr_miss_streak = 0
                    return ShopDecision(
                        "refresh",
                        -1,
                        f"ocr_miss_streak_recover click={click_res.method}:{click_res.ok}",
                        best_score,
                        {"click_error": click_res.error},
                    )
                click_res = self.input_driver.click_client_rect(self.go_rect)
                self.last_action_ts = time.time()
                self.total_go += 1
                self.last_action = "go_no_ocr"
                self.reset_cycle()
                return ShopDecision(
                    "go",
                    -1,
                    f"ocr_miss_streak_recover max_refresh={self.refresh_max} click={click_res.method}:{click_res.ok}",
                    best_score,
                    {"click_error": click_res.error},
                )
            self.last_action = "wait_ocr"
            return ShopDecision("wait", -1, "no_fresh_ocr", best_score, {"state_score": float(state_score)})

        if self.buy_fail_streak >= 2:
            if self.no_change_fail_count >= 3:
                click_res = self.input_driver.click_client_rect(self.go_rect)
                self.last_action_ts = time.time()
                self.total_go += 1
                self.last_action = "go_after_no_change_fail"
                self.reset_cycle()
                return ShopDecision(
                    "go",
                    -1,
                    f"no_change_fail={self.no_change_fail_count} click={click_res.method}:{click_res.ok}",
                    best_score,
                    {"click_error": click_res.error},
                )
            if self.refresh_count < self.refresh_max:
                fail_streak = int(self.buy_fail_streak)
                click_res = self.input_driver.click_client_rect(self.refresh_rect)
                self.last_action_ts = time.time()
                self.last_refresh_ts = self.last_action_ts
                self.refresh_count += 1
                self.total_refresh += 1
                self.blocked_slots_until = {}
                # Prevent refresh loops: after a fail-driven refresh, let policy
                # re-evaluate fresh OCR instead of refreshing again immediately.
                self.buy_fail_streak = 0
                self.last_action = "refresh_after_fail"
                return ShopDecision(
                    "refresh",
                    -1,
                    f"buy_fail_streak={fail_streak} click={click_res.method}:{click_res.ok}",
                    best_score,
                    {"click_error": click_res.error},
                )
            click_res = self.input_driver.click_client_rect(self.go_rect)
            self.last_action_ts = time.time()
            self.total_go += 1
            self.last_action = "go_after_fail"
            self.reset_cycle()
            return ShopDecision(
                "go",
                -1,
                f"buy_fail_streak={self.buy_fail_streak} click={click_res.method}:{click_res.ok}",
                best_score,
                {"click_error": click_res.error},
            )

        # Savage Protocol: if force_refresh is set, skip buy entirely and go
        # straight to refresh (or go) to keep hunting for whitelist items.
        if force_refresh:
            if self.refresh_count < self.refresh_max:
                click_res = self.input_driver.click_client_rect(self.refresh_rect)
                self.last_action_ts = time.time()
                self.last_refresh_ts = self.last_action_ts
                self.refresh_count += 1
                self.total_refresh += 1
                self.blocked_slots_until = {}
                self.last_action = "refresh_savage"
                return ShopDecision(
                    "refresh",
                    -1,
                    f"savage:no_whitelist slots={len(self.bought_weapons)}/{self.max_weapons} click={click_res.method}:{click_res.ok}",
                    best_score,
                    {"click_error": click_res.error},
                )
            # Max refreshes used and still no whitelist item -> proceed to shop end.
            click_res = self.input_driver.click_client_rect(self.go_rect)
            self.last_action_ts = time.time()
            self.total_go += 1
            self.last_action = "go_savage"
            self.reset_cycle()
            return ShopDecision(
                "go",
                -1,
                f"savage:no_whitelist max_refresh={self.refresh_max} click={click_res.method}:{click_res.ok}",
                best_score,
                {"click_error": click_res.error},
            )

        can_buy = best_idx >= 0 and best_score >= effective_buy_threshold and self.buy_count < self.max_buys
        if can_buy:
            slot = best_idx + 1
            buy_pt = self._safe_buy_point(best_idx)
            click_res = self.input_driver.click_client_point(buy_pt)
            self.last_action_ts = time.time()
            best_weapon_name = str(self.last_slots[best_idx].get("weapon_name", "") if best_idx < len(self.last_slots) else "")
            self._start_pending_buy(slot=slot, before_hash=card_hashes.get(slot, 0), weapon_name=best_weapon_name)
            self.last_action = f"buy_{slot}_click"
            return ShopDecision(
                action="buy_click",
                slot=slot,
                reason=f"kind={best_kind} score={best_score:.3f} click={click_res.method}:{click_res.ok}",
                best_score=best_score,
                debug={"click_error": click_res.error},
            )

        if self.refresh_count < self.refresh_max:
            click_res = self.input_driver.click_client_rect(self.refresh_rect)
            self.last_action_ts = time.time()
            self.last_refresh_ts = self.last_action_ts
            self.refresh_count += 1
            self.total_refresh += 1
            self.blocked_slots_until = {}
            self.last_action = "refresh"
            return ShopDecision(
                action="refresh",
                slot=-1,
                reason=f"best={best_score:.3f} buys={self.buy_count}",
                best_score=best_score,
                debug={"click_error": click_res.error},
            )

        click_res = self.input_driver.click_client_rect(self.go_rect)
        self.last_action_ts = time.time()
        self.total_go += 1
        self.last_action = "go"
        self.reset_cycle()
        return ShopDecision(
            action="go",
            slot=-1,
            reason="cycle_end",
            best_score=best_score,
            debug={"click_error": click_res.error},
        )

    def debug_snapshot(self) -> Dict[str, object]:
        return {
            "slots": list(self.last_slots),
            "best_idx": int(self.last_best_idx),
            "best_score": float(self.last_best_score),
            "action": str(self.last_action),
            "buy_count": int(self.buy_count),
            "refresh_count": int(self.refresh_count),
            "total_decisions": int(self.total_decisions),
            "total_buy": int(self.total_buy),
            "total_refresh": int(self.total_refresh),
            "total_go": int(self.total_go),
            "total_buy_fail": int(self.total_buy_fail),
            "buy_fail_streak": int(self.buy_fail_streak),
            "no_change_fail_count": int(self.no_change_fail_count),
            "last_fail_reason": str(self.last_fail_reason),
            "bought_weapons": list(self.bought_weapons),
            "ocr_error": str(self.last_ocr_error),
            "ocr_worker_running": bool(getattr(self.ocr_worker, "is_running", lambda: True)()),
            "ocr_submit_count": int(self.ocr_submit_count),
            "ocr_hit_count": int(self.ocr_hit_count),
            "ocr_miss_count": int(self.ocr_miss_count),
            "ocr_miss_streak": int(self.ocr_miss_streak),
            "ocr_last_age_sec": float(self.last_ocr_age_sec),
            "ocr_last_frame_id": int(self.last_ocr_frame_id),
            "pending_buy": dict(self.pending_buy) if self.pending_buy is not None else None,
            "blocked_slots_until": dict(self.blocked_slots_until),
        }
