import time
import unittest
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "v1"))

from shop.shop_policy import ShopPolicy
from shop.ocr_winmedia import OcrBatchResult, SlotOcrScore


class FakeInput:
    def __init__(self):
        self.clicked_points = []
        self.clicked_rects = []

    def click_client_point(self, pos):
        self.clicked_points.append(tuple(pos))
        return type("R", (), {"ok": True, "method": "post_message", "error": ""})()

    def click_client_rect(self, rect):
        self.clicked_rects.append(tuple(rect))
        return type("R", (), {"ok": True, "method": "post_message", "error": ""})()


class FakeOcrWorker:
    def __init__(self):
        self.last_submit = None

    def submit(self, frame_id, cards):
        self.last_submit = (frame_id, cards)

    def get_latest(self, _max_age):
        return OcrBatchResult(
            frame_id=1,
            ts=time.time(),
            slots=[
                SlotOcrScore(slot=1, primitive_class=1.0, primitive_weapon=1.0, non_primitive=0.0, melee_buff=0.0, primary=1.0, weapon_name="树枝", lines=["原始 树枝"]),
                SlotOcrScore(slot=2, primitive_class=0.0, primitive_weapon=0.0, non_primitive=0.0, melee_buff=0.0, primary=0.0, weapon_name="", lines=[]),
                SlotOcrScore(slot=3, primitive_class=0.0, primitive_weapon=0.0, non_primitive=0.0, melee_buff=0.0, primary=0.0, weapon_name="", lines=[]),
                SlotOcrScore(slot=4, primitive_class=0.0, primitive_weapon=0.0, non_primitive=0.0, melee_buff=0.0, primary=0.0, weapon_name="", lines=[]),
            ],
            error="",
        )


class ShopPolicyTests(unittest.TestCase):
    def test_safe_buy_point_avoids_lock_rect(self):
        policy = ShopPolicy(
            input_driver=FakeInput(),
            ocr_worker=FakeOcrWorker(),
            shop_points=[(100, 200), (300, 200)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=2,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
            lock_rects=[(80, 230, 120, 260)],
            buy_offset_y=-40,
        )
        # Offset point (100, 160) is NOT in lock rect → should return offset point
        pt = policy._safe_buy_point(0)
        self.assertEqual(pt, (100, 160))
        # Point NOT in lock rect at all
        self.assertFalse(policy._point_in_lock_rect(100, 160))
        # Point IS in lock rect
        self.assertTrue(policy._point_in_lock_rect(100, 240))

    def test_refresh_after_buying(self):
        """After buying items, refresh should still work if refresh_count < refresh_max."""
        inp = FakeInput()
        policy = ShopPolicy(
            input_driver=inp,
            ocr_worker=FakeOcrWorker(),
            shop_points=[(100, 100), (200, 100), (300, 100), (400, 100)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=3,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
        )
        # Simulate that a buy already happened
        policy.buy_count = 1
        # OCR returns nothing good (all scores 0)
        policy.ocr_worker = type("W", (), {
            "submit": lambda s, f, c: None,
            "get_latest": lambda s, a: OcrBatchResult(
                frame_id=1, ts=time.time(),
                slots=[
                    SlotOcrScore(slot=1, primitive_class=0.0, primitive_weapon=0.0, non_primitive=0.0, melee_buff=0.0, primary=0.0, weapon_name="", lines=[]),
                    SlotOcrScore(slot=2, primitive_class=0.0, primitive_weapon=0.0, non_primitive=0.0, melee_buff=0.0, primary=0.0, weapon_name="", lines=[]),
                    SlotOcrScore(slot=3, primitive_class=0.0, primitive_weapon=0.0, non_primitive=0.0, melee_buff=0.0, primary=0.0, weapon_name="", lines=[]),
                    SlotOcrScore(slot=4, primitive_class=0.0, primitive_weapon=0.0, non_primitive=0.0, melee_buff=0.0, primary=0.0, weapon_name="", lines=[]),
                ],
                error="",
            ),
        })()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dec = policy.evaluate(frame_rgb=frame, in_shop=True, state_score=1.0)
        # Should refresh even though buy_count > 0
        self.assertEqual(dec.action, "refresh")

    def test_refresh_after_fail_resets_fail_streak(self):
        inp = FakeInput()
        policy = ShopPolicy(
            input_driver=inp,
            ocr_worker=FakeOcrWorker(),
            shop_points=[(100, 100), (200, 100), (300, 100), (400, 100)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=3,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
        )
        policy.buy_fail_streak = 2
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dec = policy.evaluate(frame_rgb=frame, in_shop=True, state_score=1.0)
        self.assertEqual(dec.action, "refresh")
        self.assertEqual(policy.buy_fail_streak, 0)

    def test_no_change_fail_streak_goes_shop_end(self):
        inp = FakeInput()
        policy = ShopPolicy(
            input_driver=inp,
            ocr_worker=FakeOcrWorker(),
            shop_points=[(100, 100), (200, 100), (300, 100), (400, 100)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=3,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
        )
        policy.buy_fail_streak = 2
        policy.no_change_fail_count = 3
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dec = policy.evaluate(frame_rgb=frame, in_shop=True, state_score=1.0)
        self.assertEqual(dec.action, "go")
        self.assertIn("no_change_fail", dec.reason)

    def test_buy_path_starts_when_primitive_detected(self):
        policy = ShopPolicy(
            input_driver=FakeInput(),
            ocr_worker=FakeOcrWorker(),
            shop_points=[(100, 100), (200, 100), (300, 100), (400, 100)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=2,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
        )

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dec = policy.evaluate(frame_rgb=frame, in_shop=True, state_score=1.0)
        self.assertIn(dec.action, ("buy_click", "wait_confirm", "buy_confirmed"))

    def test_attack_stat_item_is_whitelisted_for_buy(self):
        policy = ShopPolicy(
            input_driver=FakeInput(),
            ocr_worker=type("W", (), {
                "submit": lambda s, f, c: None,
                "get_latest": lambda s, a: OcrBatchResult(
                    frame_id=1,
                    ts=time.time(),
                    slots=[
                        SlotOcrScore(
                            slot=1,
                            primitive_class=0.0,
                            primitive_weapon=0.0,
                            non_primitive=0.0,
                            melee_buff=0.0,
                            primary=0.95,
                            weapon_name="",
                            lines=["+15% 攻击速度"],
                            attack_buff=0.0,
                            attack_speed_buff=1.0,
                        ),
                    ],
                    error="",
                ),
            })(),
            shop_points=[(100, 100), (200, 100), (300, 100), (400, 100)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=2,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dec = policy.evaluate(frame_rgb=frame, in_shop=True, state_score=1.0)
        self.assertEqual(dec.action, "buy_click")

    def test_non_whitelist_item_is_skipped(self):
        policy = ShopPolicy(
            input_driver=FakeInput(),
            ocr_worker=type("W", (), {
                "submit": lambda s, f, c: None,
                "get_latest": lambda s, a: OcrBatchResult(
                    frame_id=1,
                    ts=time.time(),
                    slots=[
                        SlotOcrScore(
                            slot=1,
                            primitive_class=0.0,
                            primitive_weapon=0.0,
                            non_primitive=0.0,
                            melee_buff=1.0,
                            primary=0.9,
                            weapon_name="",
                            lines=["+10 近战伤害"],
                            attack_buff=0.0,
                            attack_speed_buff=0.0,
                        ),
                    ],
                    error="",
                ),
            })(),
            shop_points=[(100, 100), (200, 100), (300, 100), (400, 100)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=2,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dec = policy.evaluate(frame_rgb=frame, in_shop=True, state_score=1.0)
        self.assertEqual(dec.action, "refresh")
        self.assertIn("no_whitelist", dec.reason)

    def test_non_branch_weapon_with_attack_text_is_skipped(self):
        policy = ShopPolicy(
            input_driver=FakeInput(),
            ocr_worker=type("W", (), {
                "submit": lambda s, f, c: None,
                "get_latest": lambda s, a: OcrBatchResult(
                    frame_id=1,
                    ts=time.time(),
                    slots=[
                        SlotOcrScore(
                            slot=1,
                            primitive_class=1.0,
                            primitive_weapon=1.0,
                            non_primitive=0.0,
                            melee_buff=0.0,
                            primary=0.95,
                            weapon_name="长矛",
                            lines=["原始 长矛 +攻击"],
                            attack_buff=1.0,
                            attack_speed_buff=0.0,
                        ),
                    ],
                    error="",
                ),
            })(),
            shop_points=[(100, 100), (200, 100), (300, 100), (400, 100)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=2,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dec = policy.evaluate(frame_rgb=frame, in_shop=True, state_score=1.0)
        self.assertEqual(dec.action, "refresh")
        self.assertIn("no_whitelist", dec.reason)

    def test_persistent_ocr_miss_triggers_refresh_recovery(self):
        class EmptyOcrWorker:
            def submit(self, _frame_id, _cards):
                return None

            def get_latest(self, _max_age):
                return None

        inp = FakeInput()
        policy = ShopPolicy(
            input_driver=inp,
            ocr_worker=EmptyOcrWorker(),
            shop_points=[(100, 100), (200, 100), (300, 100), (400, 100)],
            refresh_rect=(10, 10, 20, 20),
            go_rect=(30, 30, 40, 40),
            card_w=100,
            card_h=120,
            card_down=30,
            buy_min_score=0.4,
            refresh_max=2,
            max_buys=4,
            action_cooldown_sec=0.0,
            confirm_frames=2,
            confirm_hamming=1,
            buy_retry_max=0,
            ocr_max_age_sec=1.0,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dec = None
        for _ in range(6):
            dec = policy.evaluate(frame_rgb=frame, in_shop=True, state_score=1.0)
        self.assertIsNotNone(dec)
        self.assertEqual(dec.action, "refresh")
        self.assertIn("ocr_miss_streak_recover", dec.reason)
        self.assertEqual(inp.clicked_rects[-1], (10, 10, 20, 20))


if __name__ == "__main__":
    unittest.main()
