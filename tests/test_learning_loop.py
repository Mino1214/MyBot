from __future__ import annotations

import unittest
from unittest.mock import patch

from rebalancing.learning.alerts import learning_result_message
from rebalancing.learning.loop import maybe_promote_stage, run_learning_cycle


class LearningLoopTest(unittest.TestCase):
    def test_maybe_promote_requires_claude_and_metrics(self) -> None:
        metrics = {
            "closed_trade_result_count": 20,
            "win_rate": 0.56,
            "realized_pnl_total": 10,
            "max_drawdown_pnl": -20,
        }

        self.assertEqual(maybe_promote_stage("BABY", metrics, {"ready_for_promotion": False}), ("BABY", False))

        with patch("rebalancing.learning.loop.update_stage") as update_stage:
            self.assertEqual(maybe_promote_stage("BABY", metrics, {"ready_for_promotion": True}), ("JUNIOR", True))

        update_stage.assert_called_once_with("JUNIOR")

    def test_maybe_promote_blocks_weak_metrics(self) -> None:
        metrics = {
            "closed_trade_result_count": 3,
            "win_rate": 1.0,
            "realized_pnl_total": 10,
            "max_drawdown_pnl": 0,
        }

        self.assertEqual(maybe_promote_stage("BABY", metrics, {"ready_for_promotion": True}), ("BABY", False))

    def test_run_learning_cycle_records_failed_diagnosis(self) -> None:
        with (
            patch("rebalancing.learning.loop.current_stage", return_value="BABY"),
            patch("rebalancing.learning.loop.load_recent_records", return_value=()),
            patch("rebalancing.learning.loop.review_active_param_effects", return_value={"rollback_recommended": False}),
            patch("rebalancing.learning.loop.run_diagnosis", return_value=None),
            patch("rebalancing.learning.loop._record_learning_run") as record_run,
            patch("rebalancing.learning.loop.notify_learning_result") as notify,
        ):
            result = run_learning_cycle(window=10, mode="paper")

        self.assertEqual(result["status"], "diagnosis_failed")
        record_run.assert_called_once()
        notify.assert_called_once()

    def test_run_learning_cycle_applies_and_promotes(self) -> None:
        diagnosis = {"evaluation_id": 5, "stage_eval": {"ready_for_promotion": True}}
        metrics = {
            "closed_trade_result_count": 20,
            "win_rate": 0.6,
            "realized_pnl_total": 1,
            "max_drawdown_pnl": 0,
        }
        with (
            patch("rebalancing.learning.loop.current_stage", return_value="BABY"),
            patch("rebalancing.learning.loop.load_recent_records", return_value=("record",)),
            patch("rebalancing.learning.loop.summarize_records", return_value=metrics),
            patch("rebalancing.learning.loop.review_active_param_effects", return_value={"rollback_recommended": False}),
            patch("rebalancing.learning.loop.run_diagnosis", return_value=diagnosis),
            patch("rebalancing.learning.loop.apply_evaluation_suggestions", return_value={"version": 1, "active": False}),
            patch("rebalancing.learning.loop.update_stage"),
            patch("rebalancing.learning.loop._record_learning_run"),
            patch("rebalancing.learning.loop.notify_learning_result"),
        ):
            result = run_learning_cycle(window=10, mode="paper", apply_policy="approve")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["evaluation_id"], 5)
        self.assertEqual(result["stage_after"], "JUNIOR")
        self.assertTrue(result["promoted"])

    def test_run_learning_cycle_rolls_back_before_new_diagnosis(self) -> None:
        review = {
            "rollback_recommended": True,
            "rollback_target_version": 2,
            "active_version": 3,
            "reasons": ["marked PnL worsened"],
        }
        with (
            patch("rebalancing.learning.loop.current_stage", return_value="BABY"),
            patch("rebalancing.learning.loop.load_recent_records", return_value=("record",)),
            patch("rebalancing.learning.loop.summarize_records", return_value={"closed_trade_result_count": 30}),
            patch("rebalancing.learning.loop.review_active_param_effects", return_value=review),
            patch("rebalancing.learning.loop.activate_bot_params_version", return_value={"version": 2, "active": True}) as activate,
            patch("rebalancing.learning.loop.run_diagnosis") as diagnosis,
            patch("rebalancing.learning.loop._record_learning_run"),
            patch("rebalancing.learning.loop.notify_learning_result"),
        ):
            result = run_learning_cycle(window=10, mode="paper")

        self.assertEqual(result["status"], "rolled_back")
        activate.assert_called_once_with(2)
        diagnosis.assert_not_called()

    def test_learning_result_message_summarizes_changes(self) -> None:
        message = learning_result_message(
            {
                "status": "ok",
                "evaluation_id": 1,
                "stage_before": "BABY",
                "stage_after": "JUNIOR",
                "promoted": True,
                "apply_result": {
                    "version": 2,
                    "active": False,
                    "accepted": [{"name": "drift_threshold"}],
                },
            }
        )

        self.assertIn("status: ok", message)
        self.assertIn("changed: drift_threshold", message)
        self.assertIn("BABY -> JUNIOR", message)


if __name__ == "__main__":
    unittest.main()
