from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from rebalancing.learning.params import (
    activate_bot_params_version,
    apply_evaluation_suggestions,
    engine_config_from_params,
    prepare_param_update,
    review_active_param_effects,
)


class ParamTuningTest(unittest.TestCase):
    def test_prepare_param_update_clamps_and_ignores_unknowns(self) -> None:
        update = prepare_param_update(
            {},
            [
                {"name": "bull_target_leverage", "suggested": 9, "reason": "too high"},
                {"name": "confirmation_candles", "suggested": 2.2},
                {"name": "max_leverage", "suggested": 10},
                {"name": "adx_threshold", "suggested": "bad"},
            ],
        )

        self.assertEqual(update["params"]["bull_target_leverage"], 2.0)
        self.assertEqual(update["params"]["confirmation_candles"], 2)
        self.assertTrue(update["accepted"][0]["clamped"])
        self.assertEqual([item["name"] for item in update["ignored"]], ["max_leverage", "adx_threshold"])

    def test_engine_config_from_params_clamps_active_values(self) -> None:
        config = engine_config_from_params(
            {
                "bull_target_leverage": 4.0,
                "drift_threshold": 0.1,
                "unknown": 123,
            }
        )

        self.assertEqual(config.bull_target_leverage, 2.0)
        self.assertEqual(config.drift_threshold, 0.1)
        self.assertFalse(hasattr(config, "unknown"))

    def test_apply_evaluation_suggestions_stages_in_approve_mode(self) -> None:
        connection = _FakeConnection(
            fetches=[
                ([{"name": "bull_target_leverage", "suggested": 1.25, "reason": "paper drawdown"}],),
                None,
                (3,),
                (9,),
            ]
        )
        with patch("rebalancing.learning.params._with_connection", side_effect=lambda write: write(connection)):
            result = apply_evaluation_suggestions(42, policy="approve")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["version"], 3)
        self.assertFalse(result["active"])
        self.assertEqual(result["accepted"][0]["applied"], 1.25)
        self.assertTrue(any("INSERT INTO bot_params" in statement for statement in connection.cursor_obj.statements))
        self.assertFalse(any("UPDATE bot_params SET active = false" in statement for statement in connection.cursor_obj.statements))
        self.assertTrue(any("UPDATE evaluations" in statement for statement in connection.cursor_obj.statements))

    def test_apply_evaluation_suggestions_activates_in_auto_mode(self) -> None:
        connection = _FakeConnection(
            fetches=[
                ([{"name": "drift_threshold", "suggested": 0.15}],),
                ({"drift_threshold": 0.25},),
                (2,),
                (8,),
            ]
        )
        with patch("rebalancing.learning.params._with_connection", side_effect=lambda write: write(connection)):
            result = apply_evaluation_suggestions(7, policy="auto")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result["active"])
        self.assertTrue(any("UPDATE bot_params SET active = false" in statement for statement in connection.cursor_obj.statements))

    def test_activate_bot_params_version_switches_active_version(self) -> None:
        connection = _FakeConnection(fetches=[(11, 4)])
        with patch("rebalancing.learning.params._with_connection", side_effect=lambda write: write(connection)):
            result = activate_bot_params_version(4)

        self.assertEqual(result, {"bot_param_id": 11, "version": 4, "active": True})
        self.assertTrue(any("UPDATE bot_params SET active = false" in statement for statement in connection.cursor_obj.statements))
        self.assertTrue(any("UPDATE bot_params SET active = true" in statement for statement in connection.cursor_obj.statements))

    def test_review_active_param_effects_recommends_rollback_on_worse_metrics(self) -> None:
        now = datetime(2026, 5, 24, tzinfo=timezone.utc)
        connection = _FakeConnection(
            fetches=[
                (3, now),
                (
                    20,
                    now,
                    {
                        "closed_trade_result_count": 30,
                        "marked_pnl_latest": 0.0,
                        "realized_pnl_total": 1.0,
                        "win_rate": 0.55,
                        "max_drawdown_pnl": -1.0,
                    },
                ),
                (2,),
            ]
        )

        with patch("rebalancing.learning.params._with_connection", side_effect=lambda write: write(connection)):
            review = review_active_param_effects(
                {
                    "closed_trade_result_count": 30,
                    "marked_pnl_latest": -8.0,
                    "realized_pnl_total": -2.0,
                    "win_rate": 0.30,
                    "max_drawdown_pnl": -4.0,
                }
            )

        self.assertTrue(review["rollback_recommended"])
        self.assertEqual(review["active_version"], 3)
        self.assertEqual(review["rollback_target_version"], 2)
        self.assertLess(review["deltas"]["marked_pnl_latest"], 0)


class _FakeConnection:
    def __init__(self, *, fetches: list[tuple | None]) -> None:
        self.cursor_obj = _FakeCursor(fetches=fetches)

    def cursor(self) -> "_FakeCursor":
        return self.cursor_obj


class _FakeCursor:
    def __init__(self, *, fetches: list[tuple | None]) -> None:
        self.fetches = list(fetches)
        self.statements: list[str] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: str, _params: object = None) -> None:
        self.statements.append(statement)

    def fetchone(self) -> tuple | None:
        return self.fetches.pop(0)


if __name__ == "__main__":
    unittest.main()
