from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from rebalancing.models import MarketCandidate, PositionSide, TargetPosition
from rebalancing.paper import _mark_state, _rebalance_state, _targets_for_alert
from rebalancing.tradingview import TradingViewAlert


class PaperTradingTest(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(
            os.environ,
            {
                "PAPER_FEE_RATE": "0.0004",
                "PAPER_SLIPPAGE_BPS": "0",
                "PAPER_MAKER_FEE_RATE": "0.0004",
                "PAPER_TAKER_FEE_RATE": "0.0004",
                "PAPER_MAKER_SLIPPAGE_BPS": "0",
                "PAPER_TAKER_SLIPPAGE_BPS": "0",
                "PAPER_ENTRY_LIQUIDITY": "taker",
                "PAPER_REBALANCE_LIQUIDITY": "taker",
                "PAPER_EXIT_LIQUIDITY": "taker",
                "PAPER_FUNDING_RATE": "0",
                "PAPER_FUNDING_INTERVAL_HOURS": "8",
                "PAPER_TAKE_PROFIT_PCT": "0",
                "PAPER_STOP_LOSS_PCT": "0",
                "PAPER_MAX_POSITION_MINUTES": "0",
            },
        )
        env.start()
        self.addCleanup(env.stop)

    def test_open_marks_and_closes_long_position(self) -> None:
        alert = _alert("TOP10_LONG", 1.0)
        opened = _rebalance_state(
            state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
            alert=alert,
            targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
            prices={"BTCUSDT": 100.0},
        )
        marked = _mark_state(opened, {"BTCUSDT": 110.0})

        self.assertEqual(len(marked["positions"]), 1)
        self.assertAlmostEqual(marked["unrealized_pnl"], 10.0)
        self.assertAlmostEqual(marked["fees_paid"], 0.04)
        self.assertAlmostEqual(marked["trading_costs"], 0.04)
        self.assertAlmostEqual(marked["equity"], 1_009.96)
        self.assertEqual(opened["last_rebalance"]["position_count_before"], 0)
        self.assertEqual(opened["last_rebalance"]["position_count_after"], 1)
        self.assertEqual(opened["last_rebalance"]["opened_symbols"], ["BTCUSDT"])

        closed = _rebalance_state(
            state=marked,
            alert=_alert("RANGE", 0.0),
            targets=tuple(),
            prices={"BTCUSDT": 110.0},
        )

        self.assertEqual(closed["positions"], [])
        self.assertAlmostEqual(closed["gross_realized_pnl"], 10.0)
        self.assertAlmostEqual(closed["fees_paid"], 0.084)
        self.assertAlmostEqual(closed["realized_pnl"], 9.916)
        self.assertAlmostEqual(closed["equity"], 1_009.916)

    def test_short_position_profits_when_price_falls(self) -> None:
        opened = _rebalance_state(
            state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
            alert=_alert("SHORT_MODE", 1.0),
            targets=(TargetPosition("BTCUSDT", PositionSide.SHORT, 100.0, 1.0),),
            prices={"BTCUSDT": 100.0},
        )
        marked = _mark_state(opened, {"BTCUSDT": 90.0})

        self.assertAlmostEqual(marked["unrealized_pnl"], 10.0)
        self.assertAlmostEqual(marked["fees_paid"], 0.04)
        self.assertAlmostEqual(marked["equity"], 1_009.96)

    def test_hold_does_not_overwrite_last_actionable_rebalance(self) -> None:
        opened = _rebalance_state(
            state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
            alert=_alert("TOP10_LONG", 1.0),
            targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
            prices={"BTCUSDT": 100.0},
        )
        held = _rebalance_state(
            state=opened,
            alert=_alert("TOP10_LONG", 1.0),
            targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
            prices={"BTCUSDT": 100.0},
        )

        self.assertEqual(held["last_rebalance"]["event_kind"], "PAPER_ENTRY")
        self.assertEqual(held["last_check"]["event_kind"], "PAPER_HOLD")
        self.assertEqual(held["last_check"]["position_count_before"], 1)
        self.assertEqual(held["last_check"]["position_count_after"], 1)

    def test_mtf_hold_action_preserves_current_position(self) -> None:
        opened = _rebalance_state(
            state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
            alert=_alert("TOP10_LONG", 1.0),
            targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
            prices={"BTCUSDT": 100.0},
        )
        alert = _alert("TOP10_LONG", 2.0, decision_action="HOLD")
        targets = _targets_for_alert(alert, 1_000.0, [], state=opened)

        held = _rebalance_state(
            state=opened,
            alert=alert,
            targets=targets,
            prices={"BTCUSDT": 100.0},
        )

        self.assertEqual(len(held["positions"]), 1)
        self.assertEqual(held["orders"], [])
        self.assertAlmostEqual(held["current_exposure"], 100.0)

    def test_mtf_reduce_action_halves_current_position(self) -> None:
        opened = _rebalance_state(
            state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
            alert=_alert("TOP10_LONG", 1.0),
            targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
            prices={"BTCUSDT": 100.0},
        )
        alert = _alert("TOP10_LONG", 2.0, decision_action="REDUCE")
        targets = _targets_for_alert(alert, 1_000.0, [], state=opened)

        reduced = _rebalance_state(
            state=opened,
            alert=alert,
            targets=targets,
            prices={"BTCUSDT": 100.0},
        )

        self.assertEqual(len(reduced["positions"]), 1)
        self.assertAlmostEqual(reduced["current_exposure"], 50.0)
        self.assertEqual(reduced["orders"][0]["reason"], "reduce_target")

    def test_opposite_direction_alert_exits_before_flipping(self) -> None:
        opened = _rebalance_state(
            state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
            alert=_alert("TOP10_LONG", 1.0),
            targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
            prices={"BTCUSDT": 100.0},
        )
        alert = _alert("SHORT_MODE", 0.8)
        targets = _targets_for_alert(alert, 1_000.0, _candidates(), state=opened)

        closed = _rebalance_state(
            state=opened,
            alert=alert,
            targets=targets,
            prices={"BTCUSDT": 99.0},
        )

        self.assertEqual(targets, tuple())
        self.assertEqual(closed["positions"], [])
        self.assertEqual(closed["last_rebalance"]["event_kind"], "PAPER_EXIT")

    def test_same_signal_before_min_interval_preserves_current_positions(self) -> None:
        with patch.dict(os.environ, {"PAPER_MIN_REBALANCE_MINUTES": "60"}):
            opened = _rebalance_state(
                state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
                alert=_alert("SHORT_MODE", 0.8),
                targets=(TargetPosition("BTCUSDT", PositionSide.SHORT, 100.0, 1.0),),
                prices={"BTCUSDT": 100.0},
            )
            targets = _targets_for_alert(_alert("SHORT_MODE", 0.8), 1_000.0, _candidates(), state=opened)

        self.assertEqual({target.symbol for target in targets}, {"BTCUSDT"})
        self.assertEqual(targets[0].side, PositionSide.SHORT)
        self.assertAlmostEqual(targets[0].notional, 100.0)

    def test_loss_limit_blocks_new_entries(self) -> None:
        state = {"initial_equity": 1_000.0, "realized_pnl": -30.0, "positions": [], "trades": []}

        targets = _targets_for_alert(_alert("TOP10_LONG", 2.0), 970.0, _candidates(), state=state)

        self.assertEqual(targets, tuple())

    def test_maker_taker_fee_and_slippage_are_applied_by_liquidity(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PAPER_MAKER_FEE_RATE": "0.0002",
                "PAPER_TAKER_FEE_RATE": "0.0005",
                "PAPER_MAKER_SLIPPAGE_BPS": "0",
                "PAPER_TAKER_SLIPPAGE_BPS": "2",
                "PAPER_ENTRY_LIQUIDITY": "maker",
                "PAPER_EXIT_LIQUIDITY": "taker",
            },
        ):
            opened = _rebalance_state(
                state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
                alert=_alert("TOP10_LONG", 1.0),
                targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
                prices={"BTCUSDT": 100.0},
            )
            marked = _mark_state(opened, {"BTCUSDT": 110.0})
            closed = _rebalance_state(
                state=marked,
                alert=_alert("RANGE", 0.0),
                targets=tuple(),
                prices={"BTCUSDT": 110.0},
            )

        self.assertEqual(opened["orders"][0]["liquidity"], "maker")
        self.assertAlmostEqual(opened["orders"][0]["fee"], 0.02)
        self.assertAlmostEqual(opened["orders"][0]["slippage"], 0.0)
        self.assertEqual(closed["orders"][0]["liquidity"], "taker")
        self.assertAlmostEqual(closed["orders"][0]["fee"], 0.055)
        self.assertAlmostEqual(closed["orders"][0]["slippage"], 0.022)
        self.assertAlmostEqual(closed["realized_pnl"], 9.903)

    def test_funding_accrues_into_realized_pnl_and_position_state(self) -> None:
        opened_at = (datetime.now(timezone.utc) - timedelta(hours=9)).isoformat()
        state = {
            "initial_equity": 1_000.0,
            "realized_pnl": 0.0,
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "quantity": 1.0,
                    "entry_price": 100.0,
                    "last_price": 100.0,
                    "opened_at": opened_at,
                    "last_funding_at": opened_at,
                    "funding_paid": 0.0,
                }
            ],
            "trades": [],
        }

        updated = _rebalance_state(
            state=state,
            alert=_alert("TOP10_LONG", 1.0, decision_action="HOLD"),
            targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
            prices={"BTCUSDT": 100.0},
            funding_rates={"BTCUSDT": 0.001},
        )

        self.assertAlmostEqual(updated["funding_paid"], 0.1)
        self.assertAlmostEqual(updated["positions"][0]["funding_paid"], 0.1)
        self.assertAlmostEqual(updated["realized_pnl"], -0.1)
        self.assertAlmostEqual(updated["trading_costs"], 0.1)

    def test_take_profit_exit_closes_position_even_when_target_remains(self) -> None:
        with patch.dict(os.environ, {"PAPER_TAKE_PROFIT_PCT": "0.05"}):
            opened = _rebalance_state(
                state={"initial_equity": 1_000.0, "realized_pnl": 0.0, "positions": [], "trades": []},
                alert=_alert("TOP10_LONG", 1.0),
                targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
                prices={"BTCUSDT": 100.0},
            )
            marked = _mark_state(opened, {"BTCUSDT": 106.0})
            closed = _rebalance_state(
                state=marked,
                alert=_alert("TOP10_LONG", 1.0),
                targets=(TargetPosition("BTCUSDT", PositionSide.LONG, 100.0, 1.0),),
                prices={"BTCUSDT": 106.0},
            )

        self.assertEqual(closed["positions"], [])
        self.assertEqual(closed["orders"][0]["reason"], "take_profit")


def _alert(regime: str, leverage: float, *, decision_action: str | None = None) -> TradingViewAlert:
    payload = {
            "schema": "crypto_regime_v1",
            "source": "tradingview",
            "regime": regime,
            "target_leverage": leverage,
            "btc_up": regime == "TOP10_LONG",
            "btc_down": regime == "SHORT_MODE",
            "total_up": regime == "TOP10_LONG",
            "total_down": regime == "SHORT_MODE",
            "total2_up": regime == "TOP10_LONG",
            "total2_down": regime == "SHORT_MODE",
            "total3_weak": regime == "SHORT_MODE",
            "btcd_up": regime == "SHORT_MODE",
            "btcd_down": regime == "TOP10_LONG",
            "tf": "5",
            "confirmed": True,
            "time_ms": 1_779_242_700_000,
            "bar_time_ms": 1_779_242_700_000,
            "signal_id": f"{regime}_5_1779242700000",
        }
    if decision_action is not None:
        payload["decision_action"] = decision_action
    return TradingViewAlert.parse(payload)


def _candidates() -> list[MarketCandidate]:
    assets = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE"]
    return [
        MarketCandidate(
            symbol=f"{asset}USDT",
            base_asset=asset,
            quote_volume_24h=1_000_000_000 - index,
            listed_days=1_000,
            market_cap_rank=index + 1,
        )
        for index, asset in enumerate(assets)
    ]


if __name__ == "__main__":
    unittest.main()
