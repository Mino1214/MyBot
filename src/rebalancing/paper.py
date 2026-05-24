from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .binance import BinanceFuturesClient
from .market_internals import apply_market_cap_dominance, build_market_internals
from .models import MarketBias, MarketCandidate, OrderSide, PositionSide, TargetPosition, TradeMode
from .portfolio import PortfolioBuilder
from .recording import record_paper_decision
from .tradingview import TradingViewAction, TradingViewAlert, TradingViewRegime, finalize_tradingview_alert


def paper_trading_enabled() -> bool:
    return os.environ.get("PAPER_TRADING_ENABLED", "true").lower() == "true"


def paper_state_path() -> Path:
    return Path(os.environ.get("PAPER_STATE_PATH", ".state/paper_trading.json"))


@dataclass(frozen=True)
class _TradeCosts:
    fee: float
    slippage: float
    liquidity: str
    fee_rate: float
    slippage_bps: float

    @property
    def total(self) -> float:
        return self.fee + self.slippage


def process_paper_alert(payload: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    alert = TradingViewAlert.parse(payload)
    alert, _decision = finalize_tradingview_alert(
        alert,
        max_leverage=_env_float("PAPER_MAX_LEVERAGE", 2.0),
    )
    state_path = path or paper_state_path()
    state = _load_state(state_path)
    client = BinanceFuturesClient()
    candidates = _market_candidates(client)
    position_symbols = {item["symbol"] for item in state.get("positions", [])}
    position_prices = _prices(client, position_symbols) if position_symbols else {}
    state = _mark_state(state, position_prices)
    previous_trade_count = len(state.get("trades", []))
    equity_before = float(state["equity"])
    targets = _targets_for_alert(alert, equity_before, candidates, state=state)
    target_symbols = {target.symbol for target in targets}
    target_prices = _prices(client, target_symbols - position_symbols)
    prices = {**position_prices, **target_prices}
    funding_rates = _funding_rates(client, position_symbols | target_symbols)

    update = _rebalance_state(
        state=state,
        alert=alert,
        targets=targets,
        prices=prices,
        funding_rates=funding_rates,
    )
    _write_state(state_path, update)
    record_paper_decision(
        alert=alert,
        decision=_decision,
        snapshot={
            "account": {"source": "paper", "equity": equity_before},
            "positions": state.get("positions", []),
            "candidates": candidates,
            "btc": None,
            "market_internals": {},
        },
        planned_orders=update.get("orders", []),
        executions=update.get("trades", [])[previous_trade_count:],
    )
    return paper_status_payload(path=state_path) or update


def paper_status_payload(*, path: Path | None = None) -> dict[str, Any] | None:
    state_path = path or paper_state_path()
    state = _load_state(state_path)
    if not state.get("last_signal"):
        return None

    client = BinanceFuturesClient()
    symbols = {item["symbol"] for item in state.get("positions", [])}
    prices = _prices(client, symbols) if symbols else {}
    marked = _mark_state(state, prices)
    return _state_payload(marked)


def _market_candidates(client: BinanceFuturesClient) -> list[MarketCandidate]:
    candidates = client.market_candidates()
    try:
        internals = build_market_internals(binance=client, candidates=candidates)
        return apply_market_cap_dominance(candidates, internals)
    except Exception:
        return candidates


def _targets_for_alert(
    alert: TradingViewAlert,
    equity: float,
    candidates: list[MarketCandidate],
    *,
    state: dict[str, Any] | None = None,
) -> tuple[TargetPosition, ...]:
    state = state or {}
    action = alert.decision_action
    if action == TradingViewAction.HOLD:
        return _state_position_targets(state)
    if action == TradingViewAction.REDUCE:
        return _state_position_targets(state, scale=0.5)
    if action == TradingViewAction.EXIT:
        return tuple()

    leverage = max(0.0, min(alert.target_leverage, _env_float("PAPER_MAX_LEVERAGE", 2.0)))
    if leverage <= 0 or alert.regime in {TradingViewRegime.RANGE, TradingViewRegime.CHAOTIC}:
        return tuple()

    desired_side = _alert_position_side(alert.regime)
    if desired_side is not None and _has_opposite_exposure(state, desired_side):
        return tuple()

    if _paper_new_entries_blocked(state, equity):
        return _state_position_targets(state)

    if _same_signal_rebalance_too_soon(alert, state):
        return _state_position_targets(state)

    builder = PortfolioBuilder()
    target_notional = equity * leverage

    if alert.regime == TradingViewRegime.TOP10_LONG:
        symbols = [item.symbol for item in builder.select_long_universe(candidates)]
        return _weighted_targets(
            symbols=symbols,
            side=PositionSide.LONG,
            target_notional=target_notional,
            core_weights={"BTCUSDT": 0.25, "ETHUSDT": 0.20},
            residual_weight=0.55,
        )

    if alert.regime == TradingViewRegime.BTC_ETH_LONG:
        eligible = {item.symbol: item for item in builder.select_long_universe(candidates)}
        symbols = [symbol for symbol in ("BTCUSDT", "ETHUSDT") if symbol in eligible]
        if not symbols:
            symbols = [item.symbol for item in builder.select_long_universe(candidates)[:2]]
        return _weighted_targets(
            symbols=symbols,
            side=PositionSide.LONG,
            target_notional=target_notional,
            core_weights={"BTCUSDT": 0.65, "ETHUSDT": 0.35},
            residual_weight=0.0,
        )

    symbols = [item.symbol for item in builder.select_short_universe(candidates)]
    if alert.regime == TradingViewRegime.ALT_WEAK_SHORT:
        return _weighted_targets(
            symbols=symbols,
            side=PositionSide.SHORT,
            target_notional=target_notional,
            core_weights={"BTCUSDT": 0.35, "ETHUSDT": 0.25},
            residual_weight=0.40,
        )

    return _weighted_targets(
        symbols=symbols,
        side=PositionSide.SHORT,
        target_notional=target_notional,
        core_weights={"BTCUSDT": 0.40, "ETHUSDT": 0.30},
        residual_weight=0.30,
    )


def _alert_position_side(regime: TradingViewRegime) -> PositionSide | None:
    if regime in {TradingViewRegime.TOP10_LONG, TradingViewRegime.BTC_ETH_LONG}:
        return PositionSide.LONG
    if regime in {TradingViewRegime.ALT_WEAK_SHORT, TradingViewRegime.SHORT_MODE}:
        return PositionSide.SHORT
    return None


def _has_opposite_exposure(state: Mapping[str, Any], desired_side: PositionSide) -> bool:
    desired = desired_side.value
    return any(
        isinstance(raw, Mapping)
        and float(raw.get("notional") or 0.0) >= _env_float("PAPER_MIN_ORDER_NOTIONAL", 10.0)
        and str(raw.get("side")) != desired
        for raw in state.get("positions", [])
    )


def _paper_new_entries_blocked(state: Mapping[str, Any], equity: float) -> bool:
    initial = float(state.get("initial_equity", _initial_equity()))
    if initial <= 0:
        return False

    loss_limit = _env_float("PAPER_BLOCK_NEW_ENTRIES_LOSS_PCT", -0.02)
    if loss_limit <= -1.0:
        return False

    return equity / initial - 1.0 <= loss_limit


def _same_signal_rebalance_too_soon(alert: TradingViewAlert, state: Mapping[str, Any]) -> bool:
    if not state.get("positions"):
        return False

    min_minutes = _env_float("PAPER_MIN_REBALANCE_MINUTES", 60.0)
    if min_minutes <= 0:
        return False

    last_signal = state.get("last_signal") or {}
    if not isinstance(last_signal, Mapping):
        return False
    if str(last_signal.get("regime")) != alert.regime.value:
        return False

    try:
        last_leverage = float(last_signal.get("target_leverage") or 0.0)
    except (TypeError, ValueError):
        return False
    if abs(last_leverage - alert.target_leverage) > 1e-9:
        return False

    last_rebalance = state.get("last_rebalance") or {}
    if not isinstance(last_rebalance, Mapping):
        return False

    last_time = _parse_datetime(str(last_rebalance.get("time") or ""))
    if last_time is None:
        return False

    return datetime.now(timezone.utc) - last_time < timedelta(minutes=min_minutes)


def _weighted_targets(
    *,
    symbols: list[str],
    side: PositionSide,
    target_notional: float,
    core_weights: dict[str, float],
    residual_weight: float,
) -> tuple[TargetPosition, ...]:
    if not symbols or target_notional <= 0:
        return tuple()

    symbol_set = set(symbols)
    weights: dict[str, float] = {}
    missing_weight = 0.0
    for symbol, weight in core_weights.items():
        if symbol in symbol_set:
            weights[symbol] = weight
        else:
            missing_weight += weight

    residual_symbols = [symbol for symbol in symbols if symbol not in weights]
    residual_total = residual_weight + missing_weight
    if residual_symbols:
        each = residual_total / len(residual_symbols)
        weights.update({symbol: each for symbol in residual_symbols})
    elif weights:
        scale = 1 / sum(weights.values())
        weights = {symbol: weight * scale for symbol, weight in weights.items()}

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return tuple()

    return tuple(
        TargetPosition(symbol, side, target_notional * weight / total_weight, weight / total_weight)
        for symbol, weight in weights.items()
    )


def _state_position_targets(state: Mapping[str, Any], *, scale: float = 1.0) -> tuple[TargetPosition, ...]:
    if scale <= 0:
        return tuple()

    min_order = _env_float("PAPER_MIN_ORDER_NOTIONAL", 10.0)
    targets: list[TargetPosition] = []
    for raw in state.get("positions", []):
        if not isinstance(raw, Mapping):
            continue
        notional = abs(float(raw.get("notional") or 0.0)) * scale
        if notional < min_order:
            continue
        try:
            side = PositionSide(str(raw["side"]))
        except (KeyError, ValueError):
            continue
        targets.append(TargetPosition(str(raw["symbol"]), side, notional))
    return tuple(targets)


def _paper_position_exit_reasons(
    positions: Any,
    *,
    now: datetime,
) -> dict[str, str]:
    take_profit = max(0.0, _env_float("PAPER_TAKE_PROFIT_PCT", 0.0))
    stop_loss = abs(_env_float("PAPER_STOP_LOSS_PCT", 0.0))
    max_minutes = _env_float("PAPER_MAX_POSITION_MINUTES", 0.0)
    reasons: dict[str, str] = {}
    for position in positions:
        if not isinstance(position, Mapping):
            continue
        symbol = str(position.get("symbol") or "")
        if not symbol:
            continue
        return_pct = _paper_position_return_pct(position)
        if take_profit > 0 and return_pct is not None and return_pct >= take_profit:
            reasons[symbol] = "take_profit"
            continue
        if stop_loss > 0 and return_pct is not None and return_pct <= -stop_loss:
            reasons[symbol] = "stop_loss"
            continue
        if max_minutes > 0 and _paper_position_age_minutes(position, now=now) >= max_minutes:
            reasons[symbol] = "max_position_minutes"
    return reasons


def _paper_position_return_pct(position: Mapping[str, Any]) -> float | None:
    entry = _optional_positive(position.get("entry_price"))
    price = _optional_positive(position.get("last_price"))
    side = str(position.get("side") or "").upper()
    if entry is None or price is None:
        return None
    if side == PositionSide.SHORT.value:
        return 1.0 - price / entry
    return price / entry - 1.0


def _paper_position_age_minutes(position: Mapping[str, Any], *, now: datetime) -> float:
    opened = _parse_datetime(str(position.get("opened_at") or ""))
    if opened is None:
        return 0.0
    return max(0.0, (now - opened).total_seconds() / 60)


def _accrue_funding(
    state: dict[str, Any],
    *,
    now: datetime,
    funding_rates: Mapping[str, float],
) -> dict[str, Any]:
    interval_hours = _env_float("PAPER_FUNDING_INTERVAL_HOURS", 8.0)
    if interval_hours <= 0:
        return state

    realized = float(state.get("realized_pnl", 0.0))
    funding_paid = float(state.get("funding_paid", 0.0))
    events = list(state.get("funding_events", []))
    positions: list[dict[str, Any]] = []
    for raw in state.get("positions", []):
        if not isinstance(raw, Mapping):
            continue
        position = dict(raw)
        cost, last_funding_at = _position_funding_cost(
            position,
            now=now,
            interval_hours=interval_hours,
            funding_rates=funding_rates,
        )
        if last_funding_at is not None:
            position["last_funding_at"] = last_funding_at.isoformat()
        if abs(cost) > 1e-12:
            symbol = str(position.get("symbol") or "")
            realized -= cost
            funding_paid += cost
            position["funding_paid"] = float(position.get("funding_paid") or 0.0) + cost
            events.append(
                {
                    "time": now.isoformat(),
                    "symbol": symbol,
                    "side": str(position.get("side") or ""),
                    "rate": _paper_funding_rate(symbol, funding_rates),
                    "funding": cost,
                    "cost": cost,
                }
            )
        positions.append(position)

    return {
        **state,
        "realized_pnl": realized,
        "funding_paid": funding_paid,
        "funding_events": events[-_env_int("PAPER_FUNDING_HISTORY_LIMIT", 200) :],
        "positions": positions,
    }


def _position_funding_cost(
    position: Mapping[str, Any],
    *,
    now: datetime,
    interval_hours: float,
    funding_rates: Mapping[str, float],
) -> tuple[float, datetime | None]:
    symbol = str(position.get("symbol") or "")
    side = str(position.get("side") or "").upper()
    if not symbol or side not in {PositionSide.LONG.value, PositionSide.SHORT.value}:
        return 0.0, None

    last = _parse_datetime(str(position.get("last_funding_at") or position.get("opened_at") or ""))
    if last is None:
        return 0.0, now
    if now <= last:
        return 0.0, last

    interval = timedelta(hours=interval_hours)
    elapsed = int((now - last).total_seconds() // interval.total_seconds())
    if elapsed <= 0:
        return 0.0, last

    price = _optional_positive(position.get("last_price")) or _optional_positive(position.get("entry_price"))
    if price is None:
        return 0.0, last + interval * elapsed
    notional = _position_notional(position, price)
    rate = _paper_funding_rate(symbol, funding_rates)
    direction = 1.0 if side == PositionSide.LONG.value else -1.0
    return notional * rate * elapsed * direction, last + interval * elapsed


def _paper_funding_rate(symbol: str, funding_rates: Mapping[str, float]) -> float:
    symbol_key = f"PAPER_FUNDING_RATE_{symbol.upper()}"
    if os.environ.get(symbol_key) is not None:
        return _env_float(symbol_key, 0.0)
    if symbol in funding_rates:
        return float(funding_rates[symbol])
    return _env_float("PAPER_FUNDING_RATE", 0.0)


def _rebalance_state(
    *,
    state: dict[str, Any],
    alert: TradingViewAlert,
    targets: tuple[TargetPosition, ...],
    prices: dict[str, float],
    funding_rates: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    state = _mark_state(state, prices)
    state = _accrue_funding(state, now=now, funding_rates=funding_rates or {})
    exposure_before = float(state.get("current_exposure", 0.0))
    realized = float(state.get("realized_pnl", 0.0))
    gross_realized = float(state.get("gross_realized_pnl", realized + _state_costs_paid(state)))
    fees_paid = float(state.get("fees_paid", 0.0))
    slippage_paid = float(state.get("slippage_paid", 0.0))
    funding_paid = float(state.get("funding_paid", 0.0))
    turnover = float(state.get("turnover", _trades_turnover(state.get("trades", []))))
    positions = {item["symbol"]: dict(item) for item in state.get("positions", [])}
    position_count_before = len(positions)
    exit_reasons = _paper_position_exit_reasons(positions.values(), now=now)
    effective_targets = tuple(target for target in targets if target.symbol not in exit_reasons)
    target_by_symbol = {target.symbol: target for target in effective_targets}
    target_exposure = sum(target.notional for target in effective_targets)
    min_order = _env_float("PAPER_MIN_ORDER_NOTIONAL", 10.0)
    orders: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = list(state.get("trades", []))

    for symbol in sorted(set(positions) - set(target_by_symbol)):
        position = positions.pop(symbol)
        price = prices.get(symbol)
        if price is None:
            positions[symbol] = position
            continue
        notional = _position_notional(position, price)
        gross_pnl = _position_unrealized(position, price)
        costs = _trade_costs(notional, liquidity=_paper_liquidity("exit"))
        gross_realized += gross_pnl
        fees_paid += costs.fee
        slippage_paid += costs.slippage
        turnover += notional
        realized += gross_pnl - costs.total
        if notional >= min_order:
            orders.append(
                _paper_order(
                    symbol,
                    _close_action(position["side"]),
                    position["side"],
                    notional,
                    True,
                    exit_reasons.get(symbol, "close_removed_target"),
                    costs=costs,
                )
            )
            trades.append(
                _trade_event(
                    now,
                    symbol,
                    "CLOSE",
                    position["side"],
                    notional,
                    price,
                    gross_pnl=gross_pnl,
                    costs=costs,
                    funding=float(position.get("funding_paid") or 0.0),
                )
            )

    for target in effective_targets:
        price = prices.get(target.symbol)
        if price is None:
            continue

        current = positions.get(target.symbol)
        if current and current["side"] != target.side.value:
            notional = _position_notional(current, price)
            gross_pnl = _position_unrealized(current, price)
            costs = _trade_costs(notional, liquidity=_paper_liquidity("exit"))
            gross_realized += gross_pnl
            fees_paid += costs.fee
            slippage_paid += costs.slippage
            turnover += notional
            realized += gross_pnl - costs.total
            if notional >= min_order:
                orders.append(
                    _paper_order(
                        target.symbol,
                        _close_action(current["side"]),
                        current["side"],
                        notional,
                        True,
                        "close_opposite_side",
                        costs=costs,
                    )
                )
                trades.append(
                    _trade_event(
                        now,
                        target.symbol,
                        "CLOSE",
                        current["side"],
                        notional,
                        price,
                        gross_pnl=gross_pnl,
                        costs=costs,
                        funding=float(current.get("funding_paid") or 0.0),
                    )
                )
            current = None
            positions.pop(target.symbol, None)

        if current is None:
            if target.notional >= min_order:
                quantity = target.notional / price
                costs = _trade_costs(target.notional, liquidity=_paper_liquidity("entry"))
                fees_paid += costs.fee
                slippage_paid += costs.slippage
                turnover += target.notional
                realized -= costs.total
                positions[target.symbol] = _position(target.symbol, target.side.value, quantity, price)
                orders.append(
                    _paper_order(
                        target.symbol,
                        _open_action(target.side.value),
                        target.side.value,
                        target.notional,
                        False,
                        "open_target",
                        costs=costs,
                    )
                )
                trades.append(
                    _trade_event(
                        now,
                        target.symbol,
                        "OPEN",
                        target.side.value,
                        target.notional,
                        price,
                        costs=costs,
                    )
                )
            continue

        current_notional = _position_notional(current, price)
        delta = target.notional - current_notional
        if abs(delta) < min_order:
            positions[target.symbol] = current
            continue

        if delta > 0:
            add_quantity = delta / price
            old_quantity = float(current["quantity"])
            new_quantity = old_quantity + add_quantity
            current["entry_price"] = (
                float(current["entry_price"]) * old_quantity + price * add_quantity
            ) / new_quantity
            current["quantity"] = new_quantity
            costs = _trade_costs(delta, liquidity=_paper_liquidity("rebalance"))
            fees_paid += costs.fee
            slippage_paid += costs.slippage
            turnover += delta
            realized -= costs.total
            orders.append(
                _paper_order(
                    target.symbol,
                    _open_action(target.side.value),
                    target.side.value,
                    delta,
                    False,
                    "increase_target",
                    costs=costs,
                )
            )
            trades.append(
                _trade_event(
                    now,
                    target.symbol,
                    "INCREASE",
                    target.side.value,
                    delta,
                    price,
                    costs=costs,
                )
            )
        else:
            reduce_notional = min(-delta, current_notional)
            reduce_quantity = reduce_notional / price
            fraction = min(1.0, reduce_quantity / float(current["quantity"]))
            gross_pnl = _position_unrealized(current, price) * fraction
            funding = float(current.get("funding_paid") or 0.0) * fraction
            costs = _trade_costs(reduce_notional, liquidity=_paper_liquidity("rebalance"))
            gross_realized += gross_pnl
            fees_paid += costs.fee
            slippage_paid += costs.slippage
            turnover += reduce_notional
            realized += gross_pnl - costs.total
            current["quantity"] = float(current["quantity"]) - reduce_quantity
            current["funding_paid"] = float(current.get("funding_paid") or 0.0) - funding
            orders.append(
                _paper_order(
                    target.symbol,
                    _close_action(target.side.value),
                    target.side.value,
                    reduce_notional,
                    True,
                    "reduce_target",
                    costs=costs,
                )
            )
            trades.append(
                _trade_event(
                    now,
                    target.symbol,
                    "REDUCE",
                    target.side.value,
                    reduce_notional,
                    price,
                    gross_pnl=gross_pnl,
                    costs=costs,
                    funding=funding,
                )
            )
            if float(current["quantity"]) <= 1e-12:
                positions.pop(target.symbol, None)
                continue

        positions[target.symbol] = current

    updated = {
        **state,
        "initial_equity": float(state.get("initial_equity", _initial_equity())),
        "realized_pnl": realized,
        "gross_realized_pnl": gross_realized,
        "fees_paid": fees_paid,
        "slippage_paid": slippage_paid,
        "funding_paid": funding_paid,
        "trading_costs": fees_paid + slippage_paid + funding_paid,
        "turnover": turnover,
        "positions": list(positions.values()),
        "targets": [_target_payload(target) for target in effective_targets],
        "orders": orders,
        "trades": trades[-_env_int("PAPER_TRADE_HISTORY_LIMIT", 500) :],
        "last_signal": _signal_payload(alert),
        "last_updated": now.isoformat(),
    }
    marked = _mark_state(updated, prices)
    rebalance = _rebalance_payload(
        now=now,
        alert=alert,
        exposure_before=exposure_before,
        exposure_after=float(marked.get("current_exposure", 0.0)),
        target_exposure=target_exposure,
        orders=orders,
        position_count_before=position_count_before,
        position_count_after=len(marked.get("positions", [])),
    )
    if orders or not marked.get("last_rebalance"):
        marked["last_rebalance"] = rebalance
    else:
        marked["last_check"] = rebalance
    return marked


def _marked_equity(state: dict[str, Any], client: BinanceFuturesClient) -> float:
    symbols = {item["symbol"] for item in state.get("positions", [])}
    prices = _prices(client, symbols) if symbols else {}
    return float(_mark_state(state, prices)["equity"])


def _mark_state(state: dict[str, Any], prices: dict[str, float]) -> dict[str, Any]:
    initial = float(state.get("initial_equity", _initial_equity()))
    realized = float(state.get("realized_pnl", 0.0))
    fees_paid = float(state.get("fees_paid", 0.0))
    slippage_paid = float(state.get("slippage_paid", 0.0))
    funding_paid = float(state.get("funding_paid", 0.0))
    trading_costs = (
        fees_paid + slippage_paid + funding_paid
        if (fees_paid or slippage_paid or funding_paid)
        else _state_costs_paid(state)
    )
    gross_realized = float(state.get("gross_realized_pnl", realized + trading_costs))
    turnover = float(state.get("turnover", _trades_turnover(state.get("trades", []))))
    positions = []
    unrealized = 0.0
    exposure = 0.0
    for raw in state.get("positions", []):
        position = dict(raw)
        price = prices.get(position["symbol"], position.get("last_price") or position.get("entry_price"))
        if price is None:
            continue
        position["last_price"] = float(price)
        position["notional"] = _position_notional(position, float(price))
        position["unrealized_pnl"] = _position_unrealized(position, float(price))
        unrealized += position["unrealized_pnl"]
        exposure += position["notional"]
        positions.append(position)

    equity = initial + realized + unrealized
    return {
        **state,
        "initial_equity": initial,
        "realized_pnl": realized,
        "gross_realized_pnl": gross_realized,
        "fees_paid": fees_paid,
        "slippage_paid": slippage_paid,
        "funding_paid": funding_paid,
        "trading_costs": trading_costs,
        "turnover": turnover,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "gross_total_pnl": gross_realized + unrealized,
        "equity": equity,
        "current_exposure": exposure,
        "positions": positions,
    }


def _state_payload(state: dict[str, Any]) -> dict[str, Any]:
    last_signal = state.get("last_signal") or {}
    target_exposure = sum(float(target.get("notional", 0.0)) for target in state.get("targets", []))
    equity = float(state.get("equity", _initial_equity()))
    total_pnl = float(state.get("total_pnl", 0.0))
    total_pnl_pct = total_pnl / float(state.get("initial_equity", _initial_equity())) * 100
    return {
        "enabled": True,
        "source": "Paper trading",
        "last_updated": state.get("last_updated"),
        "regime": last_signal.get("regime", "RANGE"),
        "market_bias": _signal_market_bias(str(last_signal.get("regime", "RANGE"))),
        "mode": _signal_mode(str(last_signal.get("regime", "RANGE"))),
        "equity": equity,
        "initial_equity": float(state.get("initial_equity", _initial_equity())),
        "realized_pnl": float(state.get("realized_pnl", 0.0)),
        "gross_realized_pnl": float(
            state.get("gross_realized_pnl", float(state.get("realized_pnl", 0.0)) + _state_costs_paid(state))
        ),
        "unrealized_pnl": float(state.get("unrealized_pnl", 0.0)),
        "total_pnl": total_pnl,
        "gross_total_pnl": float(state.get("gross_total_pnl", total_pnl)),
        "total_pnl_pct": total_pnl_pct,
        "fees_paid": float(state.get("fees_paid", 0.0)),
        "slippage_paid": float(state.get("slippage_paid", 0.0)),
        "funding_paid": float(state.get("funding_paid", 0.0)),
        "trading_costs": float(state.get("trading_costs", _state_costs_paid(state))),
        "turnover": float(state.get("turnover", _trades_turnover(state.get("trades", [])))),
        "current_exposure": float(state.get("current_exposure", 0.0)),
        "target_exposure": target_exposure,
        "leverage": float(state.get("current_exposure", 0.0)) / equity if equity > 0 else 0.0,
        "positions": [_position_payload(item) for item in state.get("positions", [])],
        "orders": list(state.get("orders", [])),
        "targets": list(state.get("targets", [])),
        "trades": list(reversed(state.get("trades", [])[-50:])),
        "funding_events": list(reversed(state.get("funding_events", [])[-20:])),
        "last_rebalance": dict(state.get("last_rebalance") or {}),
        "latest_signal_id": last_signal.get("signal_id"),
        "events": _paper_events(state),
    }


def _paper_events(state: dict[str, Any]) -> list[dict[str, str]]:
    payload = _state_payload_without_events(state)
    rebalance = dict(state.get("last_rebalance") or {})
    event_time = str(rebalance.get("time") or state.get("last_updated") or datetime.now(timezone.utc).isoformat())
    events = [
        {
            "time": event_time,
            "kind": str(rebalance.get("event_kind") or "PAPER"),
            "message": _rebalance_message(rebalance, payload),
        }
    ]
    for order in state.get("orders", [])[:12]:
        events.append(
            {
                "time": event_time,
                "kind": _order_event_kind(order),
                "message": _order_event_message(order),
            }
        )
    return events


def _state_payload_without_events(state: dict[str, Any]) -> dict[str, Any]:
    last_signal = state.get("last_signal") or {}
    equity = float(state.get("equity", _initial_equity()))
    total_pnl = float(state.get("total_pnl", 0.0))
    initial = float(state.get("initial_equity", _initial_equity()))
    return {
        "regime": last_signal.get("regime", "RANGE"),
        "equity": equity,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / initial * 100 if initial > 0 else 0.0,
        "trading_costs": float(state.get("trading_costs", _state_costs_paid(state))),
        "current_exposure": float(state.get("current_exposure", 0.0)),
    }


def _prices(client: BinanceFuturesClient, symbols: set[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol in symbols:
        try:
            prices[symbol] = client.price(symbol)
        except Exception:
            continue
    return prices


def _funding_rates(client: BinanceFuturesClient, symbols: set[str]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for symbol in symbols:
        try:
            rates[symbol] = client.funding_rate(symbol)
        except Exception:
            continue
    return rates


def _position(symbol: str, side: str, quantity: float, entry_price: float) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "entry_price": entry_price,
        "last_price": entry_price,
        "notional": quantity * entry_price,
        "unrealized_pnl": 0.0,
        "opened_at": now,
        "last_funding_at": now,
        "funding_paid": 0.0,
    }


def _position_notional(position: Mapping[str, Any], price: float) -> float:
    return abs(float(position["quantity"]) * price)


def _position_unrealized(position: Mapping[str, Any], price: float) -> float:
    quantity = float(position["quantity"])
    entry = float(position["entry_price"])
    if position["side"] == PositionSide.SHORT.value:
        return quantity * (entry - price)
    return quantity * (price - entry)


def _trade_costs(notional: float, *, liquidity: str | None = None) -> _TradeCosts:
    absolute_notional = abs(notional)
    resolved = (liquidity or _paper_liquidity("default")).lower()
    if resolved not in {"maker", "taker"}:
        resolved = "taker"
    fee_rate = _paper_fee_rate(resolved)
    slippage_bps = _paper_slippage_bps(resolved)
    slippage_rate = slippage_bps / 10_000
    return _TradeCosts(
        fee=absolute_notional * fee_rate,
        slippage=absolute_notional * slippage_rate,
        liquidity=resolved,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
    )


def _paper_liquidity(kind: str) -> str:
    specific = os.environ.get(f"PAPER_{kind.upper()}_LIQUIDITY")
    value = (specific or os.environ.get("PAPER_DEFAULT_LIQUIDITY") or "taker").lower()
    return value if value in {"maker", "taker"} else "taker"


def _paper_fee_rate(liquidity: str) -> float:
    legacy = _env_float("PAPER_FEE_RATE", 0.0004)
    return max(0.0, _env_float(f"PAPER_{liquidity.upper()}_FEE_RATE", legacy))


def _paper_slippage_bps(liquidity: str) -> float:
    legacy = _env_float("PAPER_SLIPPAGE_BPS", 0.0)
    return max(0.0, _env_float(f"PAPER_{liquidity.upper()}_SLIPPAGE_BPS", legacy))


def _state_costs_paid(state: Mapping[str, Any]) -> float:
    if "trading_costs" in state:
        return float(state.get("trading_costs") or 0.0)
    return (
        float(state.get("fees_paid") or 0.0)
        + float(state.get("slippage_paid") or 0.0)
        + float(state.get("funding_paid") or 0.0)
    )


def _trades_turnover(trades: Any) -> float:
    if not isinstance(trades, list):
        return 0.0
    total = 0.0
    for trade in trades:
        if isinstance(trade, Mapping):
            total += abs(float(trade.get("notional") or 0.0))
    return total


def _open_action(side: str) -> str:
    return OrderSide.BUY.value if side == PositionSide.LONG.value else OrderSide.SELL.value


def _close_action(side: str) -> str:
    return OrderSide.SELL.value if side == PositionSide.LONG.value else OrderSide.BUY.value


def _paper_order(
    symbol: str,
    action: str,
    side: str,
    notional: float,
    reduce_only: bool,
    reason: str,
    *,
    costs: _TradeCosts | None = None,
) -> dict[str, Any]:
    costs = costs or _trade_costs(notional)
    return {
        "symbol": symbol,
        "action": action,
        "side": side,
        "notional": notional,
        "order_type": "MARKET",
        "reduce_only": reduce_only,
        "reason": reason,
        "liquidity": costs.liquidity,
        "fee_rate": costs.fee_rate,
        "slippage_bps": costs.slippage_bps,
        "fee": costs.fee,
        "slippage": costs.slippage,
        "cost": costs.total,
    }


def _rebalance_payload(
    *,
    now: datetime,
    alert: TradingViewAlert,
    exposure_before: float,
    exposure_after: float,
    target_exposure: float,
    orders: list[dict[str, Any]],
    position_count_before: int,
    position_count_after: int,
) -> dict[str, Any]:
    event_kind = _rebalance_event_kind(exposure_before, exposure_after, orders)
    opened_symbols = _order_symbols(orders, "open")
    increased_symbols = _order_symbols(orders, "increase")
    reduced_symbols = _order_symbols(orders, "reduce")
    closed_symbols = _order_symbols(orders, "close")
    changed_symbols = sorted({str(order.get("symbol")) for order in orders if order.get("symbol")})
    return {
        "time": now.isoformat(),
        "event_kind": event_kind,
        "regime": alert.regime.value,
        "mode": _signal_mode(alert.regime.value),
        "from_exposure": exposure_before,
        "to_exposure": exposure_after,
        "target_exposure": target_exposure,
        "delta_exposure": exposure_after - exposure_before,
        "order_count": len(orders),
        "open_count": sum(1 for order in orders if not bool(order.get("reduce_only"))),
        "close_count": sum(1 for order in orders if bool(order.get("reduce_only"))),
        "position_count_before": position_count_before,
        "position_count_after": position_count_after,
        "gross_order_notional": sum(abs(float(order.get("notional") or 0.0)) for order in orders),
        "changed_symbols": changed_symbols,
        "opened_symbols": opened_symbols,
        "increased_symbols": increased_symbols,
        "reduced_symbols": reduced_symbols,
        "closed_symbols": closed_symbols,
        "signal_id": alert.dedupe_key(),
    }


def _rebalance_event_kind(
    exposure_before: float,
    exposure_after: float,
    orders: list[dict[str, Any]],
) -> str:
    if not orders:
        return "PAPER_HOLD"
    if exposure_before <= 1e-9 and exposure_after > 1e-9:
        return "PAPER_ENTRY"
    if exposure_before > 1e-9 and exposure_after <= 1e-9:
        return "PAPER_EXIT"
    return "PAPER_REBALANCE"


def _rebalance_message(rebalance: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    if not rebalance:
        return (
            f"Paper {payload['regime']} equity={payload['equity']:.2f} "
            f"PnL={payload['total_pnl']:+.2f} ({payload['total_pnl_pct']:+.2f}%) "
            f"costs={float(payload.get('trading_costs') or 0):.2f} "
            f"exposure={payload['current_exposure']:.2f}"
        )

    label = {
        "PAPER_ENTRY": "Entry",
        "PAPER_REBALANCE": "Rebalance",
        "PAPER_EXIT": "Exit",
        "PAPER_HOLD": "Hold",
    }.get(str(rebalance.get("event_kind")), "Paper")
    return (
        f"{label} {rebalance.get('regime', payload['regime'])} "
        f"{float(rebalance.get('from_exposure') or 0):.2f} -> "
        f"{float(rebalance.get('to_exposure') or 0):.2f} "
        f"(target {float(rebalance.get('target_exposure') or 0):.2f}, "
        f"positions {int(rebalance.get('position_count_before') or 0)}->"
        f"{int(rebalance.get('position_count_after') or 0)}, "
        f"turnover {float(rebalance.get('gross_order_notional') or 0):.2f}, "
        f"costs {float(payload.get('trading_costs') or 0):.2f}, "
        f"orders {int(rebalance.get('order_count') or 0)})"
    )


def _order_symbols(orders: list[dict[str, Any]], prefix: str) -> list[str]:
    return sorted(
        {
            str(order.get("symbol"))
            for order in orders
            if order.get("symbol")
            and (
                str(order.get("reason") or "").startswith(prefix)
                or (prefix == "close" and _is_paper_exit_reason(str(order.get("reason") or "")))
            )
        }
    )


def _order_event_kind(order: Mapping[str, Any]) -> str:
    reason = str(order.get("reason") or "")
    if reason.startswith("open"):
        return "PAPER_ENTRY"
    if reason.startswith("close") or _is_paper_exit_reason(reason):
        return "PAPER_EXIT"
    if reason.startswith(("increase", "reduce")):
        return "PAPER_REBALANCE"
    return "PAPER_ORDER"


def _is_paper_exit_reason(reason: str) -> bool:
    return reason in {"take_profit", "stop_loss", "max_position_minutes"}


def _order_event_message(order: Mapping[str, Any]) -> str:
    reason = str(order.get("reason") or "")
    if reason.startswith("open"):
        label = "Entry"
    elif reason.startswith("close") or _is_paper_exit_reason(reason):
        label = "Exit"
    elif reason.startswith("increase"):
        label = "Increase"
    elif reason.startswith("reduce"):
        label = "Reduce"
    else:
        label = "Order"
    return (
        f"{label} {order['symbol']} {order['side']} "
        f"{float(order['notional']):.2f} cost={float(order.get('cost') or 0):.2f} {reason}"
    )


def _trade_event(
    now: datetime,
    symbol: str,
    action: str,
    side: str,
    notional: float,
    price: float,
    *,
    gross_pnl: float = 0.0,
    costs: _TradeCosts | None = None,
    funding: float = 0.0,
) -> dict[str, Any]:
    costs = costs or _trade_costs(notional)
    cost = costs.total + funding
    return {
        "time": now.isoformat(),
        "symbol": symbol,
        "action": action,
        "side": side,
        "notional": notional,
        "price": price,
        "gross_pnl": gross_pnl,
        "liquidity": costs.liquidity,
        "fee_rate": costs.fee_rate,
        "slippage_bps": costs.slippage_bps,
        "fee": costs.fee,
        "slippage": costs.slippage,
        "funding": funding,
        "cost": cost,
        "net_pnl": gross_pnl - cost,
    }


def _position_payload(position: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": position["symbol"],
        "side": position["side"],
        "quantity": float(position.get("quantity", 0.0)),
        "notional": float(position.get("notional", 0.0)),
        "entry_price": float(position.get("entry_price", 0.0)),
        "mark_price": float(position.get("last_price", 0.0)),
        "unrealized_pnl": float(position.get("unrealized_pnl", 0.0)),
        "funding_paid": float(position.get("funding_paid", 0.0)),
        "opened_at": position.get("opened_at"),
        "last_funding_at": position.get("last_funding_at"),
    }


def _target_payload(target: TargetPosition) -> dict[str, Any]:
    return {
        "symbol": target.symbol,
        "side": target.side.value,
        "notional": target.notional,
        "weight": target.weight,
    }


def _signal_payload(alert: TradingViewAlert) -> dict[str, Any]:
    return {
        "regime": alert.regime.value,
        "target_leverage": alert.target_leverage,
        "decision_action": alert.decision_action.value if alert.decision_action else None,
        "score": alert.score,
        "tf": alert.tf,
        "time_ms": alert.time_ms,
        "bar_time_ms": alert.bar_time_ms,
        "signal_id": alert.dedupe_key(),
    }


def _signal_mode(regime: str) -> str:
    if regime in {TradingViewRegime.TOP10_LONG.value, TradingViewRegime.BTC_ETH_LONG.value}:
        return TradeMode.LONG.value
    if regime in {TradingViewRegime.ALT_WEAK_SHORT.value, TradingViewRegime.SHORT_MODE.value}:
        return TradeMode.SHORT.value
    if regime == TradingViewRegime.CHAOTIC.value:
        return TradeMode.PAUSED.value
    return TradeMode.NEUTRAL.value


def _signal_market_bias(regime: str) -> str:
    return {
        TradingViewRegime.TOP10_LONG.value: MarketBias.BROAD_BULL.value,
        TradingViewRegime.BTC_ETH_LONG.value: MarketBias.BTC_ONLY_BULL.value,
        TradingViewRegime.ALT_WEAK_SHORT.value: MarketBias.ALT_WEAK_BEAR.value,
        TradingViewRegime.SHORT_MODE.value: MarketBias.BROAD_BEAR.value,
        TradingViewRegime.CHAOTIC.value: MarketBias.CHAOTIC.value,
    }.get(regime, MarketBias.RANGE.value)


def _load_state(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"initial_equity": _initial_equity(), "realized_pnl": 0.0, "positions": [], "trades": []}
    return dict(data) if isinstance(data, dict) else {"initial_equity": _initial_equity(), "realized_pnl": 0.0, "positions": [], "trades": []}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def _initial_equity() -> float:
    return _env_float("PAPER_INITIAL_EQUITY", _env_float("FALLBACK_EQUITY", 3300.0))


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default
