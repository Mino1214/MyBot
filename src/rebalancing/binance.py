from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .indicators import build_btc_snapshot
from .models import AccountSnapshot, Candle, MarketCandidate, PlannedOrder, Position, PositionSide


STABLE_BASE_ASSETS = {
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "DAI",
    "TUSD",
    "USDE",
    "USDS",
    "USD1",
    "PYUSD",
    "FRAX",
    "LUSD",
    "GUSD",
    "USDP",
    "EURC",
    "EURS",
    "SUSD",
}


@dataclass(frozen=True)
class BinanceCredentials:
    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "BinanceCredentials":
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
        return cls(api_key=api_key, api_secret=api_secret)


@dataclass(frozen=True)
class SymbolRules:
    quantity_step: Decimal
    price_tick: Decimal
    min_quantity: Decimal


@dataclass(frozen=True)
class BinanceOrderResult:
    symbol: str
    side: str
    order_type: str
    quantity: str
    reduce_only: bool
    live: bool
    response: dict[str, Any]


class BinanceFuturesClient:
    def __init__(
        self,
        credentials: BinanceCredentials | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 10.0,
        recv_window: int = 5_000,
    ) -> None:
        self.credentials = credentials
        self.base_url = (base_url or os.environ.get("BINANCE_FAPI_BASE_URL") or "https://fapi.binance.com").rstrip("/")
        self.timeout = timeout
        self.recv_window = recv_window
        self._exchange_info_cache: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> "BinanceFuturesClient":
        return cls(credentials=BinanceCredentials.from_env())

    def exchange_info(self) -> dict[str, Any]:
        if self._exchange_info_cache is not None:
            return self._exchange_info_cache
        self._exchange_info_cache = self._request("GET", "/fapi/v1/exchangeInfo")
        return self._exchange_info_cache

    def refresh_exchange_info(self) -> dict[str, Any]:
        self._exchange_info_cache = self._request("GET", "/fapi/v1/exchangeInfo")
        return self._exchange_info_cache

    def ticker_24hr(self, symbol: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else None
        return self._request("GET", "/fapi/v1/ticker/24hr", params=params)

    def book_ticker(self, symbol: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else None
        return self._request("GET", "/fapi/v1/ticker/bookTicker", params=params)

    def price(self, symbol: str) -> float:
        data = self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})
        return float(data["price"])

    def premium_index(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})

    def funding_rate(self, symbol: str) -> float:
        data = self.premium_index(symbol)
        return float(data.get("lastFundingRate") or 0.0)

    def klines(self, symbol: str, interval: str, *, limit: int = 500) -> list[Candle]:
        data = self._request(
            "GET",
            "/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        return [self._candle_from_kline(item) for item in data]

    def btc_market_snapshot(self):
        daily = self.klines("BTCUSDT", "1d", limit=240)
        four_hour = self.klines("BTCUSDT", "4h", limit=120)
        return build_btc_snapshot(daily=daily, four_hour=four_hour)

    def account(self) -> dict[str, Any]:
        self._require_credentials()
        return self._request("GET", "/fapi/v3/account", signed=True)

    def account_snapshot(
        self,
        *,
        day_start_equity: float,
        week_start_equity: float,
        month_start_equity: float,
    ) -> AccountSnapshot:
        account = self.account()
        return AccountSnapshot(
            equity=float(account["totalMarginBalance"]),
            wallet_balance=float(account["totalWalletBalance"]),
            day_start_equity=day_start_equity,
            week_start_equity=week_start_equity,
            month_start_equity=month_start_equity,
        )

    def positions(self) -> list[Position]:
        account = self.account()
        positions: list[Position] = []
        for raw in account.get("positions", []):
            position = self._position_from_account_position(raw)
            if position is not None:
                positions.append(position)
        return positions

    def market_candidates(self, now: datetime | None = None) -> list[MarketCandidate]:
        now = now or datetime.now(timezone.utc)
        exchange_info = self.exchange_info()
        tickers = self.ticker_24hr()
        book_tickers = self.book_ticker()
        tickers_by_symbol = {ticker["symbol"]: ticker for ticker in tickers}
        books_by_symbol = {ticker["symbol"]: ticker for ticker in book_tickers}

        candidates = []
        for raw_symbol in exchange_info.get("symbols", []):
            ticker = tickers_by_symbol.get(raw_symbol.get("symbol"))
            book = books_by_symbol.get(raw_symbol.get("symbol"))
            candidate = self._market_candidate_from_raw(raw_symbol, ticker, now, book)
            if candidate is not None:
                candidates.append(candidate)

        return self._with_volume_dominance(candidates)

    def top_dominance_candidates(self, *, limit: int = 10, now: datetime | None = None) -> list[MarketCandidate]:
        eligible = [
            candidate
            for candidate in self.market_candidates(now)
            if not candidate.stablecoin and candidate.base_asset.upper() not in STABLE_BASE_ASSETS
        ]
        return sorted(
            eligible,
            key=lambda candidate: (
                candidate.dominance_rank if candidate.dominance_rank is not None else 10_000,
                -(candidate.dominance_pct or 0.0),
                -candidate.quote_volume_24h,
            ),
        )[:limit]

    def symbol_rules(self, symbol: str) -> SymbolRules:
        for raw_symbol in self.exchange_info().get("symbols", []):
            if raw_symbol.get("symbol") != symbol:
                continue

            filters = {item.get("filterType"): item for item in raw_symbol.get("filters", [])}
            lot_filter = filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE") or {}
            price_filter = filters.get("PRICE_FILTER") or {}
            return SymbolRules(
                quantity_step=Decimal(str(lot_filter.get("stepSize", "0.001"))),
                price_tick=Decimal(str(price_filter.get("tickSize", "0.01"))),
                min_quantity=Decimal(str(lot_filter.get("minQty", "0"))),
            )

        raise ValueError(f"symbol not found in Binance exchangeInfo: {symbol}")

    def create_order(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require_credentials()
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def set_leverage(self, symbol: str, leverage: int = 2) -> dict[str, Any]:
        self._require_credentials()
        return self._request("POST", "/fapi/v1/leverage", params={"symbol": symbol, "leverage": leverage}, signed=True)

    def execute_planned_order(
        self,
        order: PlannedOrder,
        *,
        live: bool | None = None,
        limit_price_offset_bps: float = 3.0,
    ) -> BinanceOrderResult:
        live = live if live is not None else live_trading_enabled()
        reference_price = self._execution_reference_price(order, limit_price_offset_bps)
        rules = self.symbol_rules(order.symbol)
        quantity = self._floor_decimal(Decimal(str(order.notional)) / Decimal(str(reference_price)), rules.quantity_step)
        if quantity < rules.min_quantity or quantity <= 0:
            raise ValueError(f"planned order quantity is below Binance minimum for {order.symbol}")

        params: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": self._decimal_to_api(quantity),
            "newOrderRespType": "RESULT",
        }
        if order.reduce_only:
            params["reduceOnly"] = "true"
        if os.environ.get("BINANCE_POSITION_MODE", "ONE_WAY").upper() == "HEDGE":
            params["positionSide"] = order.position_side.value
        if order.order_type.value == "LIMIT":
            price = self._floor_decimal(Decimal(str(reference_price)), rules.price_tick)
            params["price"] = self._decimal_to_api(price)
            params["timeInForce"] = "GTC"

        if not live:
            return BinanceOrderResult(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=params["quantity"],
                reduce_only=order.reduce_only,
                live=False,
                response={"dry_run": True, "params": params, "reference_price": reference_price},
            )

        response = self.create_order(params)
        return BinanceOrderResult(
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=params["quantity"],
            reduce_only=order.reduce_only,
            live=True,
            response=response,
        )

    def execute_planned_orders(
        self,
        orders: tuple[PlannedOrder, ...],
        *,
        live: bool | None = None,
    ) -> tuple[BinanceOrderResult, ...]:
        return tuple(self.execute_planned_order(order, live=live) for order in orders)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = False,
    ):
        params = dict(params or {})
        headers = {}

        if signed:
            self._require_credentials()
            params.setdefault("recvWindow", self.recv_window)
            params["timestamp"] = int(time.time() * 1000)
            query = urlencode(params, doseq=True)
            params["signature"] = self._signature(query)
            headers["X-MBX-APIKEY"] = self.credentials.api_key  # type: ignore[union-attr]

        query_string = urlencode(params, doseq=True)
        url = f"{self.base_url}{path}"
        if query_string:
            url = f"{url}?{query_string}"

        request = Request(url, method=method, headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def _signature(self, query: str) -> str:
        self._require_credentials()
        return hmac.new(
            self.credentials.api_secret.encode("utf-8"),  # type: ignore[union-attr]
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _require_credentials(self) -> None:
        if self.credentials is None:
            raise RuntimeError("Binance credentials are required for signed endpoints")

    @staticmethod
    def _candle_from_kline(kline: list[Any]) -> Candle:
        return Candle(
            timestamp=datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[7]),
        )

    @staticmethod
    def _position_from_account_position(raw: dict[str, Any]) -> Position | None:
        amount = float(raw.get("positionAmt", 0.0))
        notional = abs(float(raw.get("notional", 0.0)))
        if notional == 0 and amount == 0:
            return None

        raw_side = raw.get("positionSide", "BOTH")
        if raw_side == "LONG" or (raw_side == "BOTH" and amount > 0):
            side = PositionSide.LONG
        elif raw_side == "SHORT" or (raw_side == "BOTH" and amount < 0):
            side = PositionSide.SHORT
        else:
            return None

        return Position(
            symbol=raw["symbol"],
            side=side,
            notional=notional,
            entry_price=BinanceFuturesClient._optional_float(
                raw.get("entryPrice"), zero_as_none=True
            ),
            quantity=abs(amount) or None,
            mark_price=BinanceFuturesClient._optional_float(
                raw.get("markPrice"), zero_as_none=True
            ),
            unrealized_pnl=BinanceFuturesClient._optional_float(
                raw.get("unRealizedProfit")
            ),
            liquidation_price=BinanceFuturesClient._optional_float(
                raw.get("liquidationPrice"), zero_as_none=True
            ),
            leverage=BinanceFuturesClient._optional_float(
                raw.get("leverage"), zero_as_none=True
            ),
            margin_type=str(raw.get("marginType", "")) or None,
        )

    @staticmethod
    def _optional_float(value: Any, *, zero_as_none: bool = False) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if zero_as_none and parsed == 0:
            return None
        return parsed

    @staticmethod
    def _market_candidate_from_raw(
        raw_symbol: dict[str, Any],
        ticker: dict[str, Any] | None,
        now: datetime,
        book_ticker: dict[str, Any] | None = None,
    ) -> MarketCandidate | None:
        if raw_symbol.get("contractType") != "PERPETUAL":
            return None
        if raw_symbol.get("status") != "TRADING":
            return None
        if raw_symbol.get("quoteAsset") != "USDT":
            return None

        onboard_ms = int(raw_symbol.get("onboardDate", 0))
        onboard = datetime.fromtimestamp(onboard_ms / 1000, tz=timezone.utc) if onboard_ms else now
        listed_days = max(0, (now - onboard).days)
        bid = float((book_ticker or {}).get("bidPrice", 0.0))
        ask = float((book_ticker or {}).get("askPrice", 0.0))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        spread_bps = ((ask - bid) / mid * 10_000) if mid > 0 and ask >= bid else 0.0
        base_asset = raw_symbol["baseAsset"]

        return MarketCandidate(
            symbol=raw_symbol["symbol"],
            base_asset=base_asset,
            quote_asset=raw_symbol["quoteAsset"],
            is_usdt_m_perp=True,
            quote_volume_24h=float((ticker or {}).get("quoteVolume", 0.0)),
            listed_days=listed_days,
            change_24h_pct=float((ticker or {}).get("priceChangePercent", 0.0)),
            spread_bps=spread_bps,
            stablecoin=base_asset.upper() in STABLE_BASE_ASSETS,
        )

    @staticmethod
    def _with_volume_dominance(candidates: list[MarketCandidate]) -> list[MarketCandidate]:
        total_volume = sum(candidate.quote_volume_24h for candidate in candidates if candidate.quote_volume_24h > 0)
        ranked = sorted(candidates, key=lambda candidate: candidate.quote_volume_24h, reverse=True)
        annotated: list[MarketCandidate] = []
        for rank, candidate in enumerate(ranked, start=1):
            dominance_pct = (candidate.quote_volume_24h / total_volume * 100) if total_volume > 0 else 0.0
            annotated.append(
                replace(
                    candidate,
                    dominance_rank=rank,
                    dominance_pct=dominance_pct,
                )
            )
        return annotated

    def _execution_reference_price(self, order: PlannedOrder, limit_price_offset_bps: float) -> float:
        if order.order_type.value == "MARKET":
            return self.price(order.symbol)

        book = self.book_ticker(order.symbol)
        bid = float(book["bidPrice"])
        ask = float(book["askPrice"])
        offset = limit_price_offset_bps / 10_000
        if order.side.value == "BUY":
            return bid * (1 - offset)
        return ask * (1 + offset)

    @staticmethod
    def _floor_decimal(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        return (value / step).to_integral_value(rounding=ROUND_DOWN) * step

    @staticmethod
    def _decimal_to_api(value: Decimal) -> str:
        normalized = value.normalize()
        return format(normalized, "f")


def live_trading_enabled() -> bool:
    return (
        os.environ.get("BINANCE_LIVE_TRADING", "").lower() == "true"
        and os.environ.get("BINANCE_ENABLE_ORDERS", "").lower() == "true"
    )
