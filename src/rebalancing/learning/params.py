from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from typing import Any, Mapping, Sequence

from rebalancing.models import EngineConfig
from rebalancing.recording import _jsonb, _with_connection


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    minimum: float
    maximum: float
    kind: str = "float"

    def clamp(self, value: Any) -> tuple[int | float | None, bool]:
        parsed = _numeric(value)
        if parsed is None:
            return None, False

        bounded = min(max(parsed, self.minimum), self.maximum)
        clamped = bounded != parsed
        if self.kind == "int":
            return int(round(bounded)), clamped
        return float(bounded), clamped


PARAM_SPECS: dict[str, ParamSpec] = {
    "bull_target_leverage": ParamSpec("bull_target_leverage", 0.5, 2.0),
    "btc_only_target_leverage": ParamSpec("btc_only_target_leverage", 0.25, 1.5),
    "range_target_leverage": ParamSpec("range_target_leverage", 0.0, 0.5),
    "bear_initial_leverage": ParamSpec("bear_initial_leverage", 0.0, 1.0),
    "bear_confirmed_leverage": ParamSpec("bear_confirmed_leverage", 0.25, 1.5),
    "bear_strong_leverage": ParamSpec("bear_strong_leverage", 0.5, 2.0),
    "bear_initial_hours": ParamSpec("bear_initial_hours", 4.0, 72.0),
    "bear_strong_adx": ParamSpec("bear_strong_adx", 22.0, 45.0),
    "adx_threshold": ParamSpec("adx_threshold", 12.0, 30.0),
    "market_index_adx_threshold": ParamSpec("market_index_adx_threshold", 10.0, 30.0),
    "bull_score_threshold": ParamSpec("bull_score_threshold", 50.0, 90.0),
    "bear_score_threshold": ParamSpec("bear_score_threshold", -90.0, -50.0),
    "confirmation_candles": ParamSpec("confirmation_candles", 1.0, 6.0, kind="int"),
    "min_neutral_hours": ParamSpec("min_neutral_hours", 1.0, 48.0),
    "chaotic_cooldown_hours": ParamSpec("chaotic_cooldown_hours", 4.0, 96.0),
    "post_loss_cooldown_hours": ParamSpec("post_loss_cooldown_hours", 12.0, 168.0),
    "chaotic_4h_change_pct": ParamSpec("chaotic_4h_change_pct", 3.0, 12.0),
    "chaotic_atr_multiplier": ParamSpec("chaotic_atr_multiplier", 1.25, 4.0),
    "chaotic_volume_multiplier": ParamSpec("chaotic_volume_multiplier", 1.5, 6.0),
    "overheated_funding_rate": ParamSpec("overheated_funding_rate", 0.0003, 0.003),
    "min_quote_volume_24h": ParamSpec("min_quote_volume_24h", 10_000_000.0, 200_000_000.0),
    "max_spread_bps": ParamSpec("max_spread_bps", 2.0, 30.0),
    "min_listed_days": ParamSpec("min_listed_days", 7.0, 180.0, kind="int"),
    "max_abs_change_24h_pct": ParamSpec("max_abs_change_24h_pct", 10.0, 80.0),
    "long_universe_size": ParamSpec("long_universe_size", 2.0, 15.0, kind="int"),
    "short_alt_count": ParamSpec("short_alt_count", 0.0, 8.0, kind="int"),
    "drift_threshold": ParamSpec("drift_threshold", 0.05, 0.5),
    "order_split_notional": ParamSpec("order_split_notional", 20.0, 1000.0),
    "regular_rebalance_hours": ParamSpec("regular_rebalance_hours", 24.0, 336.0),
    "daily_loss_limit_pct": ParamSpec("daily_loss_limit_pct", -0.08, -0.005),
    "weekly_loss_limit_pct": ParamSpec("weekly_loss_limit_pct", -0.15, -0.01),
    "monthly_loss_limit_pct": ParamSpec("monthly_loss_limit_pct", -0.25, -0.03),
}

_ENGINE_CONFIG_FIELDS = {field.name for field in fields(EngineConfig)}


def active_engine_config() -> EngineConfig:
    return engine_config_from_params(load_active_bot_params())


def load_active_bot_params() -> dict[str, Any]:
    try:
        params = _with_connection(_read_active_params)
        return params if isinstance(params, dict) else {}
    except Exception as exc:
        logger.warning("DB active bot params load failed: %s", exc)
        return {}


def engine_config_from_params(params: Mapping[str, Any] | None) -> EngineConfig:
    values = asdict(EngineConfig())
    if not isinstance(params, Mapping):
        return EngineConfig()

    for name, value in params.items():
        if name not in values:
            continue
        spec = PARAM_SPECS.get(name)
        if spec is not None:
            parsed, _ = spec.clamp(value)
            if parsed is not None:
                values[name] = parsed
            continue
        coerced = _coerce_like(value, values[name])
        if coerced is not None:
            values[name] = coerced

    return EngineConfig(**values)


def prepare_param_update(
    base_params: Mapping[str, Any] | None,
    suggestions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    params = asdict(engine_config_from_params(base_params))
    accepted: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []

    for suggestion in suggestions:
        if not isinstance(suggestion, Mapping):
            ignored.append({"name": None, "reason": "suggestion is not an object"})
            continue

        name = str(suggestion.get("name") or "")
        spec = PARAM_SPECS.get(name)
        if spec is None:
            ignored.append({"name": name, "reason": "parameter is not bot-side auto-tunable"})
            continue

        requested = suggestion.get("suggested")
        applied, clamped = spec.clamp(requested)
        if applied is None:
            ignored.append({"name": name, "reason": "suggested value is not numeric"})
            continue

        current = params.get(name)
        params[name] = applied
        accepted.append(
            {
                "name": name,
                "current": current,
                "requested": requested,
                "applied": applied,
                "clamped": clamped,
                "reason": str(suggestion.get("reason") or ""),
            }
        )

    return {"params": params, "accepted": accepted, "ignored": ignored}


def apply_evaluation_suggestions(evaluation_id: int, *, policy: str | None = None) -> dict[str, Any] | None:
    try:
        return _with_connection(lambda conn: _apply_evaluation_suggestions(conn, evaluation_id, policy=policy))
    except Exception as exc:
        logger.warning("DB bot param suggestion apply failed: %s", exc)
        return None


def activate_bot_params_version(version: int) -> dict[str, Any] | None:
    try:
        return _with_connection(lambda conn: _activate_version(conn, version))
    except Exception as exc:
        logger.warning("DB bot param activation failed: %s", exc)
        return None


def review_active_param_effects(current_metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        return _with_connection(lambda conn: _review_active_param_effects(conn, current_metrics)) or {
            "status": "unavailable",
            "rollback_recommended": False,
            "reasons": ["database unavailable"],
        }
    except Exception as exc:
        logger.warning("DB bot param review failed: %s", exc)
        return {"status": "error", "rollback_recommended": False, "reasons": [str(exc)]}


def _apply_evaluation_suggestions(conn: Any, evaluation_id: int, *, policy: str | None) -> dict[str, Any] | None:
    resolved_policy = _apply_policy(policy)
    activate = resolved_policy == "auto"
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT param_suggestions
            FROM evaluations
            WHERE id = %s
            """,
            (int(evaluation_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        suggestions = row[0] if isinstance(row[0], list) else []
        base_params = _read_active_params_from_cursor(cursor)
        prepared = prepare_param_update(base_params, suggestions)
        if not prepared["accepted"]:
            return {
                "evaluation_id": int(evaluation_id),
                "version": None,
                "active": False,
                "policy": resolved_policy,
                "accepted": [],
                "ignored": prepared["ignored"],
            }

        cursor.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM bot_params")
        version = int(cursor.fetchone()[0])

        if activate:
            cursor.execute("UPDATE bot_params SET active = false WHERE active = true")

        cursor.execute(
            """
            INSERT INTO bot_params (version, params, active)
            VALUES (%s, CAST(%s AS jsonb), %s)
            RETURNING id
            """,
            (version, _jsonb(prepared["params"]), activate),
        )
        bot_param_id = int(cursor.fetchone()[0])
        cursor.execute(
            """
            UPDATE evaluations
            SET applied = true
            WHERE id = %s
            """,
            (int(evaluation_id),),
        )

    return {
        "evaluation_id": int(evaluation_id),
        "bot_param_id": bot_param_id,
        "version": version,
        "active": activate,
        "policy": resolved_policy,
        "accepted": prepared["accepted"],
        "ignored": prepared["ignored"],
    }


def _activate_version(conn: Any, version: int) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, version FROM bot_params WHERE version = %s", (int(version),))
        row = cursor.fetchone()
        if row is None:
            return None
        cursor.execute("UPDATE bot_params SET active = false WHERE active = true")
        cursor.execute("UPDATE bot_params SET active = true WHERE version = %s", (int(version),))
    return {"bot_param_id": int(row[0]), "version": int(row[1]), "active": True}


def _review_active_param_effects(conn: Any, current_metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT version, created_at
            FROM bot_params
            WHERE active = true
            ORDER BY version DESC
            LIMIT 1
            """
        )
        active = cursor.fetchone()
        if active is None:
            return {"status": "no_active_params", "rollback_recommended": False, "reasons": ["no active bot_params version"]}

        active_version = int(active[0])
        cursor.execute(
            """
            SELECT id, ts, metrics
            FROM learning_runs
            WHERE apply_result->>'version' = %s
              AND apply_result->>'active' = 'true'
            ORDER BY ts ASC
            LIMIT 1
            """,
            (str(active_version),),
        )
        baseline = cursor.fetchone()
        if baseline is None:
            return {
                "status": "no_activation_baseline",
                "active_version": active_version,
                "rollback_recommended": False,
                "reasons": ["active version has no recorded activation run"],
            }

        if current_metrics is None:
            cursor.execute(
                """
                SELECT id, ts, metrics
                FROM learning_runs
                WHERE metrics IS NOT NULL
                ORDER BY ts DESC
                LIMIT 1
                """
            )
            latest = cursor.fetchone()
            latest_metrics = latest[2] if latest is not None and isinstance(latest[2], Mapping) else {}
            latest_run_id = int(latest[0]) if latest is not None else None
            latest_ts = _iso(latest[1]) if latest is not None else None
        else:
            latest_metrics = dict(current_metrics)
            latest_run_id = None
            latest_ts = None

        cursor.execute(
            """
            SELECT version
            FROM bot_params
            WHERE version < %s
            ORDER BY version DESC
            LIMIT 1
            """,
            (active_version,),
        )
        previous = cursor.fetchone()

    return _param_effect_review(
        active_version=active_version,
        baseline_run_id=int(baseline[0]),
        baseline_ts=_iso(baseline[1]),
        baseline_metrics=baseline[2] if isinstance(baseline[2], Mapping) else {},
        latest_run_id=latest_run_id,
        latest_ts=latest_ts,
        latest_metrics=latest_metrics,
        rollback_target_version=int(previous[0]) if previous is not None else None,
    )


def _param_effect_review(
    *,
    active_version: int,
    baseline_run_id: int,
    baseline_ts: str | None,
    baseline_metrics: Mapping[str, Any],
    latest_run_id: int | None,
    latest_ts: str | None,
    latest_metrics: Mapping[str, Any],
    rollback_target_version: int | None,
) -> dict[str, Any]:
    deltas = {
        "marked_pnl_latest": _delta(latest_metrics, baseline_metrics, "marked_pnl_latest"),
        "realized_pnl_total": _delta(latest_metrics, baseline_metrics, "realized_pnl_total"),
        "win_rate": _delta(latest_metrics, baseline_metrics, "win_rate"),
        "max_drawdown_pnl": _delta(latest_metrics, baseline_metrics, "max_drawdown_pnl"),
    }
    closed = _metric_int(latest_metrics, "closed_trade_result_count")
    min_closed = _env_int("LEARNING_REVIEW_MIN_CLOSED_TRADES", 20)
    reasons: list[str] = []

    if closed < min_closed:
        return {
            "status": "insufficient_trade_data",
            "active_version": active_version,
            "rollback_target_version": rollback_target_version,
            "baseline_run_id": baseline_run_id,
            "baseline_ts": baseline_ts,
            "latest_run_id": latest_run_id,
            "latest_ts": latest_ts,
            "closed_trade_result_count": closed,
            "min_closed_trade_result_count": min_closed,
            "deltas": deltas,
            "rollback_recommended": False,
            "reasons": [f"closed trades {closed} below review minimum {min_closed}"],
        }

    marked_drop = _env_float("LEARNING_ROLLBACK_MARKED_PNL_DROP", 5.0)
    realized_drop = _env_float("LEARNING_ROLLBACK_REALIZED_PNL_DROP", 2.0)
    win_rate_drop = _env_float("LEARNING_ROLLBACK_WIN_RATE_DROP", 0.15)

    if _at_or_below(deltas["marked_pnl_latest"], -abs(marked_drop)):
        reasons.append(f"marked PnL worsened by {deltas['marked_pnl_latest']:.4f}")
    if _at_or_below(deltas["realized_pnl_total"], -abs(realized_drop)):
        reasons.append(f"realized PnL worsened by {deltas['realized_pnl_total']:.4f}")
    if _at_or_below(deltas["win_rate"], -abs(win_rate_drop)):
        reasons.append(f"win rate worsened by {deltas['win_rate']:.4f}")

    rollback_recommended = bool(reasons) and rollback_target_version is not None
    if not reasons:
        reasons.append("active params have not breached rollback thresholds")
    elif rollback_target_version is None:
        reasons.append("no earlier bot_params version is available")

    return {
        "status": "reviewed",
        "active_version": active_version,
        "rollback_target_version": rollback_target_version,
        "baseline_run_id": baseline_run_id,
        "baseline_ts": baseline_ts,
        "latest_run_id": latest_run_id,
        "latest_ts": latest_ts,
        "closed_trade_result_count": closed,
        "min_closed_trade_result_count": min_closed,
        "deltas": deltas,
        "rollback_recommended": rollback_recommended,
        "reasons": reasons,
    }


def _read_active_params(conn: Any) -> dict[str, Any]:
    with conn.cursor() as cursor:
        return _read_active_params_from_cursor(cursor)


def _read_active_params_from_cursor(cursor: Any) -> dict[str, Any]:
    cursor.execute(
        """
        SELECT params
        FROM bot_params
        WHERE active = true
        ORDER BY version DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone()
    if row is None or not isinstance(row[0], Mapping):
        return {}
    return {str(key): value for key, value in row[0].items() if str(key) in _ENGINE_CONFIG_FIELDS}


def _apply_policy(policy: str | None) -> str:
    value = (policy or os.environ.get("LEARNING_PARAM_APPLY_POLICY") or "approve").lower()
    return "auto" if value == "auto" else "approve"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _delta(current: Mapping[str, Any], baseline: Mapping[str, Any], key: str) -> float | None:
    current_value = _metric_float(current, key)
    baseline_value = _metric_float(baseline, key)
    if current_value is None or baseline_value is None:
        return None
    return round(current_value - baseline_value, 8)


def _metric_float(metrics: Mapping[str, Any], key: str) -> float | None:
    value = _numeric(metrics.get(key)) if isinstance(metrics, Mapping) else None
    return float(value) if value is not None else None


def _metric_int(metrics: Mapping[str, Any], key: str) -> int:
    value = _metric_float(metrics, key)
    return int(value) if value is not None else 0


def _at_or_below(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _coerce_like(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return None
    if isinstance(default, int):
        parsed = _numeric(value)
        return int(round(parsed)) if parsed is not None else None
    if isinstance(default, float):
        parsed = _numeric(value)
        return float(parsed) if parsed is not None else None
    return value


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed
