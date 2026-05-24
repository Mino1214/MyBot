from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from decimal import Decimal
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rebalancing.models import EngineConfig
from rebalancing.recording import _jsonb, _to_json, _with_connection


logger = logging.getLogger(__name__)

DIAGNOSIS_SYSTEM_PROMPT = (
    "You are a risk-first trading bot diagnostician. "
    "Return only one valid JSON object. Do not include markdown, code fences, or commentary."
)

DIAGNOSIS_SCHEMA_HINT = {
    "summary": "이번 구간 진단 요약",
    "findings": [{"issue": "...", "evidence": "...", "severity": "low|med|high"}],
    "param_suggestions": [
        {"name": "<봇영역 파라미터명>", "current": 0, "suggested": 0, "reason": "..."}
    ],
    "pine_suggestions": [{"target": "...", "suggestion": "...", "reason": "..."}],
    "stage_eval": {"current_stage": "BABY|JUNIOR|PRO", "ready_for_promotion": False, "reason": "..."},
}

DIAGNOSIS_TOOL = {
    "name": "record_diagnosis",
    "description": "Record one structured diagnosis for the rebalancing bot.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {"type": "string"},
                        "evidence": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "med", "high"]},
                    },
                    "required": ["issue", "evidence", "severity"],
                    "additionalProperties": False,
                },
            },
            "param_suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "current": {"type": ["number", "string", "boolean", "null"]},
                        "suggested": {"type": ["number", "string", "boolean", "null"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "current", "suggested", "reason"],
                    "additionalProperties": False,
                },
            },
            "pine_suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "suggestion": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["target", "suggestion", "reason"],
                    "additionalProperties": False,
                },
            },
            "stage_eval": {
                "type": "object",
                "properties": {
                    "current_stage": {"type": "string", "enum": ["BABY", "JUNIOR", "PRO"]},
                    "ready_for_promotion": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["current_stage", "ready_for_promotion", "reason"],
                "additionalProperties": False,
            },
        },
        "required": ["summary", "findings", "param_suggestions", "pine_suggestions", "stage_eval"],
        "additionalProperties": False,
    },
}


def build_diagnosis_prompt(window: int = 100, *, mode: str | None = None) -> str:
    records = load_recent_records(window=window, mode=mode)
    metrics = summarize_records(records)
    recent_limit = _env_int("LEARNING_PROMPT_RECENT_LIMIT", 20)
    payload = {
        "window": window,
        "mode_filter": mode,
        "metrics": metrics,
        "learning_review": learning_review(metrics),
        "current_bot_params": current_bot_params(),
        "recent_decisions": [_compact_record(record) for record in records[-recent_limit:]],
        "response_schema": DIAGNOSIS_SCHEMA_HINT,
        "rules": [
            "TradingView/Pine Script parameters must not be auto-applied.",
            "Only suggest bot-side EngineConfig parameters in param_suggestions.",
            "Prefer smaller, reversible changes and include evidence from the provided data.",
            "Use learning_review to judge whether the previous active parameter version helped or should be rolled back.",
            "If data is insufficient, say so and avoid aggressive parameter suggestions.",
        ],
    }
    data = json.dumps(_to_json(payload), ensure_ascii=False, indent=2, sort_keys=True)
    return (
        "Analyze the following rebalancing bot decision window and produce the requested JSON.\n"
        "The template is fixed; only the JSON data below changes between runs.\n\n"
        f"{data}"
    )


def call_diagnosis(prompt: str) -> str | None:
    api_key = _anthropic_api_key()
    if not api_key:
        logger.warning("Claude diagnosis skipped: ANTHROPIC_API_KEY is not set")
        return None

    model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("CLAUDE_MODEL") or "claude-sonnet-4-6"
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    timeout = _env_float("ANTHROPIC_TIMEOUT_SECONDS", 60.0)
    max_tokens = _env_int("ANTHROPIC_MAX_TOKENS", 2048)
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": DIAGNOSIS_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [DIAGNOSIS_TOOL],
        "tool_choice": {"type": "tool", "name": "record_diagnosis"},
    }
    request = Request(
        f"{base_url}/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            "content-type": "application/json",
            "accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        logger.warning("Claude diagnosis API failed: HTTP %s %s", exc.code, _safe_error_body(exc))
        return None
    except (OSError, URLError, json.JSONDecodeError) as exc:
        logger.warning("Claude diagnosis API failed: %s", exc)
        return None

    return _diagnosis_response_text(payload)


def parse_diagnosis(raw: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        data = dict(raw)
    else:
        text = _strip_code_fence(raw.strip())
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            extracted = _extract_json_object(text)
            if extracted is None:
                logger.warning("Claude diagnosis JSON parse failed")
                return None
            try:
                data = json.loads(extracted)
            except json.JSONDecodeError as exc:
                logger.warning("Claude diagnosis JSON parse failed: %s", exc)
                return None

    if not isinstance(data, dict):
        logger.warning("Claude diagnosis JSON must be an object")
        return None

    normalized = {
        "summary": str(data.get("summary") or ""),
        "findings": _list_or_empty(data.get("findings")),
        "param_suggestions": _list_or_empty(data.get("param_suggestions")),
        "pine_suggestions": _list_or_empty(data.get("pine_suggestions")),
        "stage_eval": data.get("stage_eval") if isinstance(data.get("stage_eval"), Mapping) else {},
    }
    return normalized


def save_evaluation(
    diagnosis: Mapping[str, Any],
    *,
    window: int,
    raw: str | Mapping[str, Any] | None = None,
) -> int | None:
    try:
        return _insert_evaluation(diagnosis, window=window, raw=raw)
    except Exception as exc:
        logger.warning("DB evaluation recording failed: %s", exc)
        return None


def run_diagnosis(*, window: int = 100, mode: str | None = None) -> dict[str, Any] | None:
    prompt = build_diagnosis_prompt(window=window, mode=mode)
    raw = call_diagnosis(prompt)
    diagnosis = parse_diagnosis(raw)
    if diagnosis is None:
        return None
    evaluation_id = save_evaluation(diagnosis, window=window, raw=raw)
    result = dict(diagnosis)
    result["evaluation_id"] = evaluation_id
    return result


def load_recent_records(*, window: int = 100, mode: str | None = None) -> tuple[dict[str, Any], ...]:
    window = max(1, int(window))
    mode = mode if mode in {"live", "paper"} else None

    def read(conn: Any) -> tuple[dict[str, Any], ...]:
        with conn.cursor() as cursor:
            if mode:
                cursor.execute(
                    """
                    SELECT
                        d.id, d.ts, d.mode, d.regime, d.raw_regime, d.market_bias,
                        d.regime_score, d.should_rebalance, d.risk_action, d.reasons, d.next_state,
                        s.account, s.positions, s.candidates, s.btc, s.market_internals,
                        COALESCE(
                            jsonb_agg(
                                jsonb_build_object(
                                    'symbol', tr.symbol,
                                    'realized_pnl', tr.realized_pnl,
                                    'opened_at', tr.opened_at,
                                    'closed_at', tr.closed_at,
                                    'status', tr.status
                                )
                            ) FILTER (WHERE tr.id IS NOT NULL),
                            '[]'::jsonb
                        ) AS trade_results
                    FROM (
                        SELECT *
                        FROM decisions
                        WHERE mode = %s
                        ORDER BY ts DESC
                        LIMIT %s
                    ) d
                    LEFT JOIN market_snapshots s ON s.decision_id = d.id
                    LEFT JOIN trade_results tr ON tr.decision_id = d.id
                    GROUP BY
                        d.id, d.ts, d.mode, d.regime, d.raw_regime, d.market_bias,
                        d.regime_score, d.should_rebalance, d.risk_action, d.reasons, d.next_state,
                        s.account, s.positions, s.candidates, s.btc, s.market_internals
                    ORDER BY d.ts ASC
                    """,
                    (mode, window),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        d.id, d.ts, d.mode, d.regime, d.raw_regime, d.market_bias,
                        d.regime_score, d.should_rebalance, d.risk_action, d.reasons, d.next_state,
                        s.account, s.positions, s.candidates, s.btc, s.market_internals,
                        COALESCE(
                            jsonb_agg(
                                jsonb_build_object(
                                    'symbol', tr.symbol,
                                    'realized_pnl', tr.realized_pnl,
                                    'opened_at', tr.opened_at,
                                    'closed_at', tr.closed_at,
                                    'status', tr.status
                                )
                            ) FILTER (WHERE tr.id IS NOT NULL),
                            '[]'::jsonb
                        ) AS trade_results
                    FROM (
                        SELECT *
                        FROM decisions
                        ORDER BY ts DESC
                        LIMIT %s
                    ) d
                    LEFT JOIN market_snapshots s ON s.decision_id = d.id
                    LEFT JOIN trade_results tr ON tr.decision_id = d.id
                    GROUP BY
                        d.id, d.ts, d.mode, d.regime, d.raw_regime, d.market_bias,
                        d.regime_score, d.should_rebalance, d.risk_action, d.reasons, d.next_state,
                        s.account, s.positions, s.candidates, s.btc, s.market_internals
                    ORDER BY d.ts ASC
                    """,
                    (window,),
                )
            rows = cursor.fetchall()
        return tuple(_row_to_record(row) for row in rows)

    try:
        return _with_connection(read) or tuple()
    except Exception as exc:
        logger.warning("DB diagnosis window load failed: %s", exc)
        return tuple()


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trade_results = [
        result
        for record in records
        for result in _list_or_empty(record.get("trade_results"))
        if _numeric(result.get("realized_pnl")) is not None
    ]
    closed_results = [
        result
        for result in trade_results
        if _is_closed_trade_result(result)
    ]
    pnl_values = [_numeric(result.get("realized_pnl")) or 0.0 for result in closed_results]
    wins = sum(1 for value in pnl_values if value > 0)
    losses = sum(1 for value in pnl_values if value < 0)
    marked_pnl_values = [
        value
        for value in (_account_numeric(record, "total_pnl") for record in records)
        if value is not None
    ]
    position_counts = [_position_count(record) for record in records]

    return {
        "decision_count": len(records),
        "mode_counts": _count_by(records, "mode"),
        "regime_counts": _count_by(records, "regime"),
        "market_bias_counts": _count_by(records, "market_bias"),
        "risk_action_counts": _count_by(records, "risk_action"),
        "should_rebalance_true_count": sum(1 for record in records if bool(record.get("should_rebalance"))),
        "trade_result_count": len(trade_results),
        "closed_trade_result_count": len(closed_results),
        "realized_pnl_total": round(sum(pnl_values), 8),
        "win_count": wins,
        "loss_count": losses,
        "win_rate": round(wins / len(pnl_values), 6) if pnl_values else None,
        "max_drawdown_pnl": round(_max_drawdown(pnl_values), 8),
        "open_feedback_count": len(marked_pnl_values),
        "marked_pnl_latest": round(marked_pnl_values[-1], 8) if marked_pnl_values else None,
        "marked_pnl_change": round(marked_pnl_values[-1] - marked_pnl_values[0], 8)
        if len(marked_pnl_values) >= 2
        else None,
        "marked_pnl_min": round(min(marked_pnl_values), 8) if marked_pnl_values else None,
        "marked_pnl_max": round(max(marked_pnl_values), 8) if marked_pnl_values else None,
        "realized_pnl_latest": _latest_account_metric(records, "realized_pnl"),
        "unrealized_pnl_latest": _latest_account_metric(records, "unrealized_pnl"),
        "open_position_count_latest": position_counts[-1] if position_counts else 0,
        "open_position_count_max": max(position_counts) if position_counts else 0,
        "regime_performance": _performance_by(records, "regime"),
        "market_bias_performance": _performance_by(records, "market_bias"),
        "btc_dominance_performance": _btc_dominance_performance(records),
        "should_rebalance_performance": _should_rebalance_performance(records),
    }


def current_bot_params() -> dict[str, Any]:
    try:
        from rebalancing.learning.params import active_engine_config

        return asdict(active_engine_config())
    except Exception as exc:
        logger.warning("Active bot params unavailable for diagnosis prompt: %s", exc)
        return asdict(EngineConfig())


def learning_review(metrics: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from rebalancing.learning.params import review_active_param_effects

        return review_active_param_effects(metrics)
    except Exception as exc:
        logger.warning("Learning review unavailable for diagnosis prompt: %s", exc)
        return {"status": "unavailable", "rollback_recommended": False, "reasons": [str(exc)]}


def _insert_evaluation(
    diagnosis: Mapping[str, Any],
    *,
    window: int,
    raw: str | Mapping[str, Any] | None,
) -> int | None:
    def write(conn: Any) -> int:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO evaluations (
                    window_size, summary, findings, param_suggestions,
                    pine_suggestions, stage_eval, raw, applied
                )
                VALUES (
                    %s, %s, CAST(%s AS jsonb), CAST(%s AS jsonb),
                    CAST(%s AS jsonb), CAST(%s AS jsonb), CAST(%s AS jsonb), false
                )
                RETURNING id
                """,
                (
                    int(window),
                    str(diagnosis.get("summary") or ""),
                    _jsonb(_list_or_empty(diagnosis.get("findings"))),
                    _jsonb(_list_or_empty(diagnosis.get("param_suggestions"))),
                    _jsonb(_list_or_empty(diagnosis.get("pine_suggestions"))),
                    _jsonb(diagnosis.get("stage_eval") if isinstance(diagnosis.get("stage_eval"), Mapping) else {}),
                    _jsonb(_raw_payload(raw, diagnosis)),
                ),
            )
            return int(cursor.fetchone()[0])

    return _with_connection(write)


def _row_to_record(row: Sequence[Any]) -> dict[str, Any]:
    keys = (
        "id",
        "ts",
        "mode",
        "regime",
        "raw_regime",
        "market_bias",
        "regime_score",
        "should_rebalance",
        "risk_action",
        "reasons",
        "next_state",
        "account",
        "positions",
        "candidates",
        "btc",
        "market_internals",
        "trade_results",
    )
    return {key: _decode_json_value(value) for key, value in zip(keys, row)}


def _compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    account = record.get("account") if isinstance(record.get("account"), Mapping) else {}
    positions = _list_or_empty(record.get("positions"))
    candidates = _list_or_empty(record.get("candidates"))
    trade_results = _list_or_empty(record.get("trade_results"))
    return {
        "id": record.get("id"),
        "ts": _to_json(record.get("ts")),
        "mode": record.get("mode"),
        "regime": record.get("regime"),
        "market_bias": record.get("market_bias"),
        "regime_score": record.get("regime_score"),
        "should_rebalance": record.get("should_rebalance"),
        "risk_action": record.get("risk_action"),
        "reasons": record.get("reasons"),
        "equity": account.get("equity"),
        "realized_pnl": account.get("realized_pnl"),
        "unrealized_pnl": account.get("unrealized_pnl"),
        "total_pnl": account.get("total_pnl"),
        "leverage": account.get("leverage"),
        "position_count": len(positions),
        "candidate_count": len(candidates),
        "market_internals": _compact_market_internals(record.get("market_internals")),
        "trade_results": trade_results,
    }


def _compact_market_internals(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "stable_dominance_pct",
        "top10_dominance_total_pct",
        "top10_dominance_total2_pct",
        "volume_breadth_pct",
        "advance_decline_ratio",
        "risk_label",
    )
    return {key: value.get(key) for key in keys if key in value}


def _performance_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for record in records:
        label = str(record.get(key) or "UNKNOWN")
        pnl = _record_realized_pnl(record)
        buckets.setdefault(label, []).append(pnl)
    return {
        label: _performance(values)
        for label, values in sorted(buckets.items())
    }


def _btc_dominance_performance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for record in records:
        label = _btc_dominance_bucket(record)
        buckets.setdefault(label, []).append(_record_realized_pnl(record))
    return {
        label: _performance(values)
        for label, values in sorted(buckets.items())
    }


def _should_rebalance_performance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    true_values = [_record_realized_pnl(record) for record in records if bool(record.get("should_rebalance"))]
    false_values = [_record_realized_pnl(record) for record in records if not bool(record.get("should_rebalance"))]
    return {
        "true": _performance(true_values),
        "false": _performance(false_values),
    }


def _performance(values: Sequence[float]) -> dict[str, Any]:
    wins = sum(1 for value in values if value > 0)
    losses = sum(1 for value in values if value < 0)
    return {
        "count": len(values),
        "pnl": round(sum(values), 8),
        "avg_pnl": round(sum(values) / len(values), 8) if values else 0.0,
        "win_rate": round(wins / len(values), 6) if values else None,
        "wins": wins,
        "losses": losses,
    }


def _record_realized_pnl(record: Mapping[str, Any]) -> float:
    total = 0.0
    for result in _list_or_empty(record.get("trade_results")):
        if not _is_closed_trade_result(result):
            continue
        total += _numeric(result.get("realized_pnl")) or 0.0
    return total


def _latest_account_metric(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    for record in reversed(records):
        value = _account_numeric(record, key)
        if value is not None:
            return round(value, 8)
    return None


def _account_numeric(record: Mapping[str, Any], key: str) -> float | None:
    account = record.get("account") if isinstance(record.get("account"), Mapping) else {}
    return _numeric(account.get(key))


def _position_count(record: Mapping[str, Any]) -> int:
    positions = record.get("positions")
    return len(positions) if isinstance(positions, list) else 0


def _btc_dominance_bucket(record: Mapping[str, Any]) -> str:
    internals = record.get("market_internals") if isinstance(record.get("market_internals"), Mapping) else {}
    btc = record.get("btc") if isinstance(record.get("btc"), Mapping) else {}
    value = (
        _numeric(internals.get("btc_dominance_pct"))
        or _numeric(internals.get("btc_dominance"))
        or _numeric(internals.get("btcd"))
        or _numeric(btc.get("dominance_pct"))
        or _numeric(btc.get("btc_dominance_pct"))
    )
    if value is None:
        return "unknown"
    if value < 45:
        return "<45"
    if value < 50:
        return "45-50"
    if value < 55:
        return "50-55"
    if value < 60:
        return "55-60"
    return ">=60"


def _max_drawdown(pnl_values: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for pnl in pnl_values:
        equity += pnl
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def _count_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.get(key) or "UNKNOWN")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _is_closed_trade_result(result: Mapping[str, Any]) -> bool:
    if _numeric(result.get("realized_pnl")) is None:
        return False
    status = str(result.get("status") or "").lower()
    return not status or status in {"closed", "settled", "done", "realized"}


def _raw_payload(raw: str | Mapping[str, Any] | None, diagnosis: Mapping[str, Any]) -> Any:
    if raw is None:
        return diagnosis
    if isinstance(raw, Mapping):
        return raw
    return {"text": raw}


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline == -1:
        return text.strip("`").strip()
    body = text[first_newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in {"{", "["}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _list_or_empty(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


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


def _safe_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")[:500]
    except Exception:
        return ""


def _diagnosis_response_text(payload: Mapping[str, Any]) -> str | None:
    for block in payload.get("content", []):
        if (
            isinstance(block, Mapping)
            and block.get("type") == "tool_use"
            and block.get("name") == "record_diagnosis"
            and isinstance(block.get("input"), Mapping)
        ):
            return json.dumps(block["input"], ensure_ascii=False, separators=(",", ":"))

    text_blocks = [
        str(block.get("text", ""))
        for block in payload.get("content", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    if not text_blocks:
        logger.warning("Claude diagnosis API returned no diagnosis content")
        return None
    return "\n".join(text_blocks).strip()


def _anthropic_api_key() -> str | None:
    value = os.environ.get("ANTHROPIC_API_KEY")
    if value:
        return value.strip()

    path = os.environ.get("ANTHROPIC_API_KEY_FILE")
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError as exc:
        logger.warning("Claude diagnosis API key file unavailable: %s", exc)
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default
