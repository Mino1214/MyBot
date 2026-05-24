from __future__ import annotations

import logging
from typing import Any

from rebalancing.recording import _with_connection


logger = logging.getLogger(__name__)


def learning_status_payload() -> dict[str, Any]:
    try:
        return _with_connection(_read_learning_status) or _empty_payload()
    except Exception as exc:
        logger.warning("DB learning status load failed: %s", exc)
        return _empty_payload(error=str(exc))


def _read_learning_status(conn: Any) -> dict[str, Any]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                (SELECT count(*) FROM learning_runs) AS run_count,
                (SELECT count(*) FROM evaluations) AS evaluation_count,
                (SELECT count(*) FROM bot_params) AS param_version_count,
                (SELECT count(*) FROM trade_results) AS trade_result_count,
                (SELECT current_stage FROM learning_stage WHERE id = true)
                    AS current_stage
            """
        )
        counts = cursor.fetchone()
        cursor.execute(
            """
            SELECT id, ts, trigger, window_size, mode, status, evaluation_id,
                   stage_before, stage_after, promoted, apply_result, metrics
            FROM learning_runs
            ORDER BY ts DESC
            LIMIT 8
            """
        )
        runs = cursor.fetchall()
        cursor.execute(
            """
            SELECT id, ts, window_size, summary, applied
            FROM evaluations
            ORDER BY ts DESC
            LIMIT 5
            """
        )
        evaluations = cursor.fetchall()
        cursor.execute(
            """
            SELECT version, active, created_at, params
            FROM bot_params
            ORDER BY version DESC
            LIMIT 5
            """
        )
        params = cursor.fetchall()

    latest_run = _run_row(runs[0]) if runs else None
    latest_evaluation = _evaluation_row(evaluations[0]) if evaluations else None
    active_params = next(
        (_param_row(row) for row in params if bool(row[1])),
        None,
    )

    return {
        "stage": str(counts[4] or "BABY") if counts else "BABY",
        "run_count": int(counts[0] or 0) if counts else 0,
        "evaluation_count": int(counts[1] or 0) if counts else 0,
        "param_version_count": int(counts[2] or 0) if counts else 0,
        "trade_result_count": int(counts[3] or 0) if counts else 0,
        "latest_run": latest_run,
        "latest_evaluation": latest_evaluation,
        "active_params": active_params,
        "runs": [_run_row(row) for row in runs],
        "evaluations": [_evaluation_row(row) for row in evaluations],
        "param_versions": [_param_row(row) for row in params],
    }


def _run_row(row: tuple[Any, ...]) -> dict[str, Any]:
    apply_result = row[10] if len(row) > 10 and isinstance(row[10], dict) else {}
    metrics = row[11] if len(row) > 11 and isinstance(row[11], dict) else {}
    return {
        "id": int(row[0]),
        "ts": row[1].isoformat()
        if hasattr(row[1], "isoformat")
        else str(row[1]),
        "trigger": str(row[2] or ""),
        "window_size": int(row[3] or 0),
        "mode": str(row[4] or ""),
        "status": str(row[5] or ""),
        "evaluation_id": int(row[6]) if row[6] is not None else None,
        "stage_before": str(row[7] or "BABY"),
        "stage_after": str(row[8] or "BABY"),
        "promoted": bool(row[9]),
        "apply_result": _apply_result_summary(apply_result),
        "metrics": _metrics_summary(metrics),
    }


def _evaluation_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": int(row[0]),
        "ts": row[1].isoformat()
        if hasattr(row[1], "isoformat")
        else str(row[1]),
        "window_size": int(row[2] or 0),
        "summary": str(row[3] or ""),
        "applied": bool(row[4]),
    }


def _param_row(row: tuple[Any, ...]) -> dict[str, Any]:
    params = row[3] if isinstance(row[3], dict) else {}
    return {
        "version": int(row[0]),
        "active": bool(row[1]),
        "created_at": row[2].isoformat()
        if hasattr(row[2], "isoformat")
        else str(row[2]),
        "range_target_leverage": params.get("range_target_leverage"),
        "confirmation_candles": params.get("confirmation_candles"),
        "min_neutral_hours": params.get("min_neutral_hours"),
    }


def _apply_result_summary(value: dict[str, Any]) -> dict[str, Any]:
    accepted = value.get("accepted") if isinstance(value.get("accepted"), list) else []
    return {
        "version": int(value["version"]) if value.get("version") is not None else None,
        "active": bool(value.get("active")),
        "policy": str(value.get("policy") or ""),
        "accepted_count": len(accepted),
        "accepted_names": [str(item.get("name")) for item in accepted if isinstance(item, dict) and item.get("name")],
    }


def _metrics_summary(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "decision_count",
        "trade_result_count",
        "closed_trade_result_count",
        "open_feedback_count",
        "marked_pnl_latest",
        "marked_pnl_change",
        "realized_pnl_latest",
        "unrealized_pnl_latest",
        "open_position_count_latest",
        "open_position_count_max",
        "realized_pnl_total",
        "win_rate",
        "learning_review",
        "rollback_result",
    )
    return {key: value.get(key) for key in keys if key in value}


def _empty_payload(*, error: str | None = None) -> dict[str, Any]:
    payload = {
        "stage": "BABY",
        "run_count": 0,
        "evaluation_count": 0,
        "param_version_count": 0,
        "trade_result_count": 0,
        "latest_run": None,
        "latest_evaluation": None,
        "active_params": None,
        "runs": [],
        "evaluations": [],
        "param_versions": [],
    }
    if error:
        payload["error"] = error
    return payload
