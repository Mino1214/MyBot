from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

from rebalancing.recording import _jsonb, _with_connection

from .alerts import notify_learning_result
from .diagnosis import load_recent_records, run_diagnosis, summarize_records
from .params import activate_bot_params_version, apply_evaluation_suggestions, review_active_param_effects


logger = logging.getLogger(__name__)

STAGES = ("BABY", "JUNIOR", "PRO")


@dataclass(frozen=True)
class PromotionRule:
    from_stage: str
    to_stage: str
    min_closed_trades: int
    min_win_rate: float
    min_total_pnl: float
    max_drawdown_pnl: float


PROMOTION_RULES = {
    "BABY": PromotionRule("BABY", "JUNIOR", min_closed_trades=20, min_win_rate=0.55, min_total_pnl=0.0, max_drawdown_pnl=-100.0),
    "JUNIOR": PromotionRule("JUNIOR", "PRO", min_closed_trades=50, min_win_rate=0.60, min_total_pnl=0.0, max_drawdown_pnl=-70.0),
}


def run_learning_cycle(
    *,
    window: int = 100,
    mode: str | None = "paper",
    trigger: str = "manual",
    apply_policy: str | None = None,
) -> dict[str, Any]:
    stage_before = current_stage()
    metrics = dict(summarize_records(load_recent_records(window=window, mode=mode)))
    review = review_active_param_effects(metrics) or {}
    if review:
        metrics["learning_review"] = review
    result: dict[str, Any] = {
        "trigger": trigger,
        "window_size": int(window),
        "mode": mode,
        "evaluation_id": None,
        "apply_result": {},
        "stage_before": stage_before,
        "stage_after": stage_before,
        "promoted": False,
        "status": "started",
        "error": None,
        "metrics": metrics,
    }

    try:
        if review.get("rollback_recommended") is True and _auto_rollback_enabled():
            target_version = review.get("rollback_target_version")
            rollback = (
                activate_bot_params_version(int(target_version))
                if target_version is not None
                else None
            )
            metrics["rollback_result"] = rollback or {}
            result["apply_result"] = {
                "policy": "rollback",
                "version": rollback.get("version") if isinstance(rollback, Mapping) else None,
                "active": bool(rollback.get("active")) if isinstance(rollback, Mapping) else False,
                "accepted": [],
                "review": review,
            }
            result["status"] = "rolled_back" if rollback else "rollback_failed"
            _record_learning_run(result)
            notify_learning_result(result)
            return result

        diagnosis = run_diagnosis(window=window, mode=mode)
        if diagnosis is None:
            result["status"] = "diagnosis_failed"
            _record_learning_run(result)
            notify_learning_result(result)
            return result

        evaluation_id = diagnosis.get("evaluation_id")
        result["evaluation_id"] = evaluation_id
        if evaluation_id is not None:
            apply_result = apply_evaluation_suggestions(int(evaluation_id), policy=apply_policy)
            result["apply_result"] = apply_result or {}

        stage_eval = diagnosis.get("stage_eval") if isinstance(diagnosis.get("stage_eval"), Mapping) else {}
        stage_after, promoted = maybe_promote_stage(stage_before, metrics, stage_eval)
        result["stage_after"] = stage_after
        result["promoted"] = promoted
        result["status"] = "ok"
        _record_learning_run(result)
        notify_learning_result(result)
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        logger.warning("Learning cycle failed: %s", exc)
        _record_learning_run(result)
        notify_learning_result(result)
        return result


def run_scheduler(
    *,
    window: int = 100,
    mode: str | None = "paper",
    interval_seconds: int | None = None,
) -> None:
    interval = interval_seconds or _env_int("LEARNING_INTERVAL_SECONDS", 7 * 24 * 60 * 60)
    while True:
        run_learning_cycle(window=window, mode=mode, trigger="scheduler")
        time.sleep(max(60, interval))


def maybe_promote_stage(
    current: str,
    metrics: Mapping[str, Any],
    stage_eval: Mapping[str, Any] | None,
) -> tuple[str, bool]:
    current = current if current in STAGES else "BABY"
    if current == "PRO":
        return current, False

    if not isinstance(stage_eval, Mapping) or stage_eval.get("ready_for_promotion") is not True:
        return current, False

    rule = PROMOTION_RULES.get(current)
    if rule is None:
        return current, False

    closed = int(metrics.get("closed_trade_result_count") or 0)
    win_rate = metrics.get("win_rate")
    total_pnl = float(metrics.get("realized_pnl_total") or 0.0)
    drawdown = float(metrics.get("max_drawdown_pnl") or 0.0)
    if win_rate is None:
        return current, False
    if closed < rule.min_closed_trades:
        return current, False
    if float(win_rate) < rule.min_win_rate:
        return current, False
    if total_pnl < rule.min_total_pnl:
        return current, False
    if drawdown < rule.max_drawdown_pnl:
        return current, False

    update_stage(rule.to_stage)
    return rule.to_stage, True


def current_stage() -> str:
    try:
        value = _with_connection(_read_stage)
        return value if value in STAGES else "BABY"
    except Exception as exc:
        logger.warning("DB learning stage load failed: %s", exc)
        return "BABY"


def update_stage(stage: str) -> None:
    if stage not in STAGES:
        return
    try:
        _with_connection(lambda conn: _write_stage(conn, stage))
    except Exception as exc:
        logger.warning("DB learning stage update failed: %s", exc)


def _record_learning_run(result: Mapping[str, Any]) -> None:
    try:
        _with_connection(lambda conn: _insert_learning_run(conn, result))
    except Exception as exc:
        logger.warning("DB learning run recording failed: %s", exc)


def _insert_learning_run(conn: Any, result: Mapping[str, Any]) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO learning_runs (
                trigger, window_size, mode, evaluation_id, apply_result,
                stage_before, stage_after, promoted, status, error, metrics
            )
            VALUES (
                %s, %s, %s, %s, CAST(%s AS jsonb),
                %s, %s, %s, %s, %s, CAST(%s AS jsonb)
            )
            """,
            (
                str(result.get("trigger") or "manual"),
                int(result.get("window_size") or 0),
                result.get("mode"),
                result.get("evaluation_id"),
                _jsonb(result.get("apply_result") or {}),
                str(result.get("stage_before") or "BABY"),
                str(result.get("stage_after") or "BABY"),
                bool(result.get("promoted")),
                str(result.get("status") or "unknown"),
                result.get("error"),
                _jsonb(result.get("metrics") or {}),
            ),
        )


def _read_stage(conn: Any) -> str:
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_stage FROM learning_stage WHERE id = true")
        row = cursor.fetchone()
    return str(row[0]) if row else "BABY"


def _write_stage(conn: Any, stage: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO learning_stage (id, current_stage, promoted_at, updated_at)
            VALUES (true, %s, now(), now())
            ON CONFLICT (id) DO UPDATE
            SET current_stage = EXCLUDED.current_stage,
                promoted_at = EXCLUDED.promoted_at,
                updated_at = now()
            """,
            (stage,),
        )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _auto_rollback_enabled() -> bool:
    return os.environ.get("LEARNING_AUTO_ROLLBACK_ENABLED", "true").lower() == "true"
