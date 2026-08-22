from __future__ import annotations

from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import binomtest, chi2, norm

from utils.statistical_analysis_constants import (
    METRIC_NAMES,
    METRIC_ORDER,
    STATISTICAL_REPORT_TYPE,
    StatisticalAnalysisArtifacts,
    StatisticalAnalysisConfig,
    StatisticalAnalysisContext,
    StrategyStatisticalInput,
)


def _metric_vector_from_confusion(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    total = float(cm.sum())
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)

    accuracy = float(tp.sum() / total) if total else 0.0
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    balanced_accuracy = float(np.mean(recall)) if len(recall) else 0.0
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    f1_macro = float(np.mean(f1)) if len(f1) else 0.0
    f1_weighted = float(np.average(f1, weights=support)) if support.sum() else 0.0
    return np.asarray([accuracy, balanced_accuracy, f1_macro, f1_weighted], dtype=float)


def _encode_inputs(
    context: StatisticalAnalysisContext,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Any]]:
    if not context.strategies:
        raise ValueError("Не завантажено жодного звіту стратегії.")
    y_true_raw = list(context.strategies[0].y_true)
    classes = sorted(set(y_true_raw), key=lambda value: str(value))
    class_to_index = {value: index for index, value in enumerate(classes)}

    try:
        y_true = np.asarray([class_to_index[value] for value in y_true_raw], dtype=np.int16)
    except (KeyError, TypeError) as exc:
        raise ValueError("Істинні класи повинні бути скалярними JSON-значеннями.") from exc

    model_records: List[Dict[str, Any]] = []
    encoded_predictions: List[np.ndarray] = []

    for strategy in context.strategies:
        for model_name, model_data in strategy.models.items():
            y_pred_raw = model_data.get("y_pred", [])
            unknown = sorted({value for value in y_pred_raw if value not in class_to_index}, key=lambda value: str(value))
            if unknown:
                raise ValueError(
                    f"Модель '{model_name}' зі стратегії '{strategy.strategy_label}' прогнозує "
                    f"невідомі класи: {unknown}."
                )
            y_pred = np.asarray([class_to_index[value] for value in y_pred_raw], dtype=np.int16)
            if len(y_pred) != len(y_true):
                raise ValueError(f"Довжина прогнозів моделі '{model_name}' не збігається з test set.")

            model_id = f"{strategy.strategy_type}::{model_name}"
            model_records.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "strategy_type": strategy.strategy_type,
                    "strategy_label": strategy.strategy_label,
                    "level": model_data.get("level", "—"),
                    "family": model_data.get("family", "—"),
                }
            )
            encoded_predictions.append(y_pred)

    if len(model_records) < 2:
        raise ValueError("Для статистичного порівняння потрібно щонайменше дві моделі.")

    return y_true, np.vstack(encoded_predictions), model_records, classes


def _confusion_from_encoded(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes)


def _stratified_bootstrap_score_cube(
    y_true: np.ndarray,
    predictions: np.ndarray,
    n_classes: int,
    config: StatisticalAnalysisConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.random_state)
    class_indices = [np.flatnonzero(y_true == class_index) for class_index in range(n_classes)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Стратифікований bootstrap неможливий: один з істинних класів не представлений у test set.")

    n_models = predictions.shape[0]
    score_cube = np.empty((config.n_bootstrap, n_models, len(METRIC_ORDER)), dtype=float)

    for bootstrap_index in range(config.n_bootstrap):
        sampled_parts = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ]
        sampled_idx = np.concatenate(sampled_parts)
        sampled_true = y_true[sampled_idx]

        for model_index in range(n_models):
            sampled_pred = predictions[model_index, sampled_idx]
            cm = _confusion_from_encoded(sampled_true, sampled_pred, n_classes)
            score_cube[bootstrap_index, model_index] = _metric_vector_from_confusion(cm)

    return score_cube


def _normal_approximation_accuracy_ci(
    accuracy: float,
    n_samples: int,
    confidence_level: float,
) -> Dict[str, float]:
    z_value = float(norm.ppf((1.0 + confidence_level) / 2.0))
    standard_error = float(np.sqrt(accuracy * (1.0 - accuracy) / n_samples))
    margin = z_value * standard_error
    return {
        "point": float(accuracy),
        "standard_error": standard_error,
        "z_value": z_value,
        "lower": float(max(0.0, accuracy - margin)),
        "upper": float(min(1.0, accuracy + margin)),
        "confidence_level": float(confidence_level),
        "method": "normal_approximation_for_proportion",
    }


def _bootstrap_interval(values: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _mcnemar_result(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, Any]:
    both_correct = int(np.sum(correct_a & correct_b))
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))
    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant < 25:
        method = "exact_two_sided_binomial"
        statistic = float(min(a_correct_b_wrong, a_wrong_b_correct))
        p_value = (
            float(binomtest(int(statistic), n=discordant, p=0.5, alternative="two-sided").pvalue)
            if discordant > 0
            else 1.0
        )
        continuity_correction = False
    else:
        method = "chi_square_with_continuity_correction"
        statistic = float((abs(a_correct_b_wrong - a_wrong_b_correct) - 1.0) ** 2 / discordant)
        p_value = float(chi2.sf(statistic, df=1))
        continuity_correction = True

    return {
        "table": [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]],
        "a_both_correct": both_correct,
        "b_a_correct_b_wrong": a_correct_b_wrong,
        "c_a_wrong_b_correct": a_wrong_b_correct,
        "d_both_wrong": both_wrong,
        "discordant": discordant,
        "method": method,
        "exact": discordant < 25,
        "continuity_correction": continuity_correction,
        "statistic": statistic,
        "p_value": p_value,
    }


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []

    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = (m - rank) * float(p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted.tolist()


def _strategy_source_record(strategy: StrategyStatisticalInput) -> Dict[str, Any]:
    return {
        "file_name": strategy.file_name,
        "strategy_type": strategy.strategy_type,
        "strategy_label": strategy.strategy_label,
        "target_col": strategy.target_col,
        "Q1": strategy.q1,
        "Q3": strategy.q3,
        "test_set_fingerprint": strategy.test_set_fingerprint,
    }


def run_statistical_analysis(
    context: StatisticalAnalysisContext,
    config: StatisticalAnalysisConfig,
) -> StatisticalAnalysisArtifacts:
    if not (0.0 < config.confidence_level < 1.0):
        raise ValueError("Рівень довіри повинен належати інтервалу (0; 1).")
    if config.n_bootstrap < 100:
        raise ValueError("Кількість bootstrap-повторень повинна бути не меншою за 100.")
    if not (0.0 < config.alpha < 1.0):
        raise ValueError("Рівень значущості alpha повинен належати інтервалу (0; 1).")

    y_true, predictions, model_records, classes = _encode_inputs(context)
    n_samples = int(len(y_true))
    n_classes = int(len(classes))
    n_models = int(predictions.shape[0])

    point_scores = np.empty((n_models, len(METRIC_ORDER)), dtype=float)
    for model_index in range(n_models):
        cm = _confusion_from_encoded(y_true, predictions[model_index], n_classes)
        point_scores[model_index] = _metric_vector_from_confusion(cm)

    bootstrap_scores = _stratified_bootstrap_score_cube(
        y_true=y_true,
        predictions=predictions,
        n_classes=n_classes,
        config=config,
    )

    model_rows: List[Dict[str, Any]] = []
    for model_index, record in enumerate(model_records):
        bootstrap_metrics: Dict[str, Dict[str, Any]] = {}
        point_metrics: Dict[str, float] = {}
        for metric_index, metric_key in enumerate(METRIC_ORDER):
            point = float(point_scores[model_index, metric_index])
            lower, upper = _bootstrap_interval(
                bootstrap_scores[:, model_index, metric_index],
                config.confidence_level,
            )
            point_metrics[metric_key] = point
            bootstrap_metrics[metric_key] = {
                "point": point,
                "lower": lower,
                "upper": upper,
                "confidence_level": config.confidence_level,
                "method": "stratified_percentile_bootstrap",
            }

        model_rows.append(
            {
                **record,
                "n_samples": n_samples,
                "point_metrics": point_metrics,
                "approximate_accuracy_ci": _normal_approximation_accuracy_ci(
                    point_metrics["accuracy"],
                    n_samples,
                    config.confidence_level,
                ),
                "bootstrap_intervals": bootstrap_metrics,
            }
        )

    pairwise_rows: List[Dict[str, Any]] = []
    raw_mcnemar_p_values: List[float] = []
    for model_a_index, model_b_index in combinations(range(n_models), 2):
        model_a = model_records[model_a_index]
        model_b = model_records[model_b_index]
        scope = (
            "within_strategy"
            if model_a["strategy_type"] == model_b["strategy_type"]
            else "cross_strategy"
        )

        metric_differences: Dict[str, Dict[str, Any]] = {}
        for metric_index, metric_key in enumerate(METRIC_ORDER):
            differences = bootstrap_scores[:, model_a_index, metric_index] - bootstrap_scores[:, model_b_index, metric_index]
            lower, upper = _bootstrap_interval(differences, config.confidence_level)
            point_difference = float(point_scores[model_a_index, metric_index] - point_scores[model_b_index, metric_index])
            contains_zero = bool(lower <= 0.0 <= upper)
            if lower > 0.0:
                direction = "model_a"
            elif upper < 0.0:
                direction = "model_b"
            else:
                direction = "not_confirmed"

            metric_differences[metric_key] = {
                "point_difference": point_difference,
                "lower": lower,
                "upper": upper,
                "confidence_level": config.confidence_level,
                "contains_zero": contains_zero,
                "statistically_significant": not contains_zero,
                "direction": direction,
                "method": "paired_stratified_percentile_bootstrap",
            }

        correct_a = predictions[model_a_index] == y_true
        correct_b = predictions[model_b_index] == y_true
        mcnemar = _mcnemar_result(correct_a, correct_b)
        raw_mcnemar_p_values.append(float(mcnemar["p_value"]))

        pairwise_rows.append(
            {
                "comparison_id": f"{model_a['model_id']}__vs__{model_b['model_id']}",
                "scope": scope,
                "model_a": model_a,
                "model_b": model_b,
                "metric_differences": metric_differences,
                "mcnemar": mcnemar,
            }
        )

    adjusted_p_values = _holm_adjust(raw_mcnemar_p_values)
    for comparison, adjusted_p in zip(pairwise_rows, adjusted_p_values):
        mcnemar = comparison["mcnemar"]
        mcnemar["p_value_holm"] = float(adjusted_p)
        mcnemar["reject_h0_raw"] = bool(mcnemar["p_value"] < config.alpha)
        mcnemar["reject_h0_holm"] = bool(adjusted_p < config.alpha)
        b_count = int(mcnemar["b_a_correct_b_wrong"])
        c_count = int(mcnemar["c_a_wrong_b_correct"])
        if not mcnemar["reject_h0_holm"] or b_count == c_count:
            mcnemar["direction_holm"] = "not_confirmed"
        elif b_count > c_count:
            mcnemar["direction_holm"] = "model_a"
        else:
            mcnemar["direction_holm"] = "model_b"

    summary = {
        "number_of_models": n_models,
        "number_of_pairwise_comparisons": len(pairwise_rows),
        "number_of_cross_strategy_comparisons": sum(row["scope"] == "cross_strategy" for row in pairwise_rows),
        "significant_mcnemar_raw": sum(row["mcnemar"]["reject_h0_raw"] for row in pairwise_rows),
        "significant_mcnemar_holm": sum(row["mcnemar"]["reject_h0_holm"] for row in pairwise_rows),
    }

    class_counts = {
        str(classes[class_index]): int(np.sum(y_true == class_index))
        for class_index in range(n_classes)
    }

    report = {
        "report_type": STATISTICAL_REPORT_TYPE,
        "schema_version": 2,
        "generated_at": datetime.now(ZoneInfo("Europe/Kyiv")).isoformat(),
        "methodology": {
            "approximate_accuracy_ci": "normal approximation p_hat ± z * sqrt(p_hat(1-p_hat)/n)",
            "individual_metric_ci": "stratified percentile bootstrap",
            "pairwise_metric_difference_ci": "paired stratified percentile bootstrap",
            "mcnemar": "exact two-sided binomial when b+c<25; otherwise chi-square with continuity correction",
            "multiple_comparisons": "Holm adjustment over all McNemar pairwise comparisons",
        },
        "config": {
            "confidence_level": config.confidence_level,
            "n_bootstrap": config.n_bootstrap,
            "random_state": config.random_state,
            "alpha": config.alpha,
            "stratified_bootstrap": True,
        },
        "source_reports": [_strategy_source_record(strategy) for strategy in context.strategies],
        "test_set": {
            "fingerprint": context.strategies[0].test_set_fingerprint,
            "n_samples": n_samples,
            "classes": classes,
            "class_counts": class_counts,
            "target_col": context.strategies[0].target_col,
            "Q1": context.strategies[0].q1,
            "Q3": context.strategies[0].q3,
        },
        "models": model_rows,
        "pairwise_comparisons": pairwise_rows,
        "summary": summary,
    }

    return StatisticalAnalysisArtifacts(
        report=report,
        model_rows=model_rows,
        pairwise_rows=pairwise_rows,
    )
