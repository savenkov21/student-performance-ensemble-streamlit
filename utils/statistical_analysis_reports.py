from __future__ import annotations

import hashlib
import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from utils.statistical_analysis_constants import (
    CLASSIFICATION_STRATEGY,
    REGRESSION_STRATEGY,
    SINGLE_STRATEGY_REPORT_TYPE,
    STATISTICAL_REPORT_TYPE,
    STRATEGY_LABELS,
    StatisticalAnalysisContext,
    StrategyStatisticalInput,
)


def uploaded_name(uploaded_file: Any) -> str:
    return getattr(uploaded_file, "name", "uploaded_report.json")


def parse_uploaded_json(uploaded_file: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if uploaded_file is None:
        return None, None

    try:
        raw_bytes = uploaded_file.getvalue()
        raw_text = raw_bytes.decode("utf-8")
    except Exception:
        return None, f"Файл '{uploaded_name(uploaded_file)}' не вдалося прочитати як UTF-8 JSON."

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"Файл '{uploaded_name(uploaded_file)}' не є коректним JSON: {exc}."

    if not isinstance(payload, dict):
        return None, f"Файл '{uploaded_name(uploaded_file)}' повинен містити JSON-об'єкт."
    return payload, None


def _safe_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Поле '{field_name}' повинно містити числове значення.") from exc
    if not np.isfinite(number):
        raise ValueError(f"Поле '{field_name}' повинно містити скінченне числове значення.")
    return number


def _validate_strategy_payload(payload: Dict[str, Any]) -> Optional[str]:
    if payload.get("report_type") != SINGLE_STRATEGY_REPORT_TYPE:
        return (
            f"Очікується звіт типу '{SINGLE_STRATEGY_REPORT_TYPE}', "
            f"отримано '{payload.get('report_type')}'."
        )

    strategy_type = payload.get("strategy_type")
    if strategy_type not in {CLASSIFICATION_STRATEGY, REGRESSION_STRATEGY}:
        return f"Непідтримуване значення 'strategy_type': '{strategy_type}'."

    statistical_data = payload.get("statistical_data")
    if not isinstance(statistical_data, dict):
        return (
            "У звіті відсутній блок 'statistical_data'. Повторно відкрийте модуль "
            "'Оцінювання якості моделей' і завантажте оновлений JSON після інтеграції змін."
        )

    test_set = statistical_data.get("test_set")
    models = statistical_data.get("models")
    if not isinstance(test_set, dict):
        return "У блоці 'statistical_data' відсутній коректний об'єкт 'test_set'."
    if not isinstance(models, dict) or not models:
        return "У блоці 'statistical_data' відсутні покрокові прогнози моделей."

    required_test_keys = ["fingerprint", "n_samples", "sample_ids", "y_true", "classes"]
    missing = [key for key in required_test_keys if key not in test_set]
    if missing:
        return f"У блоці 'test_set' відсутні поля: {', '.join(missing)}."

    if not isinstance(test_set.get("sample_ids"), list) or not isinstance(test_set.get("y_true"), list):
        return "Поля 'sample_ids' і 'y_true' повинні бути списками."

    n_samples = test_set.get("n_samples")
    if not isinstance(n_samples, int) or n_samples <= 0:
        return "Поле 'n_samples' повинно бути додатним цілим числом."
    if len(test_set["sample_ids"]) != n_samples or len(test_set["y_true"]) != n_samples:
        return "Довжини 'sample_ids' та 'y_true' не збігаються з 'n_samples'."

    for model_name, model_data in models.items():
        if not isinstance(model_data, dict):
            return f"Дані моделі '{model_name}' повинні бути JSON-об'єктом."
        y_pred = model_data.get("y_pred")
        if not isinstance(y_pred, list):
            return f"Для моделі '{model_name}' поле 'y_pred' повинно бути списком."
        if len(y_pred) != n_samples:
            return (
                f"Для моделі '{model_name}' довжина 'y_pred' ({len(y_pred)}) "
                f"не збігається з n_samples ({n_samples})."
            )

    return None


def build_strategy_input(payload: Dict[str, Any], file_name: str) -> StrategyStatisticalInput:
    validation_error = _validate_strategy_payload(payload)
    if validation_error:
        raise ValueError(validation_error)

    strategy_type = payload["strategy_type"]
    statistical_data = payload["statistical_data"]
    test_set = statistical_data["test_set"]

    return StrategyStatisticalInput(
        file_name=file_name,
        strategy_type=strategy_type,
        strategy_label=payload.get("strategy_label", STRATEGY_LABELS[strategy_type]),
        target_col=str(payload.get("target_col", "—")),
        q1=_safe_float(payload.get("Q1"), "Q1"),
        q3=_safe_float(payload.get("Q3"), "Q3"),
        test_set_fingerprint=str(test_set["fingerprint"]),
        sample_ids=list(test_set["sample_ids"]),
        y_true=list(test_set["y_true"]),
        classes=list(test_set.get("classes", [])),
        models=dict(statistical_data["models"]),
    )


def _source_signature(uploaded_files: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for uploaded_file in sorted(uploaded_files, key=uploaded_name):
        digest.update(uploaded_name(uploaded_file).encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def load_statistical_context(uploaded_files: List[Any]) -> Tuple[Optional[StatisticalAnalysisContext], List[str]]:
    errors: List[str] = []
    if len(uploaded_files) not in {1, 2}:
        return None, [
            "Завантажте один JSON-звіт для аналізу моделей однієї стратегії або два JSON-звіти для додаткового міжстратегічного порівняння."
        ]

    reports: Dict[str, StrategyStatisticalInput] = {}
    for uploaded_file in uploaded_files:
        payload, parse_error = parse_uploaded_json(uploaded_file)
        if parse_error:
            errors.append(parse_error)
            continue
        assert payload is not None
        try:
            strategy_input = build_strategy_input(payload, uploaded_name(uploaded_file))
        except ValueError as exc:
            errors.append(f"Файл '{uploaded_name(uploaded_file)}': {exc}")
            continue

        if strategy_input.strategy_type in reports:
            errors.append(f"Завантажено два звіти для стратегії '{strategy_input.strategy_label}'.")
            continue
        reports[strategy_input.strategy_type] = strategy_input

    if errors:
        return None, errors
    strategies = list(reports.values())
    if len(strategies) == 2:
        first, second = strategies
        alignment_errors: List[str] = []
        if first.test_set_fingerprint != second.test_set_fingerprint:
            alignment_errors.append(
                "Fingerprint тестових вибірок не збігається. Тест Мак-Немара та парний bootstrap "
                "між стратегіями допустимі лише для прогнозів на тих самих об'єктах у тому самому порядку."
            )
        if first.sample_ids != second.sample_ids:
            alignment_errors.append("Ідентифікатори або порядок об'єктів test set у двох звітах не збігаються.")
        if first.y_true != second.y_true:
            alignment_errors.append("Масиви істинних класів у двох звітах не збігаються.")
        if not np.isclose(first.q1, second.q1) or not np.isclose(first.q3, second.q3):
            alignment_errors.append("Пороги категоризації Q1/Q3 у двох стратегіях не збігаються.")
        if first.target_col != second.target_col:
            alignment_errors.append("Цільові змінні у двох звітах не збігаються.")
        if alignment_errors:
            return None, alignment_errors

    return (
        StatisticalAnalysisContext(
            strategies=strategies,
            source_signature=_source_signature(uploaded_files),
        ),
        [],
    )


def results_to_json_bytes(report: Dict[str, Any]) -> BytesIO:
    buffer = BytesIO()
    buffer.write(json.dumps(report, indent=4, ensure_ascii=False).encode("utf-8"))
    buffer.seek(0)
    return buffer


def build_statistical_export_filename() -> str:
    kyiv_now = datetime.now(ZoneInfo("Europe/Kyiv"))
    return f"classification_statistical_analysis_{kyiv_now.strftime('%Y-%m-%d')}.json"


def validate_statistical_report(payload: Dict[str, Any]) -> Optional[str]:
    if payload.get("report_type") != STATISTICAL_REPORT_TYPE:
        return f"Очікується JSON типу '{STATISTICAL_REPORT_TYPE}'."
    required = ["config", "source_reports", "test_set", "models", "pairwise_comparisons", "summary"]
    missing = [key for key in required if key not in payload]
    if missing:
        return f"У статистичному JSON відсутні поля: {', '.join(missing)}."
    if not isinstance(payload.get("models"), list):
        return "Поле 'models' повинно бути списком."
    if not isinstance(payload.get("pairwise_comparisons"), list):
        return "Поле 'pairwise_comparisons' повинно бути списком."
    return None
