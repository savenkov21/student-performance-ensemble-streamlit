from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from utils.statistical_analysis_constants import METRIC_NAMES, METRIC_ORDER, StatisticalAnalysisConfig
from utils.statistical_analysis_core import run_statistical_analysis
from utils.statistical_analysis_reports import (
    build_statistical_export_filename,
    load_statistical_context,
    results_to_json_bytes,
)
from utils.statistical_analysis_state import load_statistical_analysis, save_statistical_analysis


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _format_p_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "<0.0001" if number < 0.0001 else f"{number:.4f}"


def _ci_text(interval: Dict[str, Any]) -> str:
    return f"[{_format_number(interval.get('lower'))}; {_format_number(interval.get('upper'))}]"


def _model_label(model: Dict[str, Any]) -> str:
    return f"{model.get('strategy_label', '—')} — {model.get('model_name', '—')}"


def _request_signature(source_signature: str, config: StatisticalAnalysisConfig) -> str:
    raw = json.dumps(
        {
            "source_signature": source_signature,
            "confidence_level": config.confidence_level,
            "n_bootstrap": config.n_bootstrap,
            "random_state": config.random_state,
            "alpha": config.alpha,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def model_intervals_dataframe(report: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model in report.get("models", []):
        point = model.get("point_metrics", {})
        bootstrap = model.get("bootstrap_intervals", {})
        rows.append(
            {
                "Стратегія": model.get("strategy_label", "—"),
                "Модель": model.get("model_name", "—"),
                "Рівень": model.get("level", "—"),
                "Accuracy": point.get("accuracy"),
                "Наближений 95% CI Accuracy": _ci_text(model.get("approximate_accuracy_ci", {})),
                "Bootstrap 95% CI Accuracy": _ci_text(bootstrap.get("accuracy", {})),
                "Balanced Accuracy": point.get("balanced_accuracy"),
                "Bootstrap 95% CI Balanced Accuracy": _ci_text(bootstrap.get("balanced_accuracy", {})),
                "F1 macro": point.get("f1_macro"),
                "Bootstrap 95% CI F1 macro": _ci_text(bootstrap.get("f1_macro", {})),
                "F1 weighted": point.get("f1_weighted"),
                "Bootstrap 95% CI F1 weighted": _ci_text(bootstrap.get("f1_weighted", {})),
            }
        )
    return pd.DataFrame(rows)


def mcnemar_dataframe(report: Dict[str, Any], scope: str = "all") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for comparison in report.get("pairwise_comparisons", []):
        if scope != "all" and comparison.get("scope") != scope:
            continue
        model_a = comparison.get("model_a", {})
        model_b = comparison.get("model_b", {})
        result = comparison.get("mcnemar", {})
        direction = result.get("direction_holm")
        if direction == "model_a":
            conclusion = f"Перевага: {model_a.get('model_name', 'A')}"
        elif direction == "model_b":
            conclusion = f"Перевага: {model_b.get('model_name', 'B')}"
        else:
            conclusion = "Значущої відмінності не виявлено"

        rows.append(
            {
                "Модель A": _model_label(model_a),
                "Модель B": _model_label(model_b),
                "Тип порівняння": "Між стратегіями" if comparison.get("scope") == "cross_strategy" else "У межах стратегії",
                "b (A правильна, B помилкова)": result.get("b_a_correct_b_wrong"),
                "c (A помилкова, B правильна)": result.get("c_a_wrong_b_correct"),
                "b+c": result.get("discordant"),
                "Метод": "Точний" if result.get("exact") else "χ² з поправкою неперервності",
                "Статистика": result.get("statistic"),
                "p": _format_p_value(result.get("p_value")),
                "p (Holm)": _format_p_value(result.get("p_value_holm")),
                "Висновок (Holm, α=0.05)": conclusion,
            }
        )
    return pd.DataFrame(rows)


def paired_difference_dataframe(
    report: Dict[str, Any],
    metric_key: str,
    scope: str = "all",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for comparison in report.get("pairwise_comparisons", []):
        if scope != "all" and comparison.get("scope") != scope:
            continue
        model_a = comparison.get("model_a", {})
        model_b = comparison.get("model_b", {})
        difference = comparison.get("metric_differences", {}).get(metric_key, {})
        direction = difference.get("direction")
        if direction == "model_a":
            conclusion = f"Перевага: {model_a.get('model_name', 'A')}"
        elif direction == "model_b":
            conclusion = f"Перевага: {model_b.get('model_name', 'B')}"
        else:
            conclusion = "Значущу перевагу не підтверджено"

        rows.append(
            {
                "Модель A": _model_label(model_a),
                "Модель B": _model_label(model_b),
                "Тип порівняння": "Між стратегіями" if comparison.get("scope") == "cross_strategy" else "У межах стратегії",
                f"Δ {METRIC_NAMES.get(metric_key, metric_key)} (A−B)": difference.get("point_difference"),
                "95% bootstrap CI для Δ": _ci_text(difference),
                "Містить 0": "Так" if difference.get("contains_zero") else "Ні",
                "Висновок": conclusion,
            }
        )
    return pd.DataFrame(rows)


def render_saved_statistical_report(report: Dict[str, Any]) -> None:
    test_set = report.get("test_set", {})
    config = report.get("config", {})
    summary = report.get("summary", {})

    st.markdown("### Параметри аналізу")
    overview = pd.DataFrame(
        [
            {"Параметр": "Кількість тестових об'єктів", "Значення": test_set.get("n_samples", "—")},
            {"Параметр": "Класи", "Значення": ", ".join(map(str, test_set.get("classes", [])))},
            {"Параметр": "Кількість bootstrap-повторень", "Значення": config.get("n_bootstrap", "—")},
            {"Параметр": "Рівень довіри", "Значення": config.get("confidence_level", "—")},
            {"Параметр": "Рівень значущості α", "Значення": config.get("alpha", "—")},
            {"Параметр": "Кількість моделей", "Значення": summary.get("number_of_models", "—")},
            {"Параметр": "Кількість парних порівнянь", "Значення": summary.get("number_of_pairwise_comparisons", "—")},
        ]
    )
    st.dataframe(overview, use_container_width=True, hide_index=True)

    st.info(
        "Статистичний звіт містить усі парні порівняння моделей. "
    )

    st.markdown("### Довірчі інтервали окремих моделей")
    intervals_df = model_intervals_dataframe(report)
    if intervals_df.empty:
        st.warning("У JSON відсутні результати окремих моделей.")
    else:
        st.dataframe(
            intervals_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "Balanced Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "F1 macro": st.column_config.NumberColumn(format="%.4f"),
                "F1 weighted": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    st.markdown("### Тест Мак-Немара")
    scope_label = st.selectbox(
        "Показати порівняння",
        options=["Усі", "Між стратегіями", "У межах стратегій"],
        key="statistical_mcnemar_scope",
    )
    scope_map = {"Усі": "all", "Між стратегіями": "cross_strategy", "У межах стратегій": "within_strategy"}
    mcnemar_df = mcnemar_dataframe(report, scope=scope_map[scope_label])
    st.dataframe(
        mcnemar_df,
        use_container_width=True,
        hide_index=True,
        column_config={"Статистика": st.column_config.NumberColumn(format="%.4f")},
    )
    st.caption(
        "Поправка Холма застосована до всіх парних тестів Мак-Немара. "
        "p ≥ 0.05 не доводить однаковість моделей, а лише означає, що значущої відмінності не виявлено."
    )

    st.markdown("### Парний bootstrap для різниць метрик")
    metric_label = st.selectbox(
        "Метрика",
        options=[METRIC_NAMES[key] for key in METRIC_ORDER],
        index=3,
        key="statistical_difference_metric",
    )
    metric_key = next(key for key, label in METRIC_NAMES.items() if label == metric_label)
    difference_df = paired_difference_dataframe(report, metric_key=metric_key, scope=scope_map[scope_label])
    st.dataframe(
        difference_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            f"Δ {metric_label} (A−B)": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "Якщо 95% bootstrap-інтервал різниці містить 0, статистично значущу перевагу "
        "за відповідною метрикою не підтверджено."
    )


def render_statistical_analysis() -> None:
    st.title("Статистичний аналіз результатів класифікації")
    st.caption(
        "Наближений 95% довірчий інтервал для Accuracy, стратифікований bootstrap для метрик "
        "класифікації, парний bootstrap різниць і тест Мак-Немара для моделей однієї або двох стратегій."
    )

    uploaded_files = st.file_uploader(
        "JSON-звіти двох стратегій",
        type=["json"],
        accept_multiple_files=True,
        key="module8_statistical_source_reports",
        help="Один файл дає аналіз усіх моделей однієї стратегії; два узгоджені файли додають міжстратегічні порівняння.",
    ) or []

    c1, c2 = st.columns(2)
    n_bootstrap = int(
        c1.number_input(
            "Кількість bootstrap-повторень",
            min_value=500,
            max_value=10000,
            value=2000,
            step=500,
            help="Рекомендація: B=2000. Більше значення підвищує стабільність оцінок, але збільшує час розрахунку.",
        )
    )
    random_state = int(c2.number_input("Random state", min_value=0, max_value=2_147_483_647, value=42, step=1))
    config = StatisticalAnalysisConfig(
        confidence_level=0.95,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        alpha=0.05,
    )

    context, errors = load_statistical_context(uploaded_files)
    for error in errors:
        st.error(error)

    if context is not None:
        source_df = pd.DataFrame([
            {"Файл": strategy.file_name, "Стратегія": strategy.strategy_label, "Моделей": len(strategy.models)}
            for strategy in context.strategies
        ])
        st.dataframe(source_df, use_container_width=True, hide_index=True)
        if len(context.strategies) == 2:
            st.success("Тестові вибірки двох стратегій збігаються. Можна виконувати внутрішні та міжстратегічні парні порівняння.")
        else:
            st.success("Звіт перевірено. Будуть проаналізовані всі моделі цієї стратегії; за бажанням додайте другий звіт.")

        request_signature = _request_signature(context.source_signature, config)
        if st.button("Виконати статистичний аналіз", type="primary", use_container_width=True):
            try:
                with st.spinner("Обчислення bootstrap-розподілів і парних тестів..."):
                    artifacts = run_statistical_analysis(context, config)
                save_statistical_analysis(artifacts.report, request_signature)
                st.success("Статистичний аналіз завершено.")
            except ValueError as exc:
                st.error(str(exc))

        report = load_statistical_analysis(request_signature)
        if report is not None:
            st.markdown("---")
            render_saved_statistical_report(report)
            st.markdown("### Експорт повного статистичного звіту")
            st.download_button(
                "Завантажити статистичний аналіз (JSON)",
                data=results_to_json_bytes(report),
                file_name=build_statistical_export_filename(),
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.button(
            "Виконати статистичний аналіз",
            type="primary",
            use_container_width=True,
            disabled=True,
            help="Кнопка стане активною після завантаження одного коректного JSON або двох узгоджених JSON-звітів.",
        )
        if not uploaded_files:
            st.info(
                "Спочатку пройдіть етап 'Оцінювання якості моделей' і завантажте щонайменше один оновлений JSON-звіт."
            )
