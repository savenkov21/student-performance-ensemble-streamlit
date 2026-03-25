from __future__ import annotations

import json
import re
import ast
from datetime import datetime
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


CLASSIFICATION_STRATEGY = "direct_multiclass_classification"
REGRESSION_STRATEGY = "regression_with_categorization"


@dataclass
class EvaluationContext:
    strategy_type: str
    target_col: str
    q1: Any
    q3: Any
    meta_ready: bool
    first_level_results: Dict[str, Dict[str, Any]]
    second_level_results: Dict[str, Dict[str, Any]]
    first_level_params: Dict[str, Any]
    meta_generation_params: Dict[str, Any]
    second_level_params: Dict[str, Any]


@dataclass
class EvaluationArtifacts:
    comparison_df: pd.DataFrame
    best_model: Dict[str, Any]
    final_report: Dict[str, Any]


def _results_to_json_bytes(results_dict):
    json_bytes = BytesIO()
    json_bytes.write(json.dumps(results_dict, indent=4, ensure_ascii=False).encode("utf-8"))
    json_bytes.seek(0)
    return json_bytes


def _strategy_label(strategy_type):
    mapping = {
        CLASSIFICATION_STRATEGY: "Пряма багатокласова класифікація",
        REGRESSION_STRATEGY: "Регресія з подальшою категоризацією",
    }
    return mapping.get(strategy_type, strategy_type)



def _get_primary_metric(strategy_type):
    return "F1 weighted" if strategy_type == CLASSIFICATION_STRATEGY else "F1 weighted"


def _strategy_export_slug(strategy_type: str) -> str:
    mapping = {
        CLASSIFICATION_STRATEGY: "classification",
        REGRESSION_STRATEGY: "regression",
    }
    return mapping.get(strategy_type, "strategy")


def _build_export_filename(strategy_type: str) -> str:
    kyiv_now = datetime.now(ZoneInfo("Europe/Kyiv"))
    date_part = kyiv_now.strftime("%Y-%m-%d")
    return f"{_strategy_export_slug(strategy_type)}_strategy_results_{date_part}.json"


def _check_ready_for_evaluation():
    required_keys = [
        "strategy_type",
        "first_level_results",
        "first_level_strategy_type",
        "second_level_results",
        "second_level_strategy_type",
    ]
    missing = [key for key in required_keys if key not in st.session_state]

    if missing:
        st.warning(
            "Спочатку завантажте дані на сторінці 'Завантаження та первинний аналіз даних'."
        )
        return False

    strategy_type = st.session_state.get("strategy_type")

    if st.session_state.get("first_level_strategy_type") != strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть базові моделі на попередньому етапі.")
        return False

    if st.session_state.get("second_level_strategy_type") != strategy_type:
        st.warning("Стратегія змінилася. Перенавчіть ансамблі другого рівня.")
        return False

    return True


def _load_context() -> Optional[EvaluationContext]:
    if not _check_ready_for_evaluation():
        return None

    return EvaluationContext(
        strategy_type=st.session_state.get("strategy_type"),
        target_col=st.session_state.get("target_col", "—"),
        q1=st.session_state.get("Q1", "—"),
        q3=st.session_state.get("Q3", "—"),
        meta_ready=bool(st.session_state.get("meta_ready", False)),
        first_level_results=st.session_state.get("first_level_results", {}),
        second_level_results=st.session_state.get("second_level_results", {}),
        first_level_params=st.session_state.get("first_level_params", {}),
        meta_generation_params=st.session_state.get("meta_generation_params", {}),
        second_level_params=st.session_state.get("second_level_params", {}),
    )


def _get_result_value(result: Dict[str, Any], *keys: str, default: Any = None):
    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
    if default is not None:
        return default
    raise KeyError(keys[0] if keys else "missing_key")

def _classification_row(model_name, result, level, family):
    return {
        "Strategy": "Direct multiclass classification",
        "Model": model_name,
        "Level": level,
        "Family": family,
        "Accuracy": float(_get_result_value(result, "Accuracy", default=0.0)),
        "Balanced Accuracy": float(_get_result_value(result, "Balanced_accuracy", default=0.0)),
        "F1 weighted": float(_get_result_value(result, "F1_weighted", default=0.0)),
    }


def _regression_row(model_name, result, level, family):
    return {
        "Strategy": "Regression with categorization",
        "Model": model_name,
        "Level": level,
        "Family": family,
        "MAE": float(_get_result_value(result, "MAE", default=0.0)),
        "MSE": float(_get_result_value(result, "MSE", default=0.0)),
        "RMSE": float(_get_result_value(result, "RMSE", default=0.0)),
        "R2": float(_get_result_value(result, "R2", "R²", default=0.0)),
        "Accuracy": float(_get_result_value(result, "Accuracy", "Accuracy_after_discretization", default=0.0)),
        "Balanced Accuracy": float(_get_result_value(result, "Balanced_accuracy", "Balanced_accuracy_after_discretization", "Balanced_accuracyter", default=0.0)),
        "F1 weighted": float(_get_result_value(result, "F1_weighted", "F1_weighted_after_discretization", default=0.0)),
    }


def _build_comparison_dataframe(strategy_type, first_level_results, second_level_results):
    rows: List[Dict[str, Any]] = []

    if strategy_type == CLASSIFICATION_STRATEGY:
        for model_name, result in first_level_results.items():
            rows.append(_classification_row(model_name, result, "Перший рівень", "Базова модель"))
        for model_name, result in second_level_results.items():
            rows.append(_classification_row(model_name, result, "Другий рівень", "Ансамбль"))
    else:
        for model_name, result in first_level_results.items():
            rows.append(_regression_row(model_name, result, "Перший рівень", "Базова модель"))
        for model_name, result in second_level_results.items():
            rows.append(_regression_row(model_name, result, "Другий рівень", "Ансамбль"))

    return pd.DataFrame(rows)


def _select_best_model(strategy_type, comparison_df):
    metric = _get_primary_metric(strategy_type)
    best_idx = comparison_df[metric].idxmax()
    best_row = comparison_df.loc[best_idx]
    return {
        "name": best_row["Model"],
        "level": best_row["Level"],
        "family": best_row["Family"],
        "metric": metric,
        "value": float(best_row[metric]),
    }



def _build_final_report(context: EvaluationContext, comparison_df, best_model):
    return {
        "report_type": "single_strategy_evaluation",
        "strategy_type": context.strategy_type,
        "strategy_label": _strategy_label(context.strategy_type),
        "target_col": context.target_col,
        "Q1": context.q1,
        "Q3": context.q3,
        "best_model": best_model,
        "first_level_params": context.first_level_params,
        "meta_generation_params": context.meta_generation_params,
        "second_level_params": context.second_level_params,
        "first_level_results": context.first_level_results,
        "second_level_results": context.second_level_results,
        "comparison_table": comparison_df.to_dict(orient="records"),
    }


def _build_artifacts(context: EvaluationContext) -> EvaluationArtifacts:
    comparison_df = _build_comparison_dataframe(
        strategy_type=context.strategy_type,
        first_level_results=context.first_level_results,
        second_level_results=context.second_level_results,
    )
    best_model = _select_best_model(context.strategy_type, comparison_df)
    final_report = _build_final_report(context, comparison_df, best_model)
    return EvaluationArtifacts(comparison_df=comparison_df, best_model=best_model, final_report=final_report)


def _metric_value(value):
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def _inject_page_style() -> None:
    st.markdown(
        """
        <style>
        .m4-hero {
            padding: 1rem 1.15rem;
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(70,130,180,0.08), rgba(255,255,255,0.02));
            margin-bottom: 0.8rem;
        }
        .m4-section-title {
            margin-top: 0.55rem;
            margin-bottom: 0.15rem;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .m4-subtle {
            color: rgba(120,120,120,0.95);
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }
        .m4-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            background: rgba(255,255,255,0.02);
            min-height: 110px;
        }
        .m4-card-kicker {
            font-size: 0.82rem;
            color: rgba(120,120,120,0.95);
            margin-bottom: 0.35rem;
        }
        .m4-card-value {
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.3rem;
            word-break: break-word;
        }
        .m4-card-note {
            font-size: 0.86rem;
            color: rgba(120,120,120,0.92);
        }
        .m4-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(120,120,120,0.25);
            font-size: 0.8rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .m4-note {
            border-left: 4px solid rgba(70,130,180,0.55);
            padding: 0.75rem 0.9rem;
            background: rgba(70,130,180,0.06);
            border-radius: 10px;
            margin-top: 0.4rem;
            margin-bottom: 0.35rem;
        }
        .m4-mini-note {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="m4-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="m4-subtle">{subtitle}</div>', unsafe_allow_html=True)



def _render_info_note(text: str) -> None:
    st.markdown(f"<div class='m4-note'>{text}</div>", unsafe_allow_html=True)



def _render_hero_section(context: EvaluationContext, artifacts: EvaluationArtifacts):
    st.markdown(
        f"""
        <div class="m4-hero">
            <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;">Оцінювання якості моделей</div>
            <div style="margin-bottom:0.6rem;">
                На цьому етапі підсумовуються результати поточної стратегії прогнозування успішності студентів.
            </div>
            <span class="m4-pill">{_strategy_label(context.strategy_type)}</span>
            <span class="m4-pill">Оцінювання якості моделей</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    

def _render_experiment_snapshot(context: EvaluationContext):
    _render_section_header(
        "1. Поточний стан експерименту"
    )
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(f"**Стратегія:** {_strategy_label(context.strategy_type)}")
    st.markdown(f"**Цільова змінна:** {context.target_col}")
    
    c1, c2 = st.columns(2)
    c1.metric("Нижній поріг категоризації: Q1", f"{context.q1:.0f}")
    c2.metric("Верхній поріг категоризації: Q3", f"{context.q3:.0f}")
    
    st.info(
            "Для класифікації головною метрикою підсумкового порівняння є F1 weighted. "
            "Для регресійної стратегії використовується F1 weighted, "
            "тобто після переведення неперервних прогнозів у класи за межами Q1 і Q3."
        )
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


def _prepare_ranking_df(comparison_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ranking_df = comparison_df.copy()
    ranking_df = ranking_df.sort_values(metric, ascending=False).reset_index(drop=True)
    ranking_df["Rank"] = range(1, len(ranking_df) + 1)
    return ranking_df


def _render_comparison_section(context: EvaluationContext, artifacts: EvaluationArtifacts):
    metric = _get_primary_metric(context.strategy_type)
    _render_section_header(
        "2. Порівняння результатів",
        "Зведений підсумок для моделей першого рівня та ансамблів другого рівня в межах поточної стратегії.",
    )

    ranking_df = _prepare_ranking_df(artifacts.comparison_df, metric)

    _render_info_note(
        f"Фінальне ранжування виконується за метрикою <b>{metric}</b>. "
        f"Найкращий результат у поточній стратегії показала модель <b>{artifacts.best_model['name']}</b> "
        f"({artifacts.best_model['level']}, {artifacts.best_model['family']})."
    )

    st.markdown("#### Рейтинг моделей")
    summary_df = ranking_df[["Rank", "Model", metric]].copy()
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    with st.expander("Показати повну таблицю метрик", expanded=False):
        display_df = artifacts.comparison_df.copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_classification_metrics(result):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", _metric_value(result["Accuracy"]))
    c2.metric("Balanced Accuracy", _metric_value(result["Balanced_accuracy"]))
    c3.metric("F1 weighted", _metric_value(result["F1_weighted"]))


def _render_regression_metrics(result):
    st.markdown("**Регресійні метрики**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", _metric_value(_get_result_value(result, "MAE", default=0.0)))
    c2.metric("MSE", _metric_value(_get_result_value(result, "MSE", default=0.0)))
    c3.metric("RMSE", _metric_value(_get_result_value(result, "RMSE", default=0.0)))
    c4.metric("R²", _metric_value(_get_result_value(result, "R2", "R²", default=0.0)))

    st.markdown("**Після категоризації**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", _metric_value(_get_result_value(result, "Accuracy", "Accuracy_after_discretization", default=0.0)))
    c2.metric("Balanced Accuracy", _metric_value(_get_result_value(result, "Balanced_accuracy", "Balanced_accuracy_after_discretization", "Balanced_accuracyter", default=0.0)))
    c3.metric("F1 weighted", _metric_value(_get_result_value(result, "F1_weighted", "F1_weighted_after_discretization", default=0.0)))


def _classification_metric_rows(result):
    return pd.DataFrame(
        [
            {"Метрика": "Accuracy", "Значення": float(result["Accuracy"])},
            {"Метрика": "Balanced Accuracy", "Значення": float(result["Balanced_accuracy"])},
            {"Метрика": "F1 weighted", "Значення": float(result["F1_weighted"])},
        ]
    )


def _parse_classification_report_text(report_text: str) -> pd.DataFrame:
    if not report_text or not isinstance(report_text, str):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("precision") or line.startswith("accuracy"):
            continue

        parts = re.split(r"\s{2,}", line)
        if len(parts) < 2:
            continue

        label = parts[0]
        numeric_parts = parts[1:]
        if len(numeric_parts) >= 4:
            try:
                rows.append(
                    {
                        "Клас / середнє": label,
                        "Precision": float(numeric_parts[0]),
                        "Recall": float(numeric_parts[1]),
                        "F1-score": float(numeric_parts[2]),
                        "Support": int(float(numeric_parts[3])),
                    }
                )
            except ValueError:
                continue

    return pd.DataFrame(rows)


def _normalize_report_df(report_df: pd.DataFrame) -> pd.DataFrame:
    if report_df.empty:
        return report_df

    rename_map = {
        "index": "Клас / середнє",
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1-score",
        "support": "Support",
    }
    report_df = report_df.rename(columns=rename_map)

    preferred_columns = [
        "Клас / середнє",
        "Precision",
        "Recall",
        "F1-score",
        "Support",
    ]
    existing = [col for col in preferred_columns if col in report_df.columns]
    remaining = [col for col in report_df.columns if col not in existing]
    return report_df[existing + remaining]


def _report_to_dataframe(report_value: Any) -> pd.DataFrame:
    if report_value is None:
        return pd.DataFrame()

    if isinstance(report_value, dict):
        try:
            return _normalize_report_df(pd.DataFrame(report_value).T.reset_index())
        except Exception:
            return pd.DataFrame()

    if isinstance(report_value, str):
        stripped = report_value.strip()
        if not stripped:
            return pd.DataFrame()

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    return _report_to_dataframe(parsed)
            except Exception:
                pass

        return _normalize_report_df(_parse_classification_report_text(stripped))

    return pd.DataFrame()


def _render_classification_report(report_value: Any, title: str) -> None:
    if report_value is None:
        return

    st.markdown(f"**{title}**")
    report_df = _report_to_dataframe(report_value)
    if not report_df.empty:
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        return

    if isinstance(report_value, str):
        st.code(report_value, language="text")
        return

    st.caption("Не вдалося відобразити classification report у табличному вигляді.")



def _render_result_block(strategy_type, title, result):
    st.markdown(f"### {title}")

    if strategy_type == CLASSIFICATION_STRATEGY:
        _render_classification_metrics(result)

        metric_df = _classification_metric_rows(result)
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        if "used_probability_groups" in result and result["used_probability_groups"]:
            used_groups_text = ", ".join(
                str(group).replace("meta_", "") for group in result["used_probability_groups"]
            )
            st.caption(f"Використані probability-блоки: {used_groups_text}")

        if "classification_report" in result:
            _render_classification_report(result["classification_report"], "Classification report")
        return

    _render_regression_metrics(result)

    report_value = _get_result_value(
        result,
        "classification_report",
        "classification_report_after_discretization",
        default=None,
    )
    if report_value is not None:
        _render_classification_report(report_value, "Звіт після категоризації")


def _ordered_second_level_names(strategy_type, second_level_results):
    preferred = ["SoftVoting", "Stacking"] if strategy_type == CLASSIFICATION_STRATEGY else ["Voting", "Stacking"]
    existing = [name for name in preferred if name in second_level_results]
    remaining = [name for name in second_level_results.keys() if name not in existing]
    return existing + remaining


def _render_details_section(context: EvaluationContext):
    _render_section_header(
        "3. Деталі моделей",
        "Розгорнуті метрики для базових моделей і ансамблів другого рівня.",
    )

    second_level_names = _ordered_second_level_names(context.strategy_type, context.second_level_results)
    tab_titles = ["Базові моделі"] + second_level_names[:2] if len(second_level_names) >= 2 else ["Базові моделі", "Ансамблі 2-го рівня"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        for model_name, result in context.first_level_results.items():
            with st.container(border=True):
                _render_result_block(context.strategy_type, model_name, result)

    if len(second_level_names) >= 2:
        with tabs[1]:
            with st.container(border=True):
                _render_result_block(
                    context.strategy_type,
                    second_level_names[0],
                    context.second_level_results[second_level_names[0]],
                )
        with tabs[2]:
            with st.container(border=True):
                _render_result_block(
                    context.strategy_type,
                    second_level_names[1],
                    context.second_level_results[second_level_names[1]],
                )
        if len(second_level_names) > 2:
            with st.expander("Додаткові ансамблі другого рівня"):
                for extra_name in second_level_names[2:]:
                    with st.container(border=True):
                        _render_result_block(
                            context.strategy_type,
                            extra_name,
                            context.second_level_results[extra_name],
                        )
        return

    with tabs[1]:
        for ensemble_name, result in context.second_level_results.items():
            with st.container(border=True):
                _render_result_block(context.strategy_type, ensemble_name, result)


def _render_export_section(context: EvaluationContext, artifacts: EvaluationArtifacts):
    _render_section_header(
        "4. Експорт підсумкового звіту",
        "Збереження фінального JSON-звіту для поточної стратегії з параметрами та метриками.",
    )
    json_bytes = _results_to_json_bytes(artifacts.final_report)

    st.download_button(
            label="Завантажити фінальний звіт (JSON)",
            data=json_bytes,
            file_name=_build_export_filename(context.strategy_type),
            mime="application/json",
            use_container_width=True,
        )
    
    st.info(
            "Назва файлу автоматично містить назву стратегії та дату завантаження, "
            "щоб JSON-файли було легко розрізняти."
        )

    with st.expander("Попередній перегляд JSON", expanded=False):
        st.json(artifacts.final_report)


def page_evaluation_results():
    _inject_page_style()
    context = _load_context()
    if context is None:
        return
    
    artifacts = _build_artifacts(context)

    _render_hero_section(context, artifacts)
    _render_experiment_snapshot(context)
    _render_comparison_section(context, artifacts)
    _render_details_section(context)
    _render_export_section(context, artifacts)
