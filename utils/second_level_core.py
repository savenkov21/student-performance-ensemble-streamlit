from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from utils.second_level_constants import (
    CLASSIFICATION_STRATEGY,
    RANDOM_STATE,
    REGRESSION_STRATEGY,
    SOFT_VOTING_NAME,
    STACKING_NAME,
    VOTING_NAME,
    SecondLevelArtifacts,
    SecondLevelConfig,
    SecondLevelContext,
)


def discretize_g3(y, q1, q3):
    y = np.asarray(y)
    cls = np.zeros_like(y, dtype=int)
    cls[y <= q1] = 0
    cls[(y > q1) & (y <= q3)] = 1
    cls[y > q3] = 2
    return cls


def safe_classification_report(y_true, y_pred) -> dict[str, Any]:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def classification_metrics_dict(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    report = safe_classification_report(y_true, y_pred)
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "model_name": model_name,
        "Accuracy": float(acc),
        "Balanced_accuracy": float(bal_acc),
        "F1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": labels.tolist(),
    }


def regression_with_discretization_metrics_dict(model_name, y_true_cont, y_pred_cont, q1, q3):
    mae = mean_absolute_error(y_true_cont, y_pred_cont)
    mse = mean_squared_error(y_true_cont, y_pred_cont)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true_cont, y_pred_cont)

    y_true_cls = discretize_g3(y_true_cont, q1, q3)
    y_pred_cls = discretize_g3(y_pred_cont, q1, q3)

    acc_cls = accuracy_score(y_true_cls, y_pred_cls)
    bal_acc_cls = balanced_accuracy_score(y_true_cls, y_pred_cls)
    f1_weighted_cls = f1_score(y_true_cls, y_pred_cls, average="weighted")

    report_cls = safe_classification_report(y_true_cls, y_pred_cls)
    labels_cls = np.unique(np.concatenate([np.asarray(y_true_cls), np.asarray(y_pred_cls)]))
    cm_cls = confusion_matrix(y_true_cls, y_pred_cls, labels=labels_cls)

    return {
        "model_name": model_name,
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Accuracy": float(acc_cls),
        "Balanced_accuracy": float(bal_acc_cls),
        "F1_weighted": float(f1_weighted_cls),
        "classification_report": report_cls,
        "confusion_matrix": cm_cls.tolist(),
        "classes": labels_cls.tolist(),
        "y_true_cont": np.asarray(y_true_cont, dtype=float).tolist(),
        "y_pred_cont": np.asarray(y_pred_cont, dtype=float).tolist(),
    }


def extract_proba_groups(meta_X) -> dict[str, dict[int, str]]:
    meta_df = pd.DataFrame(meta_X).copy()
    proba_groups: dict[str, dict[int, str]] = {}

    for col in meta_df.columns:
        if "_proba_" not in col:
            continue
        model_part, class_part = col.rsplit("_proba_", 1)
        try:
            class_idx = int(class_part)
        except ValueError:
            continue
        proba_groups.setdefault(model_part, {})
        proba_groups[model_part][class_idx] = col

    return proba_groups


def soft_voting_from_meta_proba(meta_X, classes, weights=None):
    meta_df = pd.DataFrame(meta_X).copy()
    classes = list(classes)
    proba_groups = extract_proba_groups(meta_df)
    if not proba_groups:
        raise ValueError(
            "Для Soft Voting не знайдено probability-колонок у meta_X_test. Переконайтеся, що сформовано класифікаційні метаознаки через predict_proba."
        )

    ordered_group_names = sorted(proba_groups.keys())
    model_prob_arrays = []
    for group_name in ordered_group_names:
        class_map = proba_groups[group_name]
        missing_classes = [cls for cls in classes if int(cls) not in class_map]
        if missing_classes:
            raise ValueError(f"У probability-блоці {group_name} відсутні класи: {missing_classes}")
        ordered_cols = [class_map[int(cls)] for cls in classes]
        model_prob_arrays.append(meta_df[ordered_cols].to_numpy(dtype=float))

    stacked = np.stack(model_prob_arrays, axis=0)
    if weights is None:
        avg_proba = np.mean(stacked, axis=0)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(model_prob_arrays):
            raise ValueError(
                f"Кількість ваг ({len(weights)}) не збігається з кількістю базових моделей ({len(model_prob_arrays)})."
            )
        if np.isclose(weights.sum(), 0.0):
            raise ValueError("Сума ваг для Soft Voting не може дорівнювати 0.")
        avg_proba = np.average(stacked, axis=0, weights=weights)

    y_pred_cls = np.asarray(classes)[np.argmax(avg_proba, axis=1)]
    return avg_proba, y_pred_cls, ordered_group_names


def extract_regression_prediction_columns(meta_X) -> list[str]:
    meta_df = pd.DataFrame(meta_X).copy()
    return [col for col in meta_df.columns if str(col).startswith("meta_") and "_proba_" not in str(col)]


def default_regression_weights(meta_X_test: pd.DataFrame) -> list[float]:
    return [1.0] * len(extract_regression_prediction_columns(meta_X_test))


def default_classification_weights(meta_X_test: pd.DataFrame) -> list[float]:
    group_count = len(extract_proba_groups(meta_X_test))
    return [1.0] * max(group_count, 0)


def train_soft_voting_classifier(meta_X_test, y_test_cls, classes, weights=None):
    avg_proba, y_pred_cls, used_groups = soft_voting_from_meta_proba(meta_X=meta_X_test, classes=classes, weights=weights)
    results = classification_metrics_dict(model_name=SOFT_VOTING_NAME, y_true=y_test_cls, y_pred=y_pred_cls)
    results["used_probability_groups"] = used_groups
    return avg_proba, np.asarray(y_pred_cls), results


def train_stacking_classifier(meta_X_train, meta_X_test, y_train_cls, y_test_cls):
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(meta_X_train, y_train_cls)
    y_pred_cls = model.predict(meta_X_test)
    results = classification_metrics_dict(model_name=STACKING_NAME, y_true=y_test_cls, y_pred=y_pred_cls)
    return model, np.asarray(y_pred_cls), results


def train_voting_regressor_from_meta(meta_X_test, y_test_cont, q1, q3, weights=None):
    meta_df = pd.DataFrame(meta_X_test).copy()
    pred_cols = extract_regression_prediction_columns(meta_df)
    if not pred_cols:
        raise ValueError("Для Voting Regressor не знайдено регресійних метапрогнозів у meta_X_test.")

    meta_np = meta_df[pred_cols].to_numpy(dtype=float)
    if weights is None:
        weights = np.ones(len(pred_cols), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(pred_cols):
            raise ValueError(f"Кількість ваг ({len(weights)}) не збігається з кількістю регресійних метапрогнозів ({len(pred_cols)}).")
        if np.isclose(weights.sum(), 0.0):
            raise ValueError("Сума ваг для Voting Regressor не може дорівнювати 0.")

    y_pred_cont = np.average(meta_np, axis=1, weights=weights)
    y_pred_cls = discretize_g3(y_pred_cont, q1, q3)
    results = regression_with_discretization_metrics_dict(
        model_name=VOTING_NAME,
        y_true_cont=y_test_cont,
        y_pred_cont=y_pred_cont,
        q1=q1,
        q3=q3,
    )
    results["used_regression_prediction_columns"] = pred_cols
    results["used_weights"] = weights.astype(float).tolist()
    return np.asarray(y_pred_cont), np.asarray(y_pred_cls), results


def train_stacking_regressor(meta_X_train, meta_X_test, y_train_cont, y_test_cont, q1, q3):
    model = Ridge(alpha=1.0)
    model.fit(meta_X_train, y_train_cont)
    y_pred_cont = model.predict(meta_X_test)
    y_pred_cls = discretize_g3(y_pred_cont, q1, q3)
    results = regression_with_discretization_metrics_dict(
        model_name=STACKING_NAME,
        y_true_cont=y_test_cont,
        y_pred_cont=y_pred_cont,
        q1=q1,
        q3=q3,
    )
    return model, np.asarray(y_pred_cont), np.asarray(y_pred_cls), results


def train_classification_ensembles(context: SecondLevelContext, config: SecondLevelConfig) -> SecondLevelArtifacts:
    assert context.y_train_cls is not None and context.y_test_cls is not None
    classes = np.unique(np.asarray(context.y_train_cls))
    vote_avg_proba, vote_pred_cls, vote_results = train_soft_voting_classifier(
        meta_X_test=context.meta_X_test,
        y_test_cls=context.y_test_cls,
        classes=classes,
        weights=config.classification_weights,
    )
    stack_model, stack_pred_cls, stack_results = train_stacking_classifier(
        meta_X_train=context.meta_X_train,
        meta_X_test=context.meta_X_test,
        y_train_cls=context.y_train_cls,
        y_test_cls=context.y_test_cls,
    )
    return SecondLevelArtifacts(
        models={SOFT_VOTING_NAME: None, STACKING_NAME: stack_model},
        results={SOFT_VOTING_NAME: vote_results, STACKING_NAME: stack_results},
        predictions={SOFT_VOTING_NAME: vote_avg_proba, STACKING_NAME: None},
        predictions_cls={SOFT_VOTING_NAME: vote_pred_cls, STACKING_NAME: stack_pred_cls},
        params={
            SOFT_VOTING_NAME: {"weights": config.classification_weights, "aggregation": "weighted mean of probabilities"},
            STACKING_NAME: {"model_name": "LogisticRegression", "max_iter": 1000, "random_state": RANDOM_STATE},
        },
    )


def train_regression_ensembles(context: SecondLevelContext, config: SecondLevelConfig) -> SecondLevelArtifacts:
    assert context.y_train_cont is not None and context.y_test_cont is not None
    vote_pred_cont, vote_pred_cls, vote_results = train_voting_regressor_from_meta(
        meta_X_test=context.meta_X_test,
        y_test_cont=context.y_test_cont,
        q1=context.q1,
        q3=context.q3,
        weights=config.regression_weights,
    )
    stack_model, stack_pred_cont, stack_pred_cls, stack_results = train_stacking_regressor(
        meta_X_train=context.meta_X_train,
        meta_X_test=context.meta_X_test,
        y_train_cont=context.y_train_cont,
        y_test_cont=context.y_test_cont,
        q1=context.q1,
        q3=context.q3,
    )
    return SecondLevelArtifacts(
        models={VOTING_NAME: None, STACKING_NAME: stack_model},
        results={VOTING_NAME: vote_results, STACKING_NAME: stack_results},
        predictions={VOTING_NAME: vote_pred_cont, STACKING_NAME: stack_pred_cont},
        predictions_cls={VOTING_NAME: vote_pred_cls, STACKING_NAME: stack_pred_cls},
        params={
            VOTING_NAME: {
                "weights": config.regression_weights,
                "aggregation": "weighted mean of regression predictions",
                "used_prediction_columns": vote_results.get("used_regression_prediction_columns", []),
            },
            STACKING_NAME: {"model_name": "Ridge", "alpha": 1.0},
        },
    )


def run_training(context: SecondLevelContext, config: SecondLevelConfig) -> SecondLevelArtifacts:
    if context.strategy_type == CLASSIFICATION_STRATEGY:
        return train_classification_ensembles(context, config)
    return train_regression_ensembles(context, config)
