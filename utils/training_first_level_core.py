from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
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
from xgboost import XGBClassifier, XGBRegressor

from utils.optimization_utils import optimize_first_level_models
from utils.training_first_level_constants import (
    CLASSIFICATION_STRATEGY,
    REGRESSION_STRATEGY,
    OptimizationConfig,
    SelectedHyperparameters,
    TrainingArtifacts,
    TrainingContext,
)
from utils.training_first_level_state import save_training_artifacts


def discretize_g3(y: Any, q1: float, q3: float) -> np.ndarray:
    y_arr = np.asarray(y)
    classes = np.zeros_like(y_arr, dtype=int)
    classes[y_arr <= q1] = 0
    classes[(y_arr > q1) & (y_arr <= q3)] = 1
    classes[y_arr > q3] = 2
    return classes


def safe_classification_report(y_true: Any, y_pred: Any) -> dict[str, Any]:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def get_selected_hyperparameters(optimization_results: dict[str, Any]) -> SelectedHyperparameters:
    return SelectedHyperparameters(
        ridge_alpha=float(optimization_results["Ridge"]["best_params"]["alpha"]),
        rf_params=dict(optimization_results["RandomForest"]["best_params"]),
        xgb_params=dict(optimization_results["XGBoost"]["best_params"]),
    )


def build_classification_models(params: SelectedHyperparameters) -> dict[str, Any]:
    return {
        "RidgeClassifier": RidgeClassifier(alpha=params.ridge_alpha),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=int(params.rf_params["n_estimators"]),
            max_depth=params.rf_params["max_depth"],
            min_samples_split=int(params.rf_params["min_samples_split"]),
            min_samples_leaf=int(params.rf_params["min_samples_leaf"]),
            random_state=42,
            n_jobs=-1,
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=int(params.xgb_params["n_estimators"]),
            max_depth=int(params.xgb_params["max_depth"]),
            learning_rate=float(params.xgb_params["learning_rate"]),
            subsample=float(params.xgb_params["subsample"]),
            colsample_bytree=float(params.xgb_params["colsample_bytree"]),
            min_child_weight=int(params.xgb_params["min_child_weight"]),
            gamma=float(params.xgb_params["gamma"]),
            reg_lambda=float(params.xgb_params["reg_lambda"]),
            reg_alpha=float(params.xgb_params["reg_alpha"]),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }


def build_regression_models(params: SelectedHyperparameters) -> dict[str, Any]:
    return {
        "Ridge": Ridge(alpha=params.ridge_alpha),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=int(params.rf_params["n_estimators"]),
            max_depth=params.rf_params["max_depth"],
            min_samples_split=int(params.rf_params["min_samples_split"]),
            min_samples_leaf=int(params.rf_params["min_samples_leaf"]),
            random_state=42,
            n_jobs=-1,
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=int(params.xgb_params["n_estimators"]),
            max_depth=int(params.xgb_params["max_depth"]),
            learning_rate=float(params.xgb_params["learning_rate"]),
            subsample=float(params.xgb_params["subsample"]),
            colsample_bytree=float(params.xgb_params["colsample_bytree"]),
            min_child_weight=int(params.xgb_params["min_child_weight"]),
            gamma=float(params.xgb_params["gamma"]),
            reg_lambda=float(params.xgb_params["reg_lambda"]),
            reg_alpha=float(params.xgb_params["reg_alpha"]),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_single_classifier(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)]))
    result = {
        "model_name": model_name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "F1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": safe_classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classes": labels.tolist(),
    }
    return model, np.asarray(y_pred), result


def train_single_regressor_with_categorization(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_cont: pd.Series,
    y_test_cont: pd.Series,
    q1: float,
    q3: float,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    model.fit(X_train, y_train_cont)
    y_pred_cont = model.predict(X_test)
    mse = mean_squared_error(y_test_cont, y_pred_cont)
    y_test_cls = discretize_g3(y_test_cont, q1, q3)
    y_pred_cls = discretize_g3(y_pred_cont, q1, q3)
    labels_cls = np.unique(np.concatenate([np.asarray(y_test_cls), np.asarray(y_pred_cls)]))
    result = {
        "model_name": model_name,
        "MAE": float(mean_absolute_error(y_test_cont, y_pred_cont)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test_cont, y_pred_cont)),
        "Accuracy": float(accuracy_score(y_test_cls, y_pred_cls)),
        "Balanced_accuracy": float(balanced_accuracy_score(y_test_cls, y_pred_cls)),
        "F1_weighted": float(f1_score(y_test_cls, y_pred_cls, average="weighted")),
        "classification_report": safe_classification_report(y_test_cls, y_pred_cls),
        "confusion_matrix": confusion_matrix(y_test_cls, y_pred_cls, labels=labels_cls).tolist(),
        "classes": labels_cls.tolist(),
        "y_true_cont": np.asarray(y_test_cont, dtype=float).tolist(),
        "y_pred_cont": np.asarray(y_pred_cont, dtype=float).tolist(),
    }
    return model, np.asarray(y_pred_cont), np.asarray(y_pred_cls), result


def train_models_for_strategy(context: TrainingContext, params: SelectedHyperparameters) -> TrainingArtifacts:
    trained_models: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    predictions_cls: dict[str, np.ndarray] = {}

    if context.strategy_type == CLASSIFICATION_STRATEGY:
        models = build_classification_models(params)
        for model_name, model in models.items():
            trained_model, y_pred, result = train_single_classifier(
                model=model,
                model_name=model_name,
                X_train=context.X_train,
                X_test=context.X_test,
                y_train=context.y_train,
                y_test=context.y_test,
            )
            trained_models[model_name] = trained_model
            results[model_name] = result
            predictions[model_name] = np.asarray(y_pred)
            predictions_cls[model_name] = np.asarray(y_pred)
    else:
        models = build_regression_models(params)
        for model_name, model in models.items():
            trained_model, y_pred_cont, y_pred_cls, result = train_single_regressor_with_categorization(
                model=model,
                model_name=model_name,
                X_train=context.X_train,
                X_test=context.X_test,
                y_train_cont=context.y_train_cont,
                y_test_cont=context.y_test_cont,
                q1=context.q1,
                q3=context.q3,
            )
            trained_models[model_name] = trained_model
            results[model_name] = result
            predictions[model_name] = np.asarray(y_pred_cont)
            predictions_cls[model_name] = np.asarray(y_pred_cls)

    return TrainingArtifacts(
        models=trained_models,
        results=results,
        predictions=predictions,
        predictions_cls=predictions_cls,
        params={
            "source": "bayesian_optimization",
            "ridge_alpha": params.ridge_alpha,
            "rf_params": params.rf_params,
            "xgb_params": params.xgb_params,
        },
    )


def run_optimization(context: TrainingContext, config: OptimizationConfig) -> None:
    with st.spinner("Триває Bayesian Optimization..."):
        st.session_state.optimization_results = optimize_first_level_models(
            strategy_type=context.strategy_type,
            X_train=context.X_train,
            y_train=context.y_train,
            y_train_cont=context.y_train_cont,
            init_points=config.init_points,
            n_iter=config.n_iter,
        )
    st.success("Оптимізацію гіперпараметрів завершено.")


def run_first_level_training(context: TrainingContext, optimization_results: dict[str, Any] | None) -> None:
    if optimization_results is None:
        st.warning("Спочатку виконайте Bayesian Optimization.")
        return

    params = get_selected_hyperparameters(optimization_results)
    with st.spinner("Триває навчання базових моделей..."):
        artifacts = train_models_for_strategy(context, params)
        save_training_artifacts(context, artifacts)
    st.success("Базові моделі першого рівня успішно навчені.")
