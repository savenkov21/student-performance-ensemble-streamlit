from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from utils.metafeatures_constants import (
    CLASSIFICATION_STRATEGY,
    RANDOM_STATE,
    REGRESSION_STRATEGY,
    MetaFeatureArtifacts,
    MetaFeatureConfig,
    MetaFeatureContext,
)


def get_classification_models(ridge_alpha, rf_params, xgb_params) -> Dict[str, object]:
    ridge_base = RidgeClassifier(alpha=float(ridge_alpha))
    ridge_calibrated = CalibratedClassifierCV(estimator=ridge_base, cv=5)

    return {
        "RidgeClassifier": ridge_calibrated,
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=int(rf_params["n_estimators"]),
            max_depth=rf_params["max_depth"],
            min_samples_split=int(rf_params["min_samples_split"]),
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
            min_child_weight=int(xgb_params["min_child_weight"]),
            gamma=float(xgb_params["gamma"]),
            reg_lambda=float(xgb_params["reg_lambda"]),
            reg_alpha=float(xgb_params["reg_alpha"]),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def get_regression_models(ridge_alpha, rf_params, xgb_params) -> Dict[str, object]:
    return {
        "Ridge": Ridge(alpha=float(ridge_alpha)),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=int(rf_params["n_estimators"]),
            max_depth=rf_params["max_depth"],
            min_samples_split=int(rf_params["min_samples_split"]),
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
            min_child_weight=int(xgb_params["min_child_weight"]),
            gamma=float(xgb_params["gamma"]),
            reg_lambda=float(xgb_params["reg_lambda"]),
            reg_alpha=float(xgb_params["reg_alpha"]),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_base_models(strategy_type: str, params: Mapping[str, object]) -> Dict[str, object]:
    ridge_alpha = params["ridge_alpha"]
    rf_params = params["rf_params"]
    xgb_params = params["xgb_params"]

    if strategy_type == CLASSIFICATION_STRATEGY:
        return get_classification_models(ridge_alpha, rf_params, xgb_params)
    if strategy_type == REGRESSION_STRATEGY:
        return get_regression_models(ridge_alpha, rf_params, xgb_params)
    raise ValueError(f"Невідома стратегія: {strategy_type}")


def predict_for_meta_classification(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X), dtype=float)


def predict_for_meta_regression(model, X) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def generate_oof_metafeatures_classification(
    models: Mapping[str, object],
    X_train,
    y_train,
    X_test,
    cv: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], List[str]]:
    X_train_np = np.asarray(X_train)
    X_test_np = np.asarray(X_test)
    y_train_np = np.asarray(y_train)

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    classes_ = np.unique(y_train_np)
    n_classes = len(classes_)

    meta_train_blocks = []
    meta_test_blocks = []
    feature_names: List[str] = []
    fitted_models_on_full_train: Dict[str, object] = {}

    for model_name, base_model in models.items():
        oof_pred = np.zeros((len(X_train_np), n_classes), dtype=float)
        test_fold_preds = []

        for train_idx, valid_idx in splitter.split(X_train_np, y_train_np):
            X_tr = X_train_np[train_idx]
            X_val = X_train_np[valid_idx]
            y_tr = y_train_np[train_idx]

            model = clone(base_model)
            model.fit(X_tr, y_tr)

            oof_pred[valid_idx, :] = predict_for_meta_classification(model, X_val)
            test_fold_preds.append(predict_for_meta_classification(model, X_test_np))

        meta_train_blocks.append(oof_pred)
        meta_test_blocks.append(np.mean(np.stack(test_fold_preds, axis=0), axis=0))
        feature_names.extend([f"meta_{model_name}_proba_{cls}" for cls in classes_])

        full_model = clone(base_model)
        full_model.fit(X_train_np, y_train_np)
        fitted_models_on_full_train[model_name] = full_model

    meta_X_train = pd.DataFrame(
        np.hstack(meta_train_blocks),
        columns=feature_names,
        index=getattr(X_train, "index", None),
    )
    meta_X_test = pd.DataFrame(
        np.hstack(meta_test_blocks),
        columns=feature_names,
        index=getattr(X_test, "index", None),
    )
    return meta_X_train, meta_X_test, fitted_models_on_full_train, feature_names


def generate_oof_metafeatures_regression(
    models: Mapping[str, object],
    X_train,
    y_train,
    X_test,
    cv: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], List[str]]:
    n_train = len(X_train)
    n_test = len(X_test)
    n_models = len(models)

    meta_train = np.zeros((n_train, n_models), dtype=float)
    meta_test = np.zeros((n_test, n_models), dtype=float)

    splitter = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    X_train_np = np.asarray(X_train)
    X_test_np = np.asarray(X_test)
    y_train_np = np.asarray(y_train)

    fitted_models_on_full_train: Dict[str, object] = {}
    feature_names: List[str] = []

    for model_idx, (model_name, base_model) in enumerate(models.items()):
        oof_pred = np.zeros(n_train, dtype=float)
        test_fold_preds = []

        for train_idx, valid_idx in splitter.split(X_train_np):
            X_tr = X_train_np[train_idx]
            X_val = X_train_np[valid_idx]
            y_tr = y_train_np[train_idx]

            model = clone(base_model)
            model.fit(X_tr, y_tr)

            oof_pred[valid_idx] = predict_for_meta_regression(model, X_val)
            test_fold_preds.append(predict_for_meta_regression(model, X_test_np))

        meta_train[:, model_idx] = oof_pred
        meta_test[:, model_idx] = np.mean(np.vstack(test_fold_preds), axis=0)
        feature_names.append(f"meta_{model_name}")

        full_model = clone(base_model)
        full_model.fit(X_train_np, y_train_np)
        fitted_models_on_full_train[model_name] = full_model

    meta_X_train = pd.DataFrame(meta_train, columns=feature_names, index=getattr(X_train, "index", None))
    meta_X_test = pd.DataFrame(meta_test, columns=feature_names, index=getattr(X_test, "index", None))
    return meta_X_train, meta_X_test, fitted_models_on_full_train, feature_names


def append_original_features(
    meta_X_train: pd.DataFrame,
    meta_X_test: pd.DataFrame,
    X_train,
    X_test,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
    meta_train_df = meta_X_train.reset_index(drop=True)
    meta_test_df = meta_X_test.reset_index(drop=True)

    combined_train = pd.concat([X_train_df, meta_train_df], axis=1)
    combined_test = pd.concat([X_test_df, meta_test_df], axis=1)
    return combined_train, combined_test, combined_train.columns.tolist()


def build_metafeatures(context: MetaFeatureContext, config: MetaFeatureConfig) -> MetaFeatureArtifacts:
    base_models = build_base_models(context.strategy_type, context.params)

    if context.strategy_type == CLASSIFICATION_STRATEGY:
        target = context.y_train
        meta_X_train, meta_X_test, full_train_models, feature_names = generate_oof_metafeatures_classification(
            models=base_models,
            X_train=context.X_train,
            y_train=target,
            X_test=context.X_test,
            cv=config.cv_folds,
        )
    else:
        target = context.y_train_cont
        meta_X_train, meta_X_test, full_train_models, feature_names = generate_oof_metafeatures_regression(
            models=base_models,
            X_train=context.X_train,
            y_train=target,
            X_test=context.X_test,
            cv=config.cv_folds,
        )

    if config.passthrough:
        meta_X_train, meta_X_test, feature_names = append_original_features(
            meta_X_train=meta_X_train,
            meta_X_test=meta_X_test,
            X_train=context.X_train,
            X_test=context.X_test,
        )

    return MetaFeatureArtifacts(
        meta_X_train=meta_X_train,
        meta_X_test=meta_X_test,
        feature_names=feature_names,
        base_model_names=list(base_models.keys()),
        full_train_models=full_train_models,
        generation_params={
            "cv_folds": int(config.cv_folds),
            "passthrough": bool(config.passthrough),
        },
    )
