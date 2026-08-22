import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from utils.constants_dataclasses import SplitArtifacts, SplitConfig


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def discretize_g3(y: pd.Series | np.ndarray, q1: float, q3: float) -> np.ndarray:
    y_arr = np.asarray(y)
    classes = np.zeros_like(y_arr, dtype=int)
    classes[(y_arr > q1) & (y_arr <= q3)] = 1
    classes[y_arr > q3] = 2
    return classes


def make_stratify_bins(y: pd.Series, n_bins: int = 3) -> pd.Series | None:
    y = pd.Series(y).reset_index(drop=True)

    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None

    bins = pd.Series(bins)
    if bins.nunique(dropna=True) < 2:
        return None

    min_count = bins.value_counts().min()
    if pd.isna(min_count) or min_count < 2:
        return None

    return bins


def get_scaler(scaler_type: str):
    if scaler_type == "StandardScaler":
        return StandardScaler()
    if scaler_type == "MinMaxScaler":
        return MinMaxScaler()
    return "passthrough"


def build_preprocessor(
    X_train_raw: pd.DataFrame,
    *,
    scaler_type: str,
    numeric_imputer_strategy: str,
    categorical_imputer_strategy: str,
    onehot_drop: str,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = numeric_columns(X_train_raw)
    categorical_cols = categorical_columns(X_train_raw)

    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer_strategy))]
    scaler = get_scaler(scaler_type)
    if scaler != "passthrough":
        numeric_steps.append(("scaler", scaler))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_imputer_strategy)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None if onehot_drop == "none" else "if_binary",
                ),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    if not transformers:
        raise ValueError("У X немає доступних ознак для preprocessing.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def transform_with_preprocessor(
    preprocessor: ColumnTransformer,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index)
    return X_train, X_test, feature_names


def prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_col]).copy(), df[target_col].copy()


def validate_split_inputs(X_raw: pd.DataFrame, y_cont: pd.Series) -> list[str]:
    errors: list[str] = []

    if X_raw.empty:
        errors.append("Матриця ознак X порожня. Потрібен хоча б один предиктор.")
    if y_cont.isna().any():
        errors.append("Цільова змінна містить пропущені значення.")
    if not pd.api.types.is_numeric_dtype(y_cont):
        errors.append(
            "Для двох стратегій цільова ознака має бути числовою "
            "(щоб можна було обчислити квартилі Q1 і Q3)."
        )

    return errors


def build_split_artifacts(df: pd.DataFrame, config: SplitConfig) -> SplitArtifacts:
    X_raw, y_cont = prepare_xy(df, config.target_col)
    errors = validate_split_inputs(X_raw, y_cont)
    if errors:
        raise ValueError("\n".join(errors))

    stratify_labels = make_stratify_bins(y_cont, n_bins=3)
    stratify_used = stratify_labels is not None

    X_train_raw, X_test_raw, y_train_cont, y_test_cont = train_test_split(
        X_raw,
        y_cont,
        test_size=float(config.test_size),
        random_state=int(config.random_state),
        stratify=stratify_labels if stratify_used else None,
    )

    q1 = float(np.percentile(y_train_cont, 25))
    q3 = float(np.percentile(y_train_cont, 75))

    y_train_cls = pd.Series(discretize_g3(y_train_cont, q1, q3), index=X_train_raw.index)
    y_test_cls = pd.Series(discretize_g3(y_test_cont, q1, q3), index=X_test_raw.index)

    preprocessor, num_cols, cat_cols = build_preprocessor(
        X_train_raw,
        scaler_type=config.scaler_type,
        numeric_imputer_strategy=config.numeric_imputer_strategy,
        categorical_imputer_strategy=config.categorical_imputer_strategy,
        onehot_drop=config.onehot_drop,
    )
    X_train, X_test, feature_names = transform_with_preprocessor(preprocessor, X_train_raw, X_test_raw)

    if config.strategy_type == "direct_multiclass_classification":
        y_train = y_train_cls.copy()
        y_test = y_test_cls.copy()
        y_active = pd.concat([y_train, y_test]).sort_index()
    else:
        y_train = pd.Series(y_train_cont, index=X_train.index)
        y_test = pd.Series(y_test_cont, index=X_test.index)
        y_active = pd.Series(y_cont).copy()

    preprocessing_config = {
        "scaler_type": config.scaler_type,
        "numeric_imputer_strategy": config.numeric_imputer_strategy,
        "categorical_imputer_strategy": config.categorical_imputer_strategy,
        "onehot_drop": config.onehot_drop,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "n_output_features": len(feature_names),
    }

    return SplitArtifacts(
        strategy_type=config.strategy_type,
        target_col=config.target_col,
        feature_cols=[col for col in df.columns if col != config.target_col],
        feature_names_after_preprocessing=feature_names,
        preprocessing_config=preprocessing_config,
        preprocessor=preprocessor,
        X_raw=X_raw,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train=X_train,
        X_test=X_test,
        y_active=y_active,
        y_train=y_train,
        y_test=y_test,
        y_cont=pd.Series(y_cont).copy(),
        y_train_cont=pd.Series(y_train_cont, index=X_train.index),
        y_test_cont=pd.Series(y_test_cont, index=X_test.index),
        y_train_cls=y_train_cls,
        y_test_cls=y_test_cls,
        q1=q1,
        q3=q3,
        test_size=float(config.test_size),
        random_state=int(config.random_state),
        stratify_used=stratify_used,
    )
