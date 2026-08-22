# Student Performance Prediction with Ensemble Machine Learning

Streamlit-based machine learning application for student performance prediction, ensemble model training, statistical comparison of model quality, and practical prediction for new student records.

## Overview

The project implements an interactive end-to-end machine learning workflow for student performance modeling. It supports two alternative predictive strategies:

1. **Direct multiclass classification** — the continuous target is discretized into performance classes before model training.
2. **Regression with subsequent categorization** — models predict a continuous performance score, which is then converted into classes using the `Q1` and `Q3` thresholds.

The application combines data preprocessing, Bayesian hyperparameter optimization, first-level models, ensemble learning, model evaluation, statistical analysis, cross-strategy comparison, and practical prediction in a single Streamlit interface.

The current codebase represents the refactored **v2.0** architecture of the project.

## Main Features

- interactive CSV upload and primary dataset analysis;
- preprocessing and train/test split formation;
- exploratory data visualization;
- two predictive strategies: multiclass classification and regression with categorization;
- Bayesian hyperparameter optimization with cross-validation;
- first-level model training;
- meta-feature generation;
- second-level ensemble learning with voting and stacking;
- evaluation using classification and regression metrics;
- export and import of evaluation reports in JSON format;
- confidence intervals for classification metrics;
- paired statistical comparison of models;
- McNemar tests with Holm correction for multiple comparisons;
- paired stratified bootstrap confidence intervals for metric differences;
- comparison of the two predictive strategies;
- practical prediction for new student records.

## Application Workflow

The Streamlit application contains nine sequential stages:

1. **Data loading and primary analysis**
2. **Data preprocessing and split formation**
3. **Visualization**
4. **Training first-level base models**
5. **Meta-feature generation**
6. **Training second-level ensemble models**
7. **Model quality evaluation and JSON export**
8. **Statistical analysis of classification results**
9. **Strategy comparison and practical prediction**

Navigation is implemented in `main.py` with a custom Streamlit sidebar. The project intentionally uses the directory name `app_pages/` instead of Streamlit's special `pages/` directory to avoid duplicate automatic navigation.

## Predictive Strategies

### 1. Direct multiclass classification

The continuous target is transformed into three performance classes before training. The base models are trained directly on the categorical target.

Base models:

- `RidgeClassifier`
- `RandomForestClassifier`
- `XGBClassifier`

Second-level models:

- Soft Voting
- Stacking

### 2. Regression with subsequent categorization

The models first predict the continuous performance score. Predictions are then categorized using the same `Q1` and `Q3` thresholds used to define the target classes.

Base models:

- `Ridge`
- `RandomForestRegressor`
- `XGBRegressor`

Second-level models:

- Voting Regressor
- Stacking Regressor

## Hyperparameter Optimization

Bayesian Optimization is used to select hyperparameters for the first-level models.

- Classification strategy: `StratifiedKFold` cross-validation with F1 macro as the optimization criterion.
- Regression strategy: cross-validation with negative mean squared error as the optimization criterion.

## Model Evaluation

### Classification metrics

- Accuracy
- Balanced Accuracy
- F1 macro
- F1 weighted
- classification report
- confusion matrix

### Regression metrics

- MAE
- MSE
- RMSE
- R²

Regression predictions are additionally categorized so that both predictive strategies can be compared using the same classification metrics.

## Statistical Analysis

The statistical analysis stage uses predictions generated for the same fixed test set. Evaluation reports include the information required to verify test-set alignment between strategies:

- test-set fingerprint;
- ordered test sample identifiers;
- true class labels;
- per-object predictions for every model;
- categorization thresholds.

Implemented statistical procedures include:

- normal-approximation confidence interval for Accuracy;
- stratified percentile bootstrap confidence intervals for Accuracy, Balanced Accuracy, F1 macro, and F1 weighted;
- paired stratified bootstrap confidence intervals for differences between model metrics;
- McNemar test for paired comparison of classification correctness;
- exact two-sided binomial McNemar test for small discordant counts;
- chi-square McNemar test with continuity correction otherwise;
- Holm correction for multiple McNemar comparisons.

Old evaluation JSON files that do not contain the statistical data block are not suitable for paired statistical analysis. Reports should be re-exported from the current evaluation stage.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── DataSet/
│   └── Student_Performance.csv
├── app_pages/
│   ├── __init__.py
│   ├── module_1.py
│   ├── module_2.py
│   ├── module_3.py
│   ├── module_4.py
│   ├── module_5.py
│   ├── module_6.py
│   ├── module_7.py
│   ├── module_8.py
│   └── module_9.py
└── utils/
    ├── __init__.py
    ├── app_state.py
    ├── session_state.py
    ├── preprocessing_core.py
    ├── preprocessing_ui.py
    ├── visualization_helpers.py
    ├── optimization_utils.py
    ├── training_first_level_*.py
    ├── metafeatures_*.py
    ├── second_level_*.py
    ├── evaluation_*.py
    ├── statistical_analysis_*.py
    └── strategy_comparison_*.py
```

The `utils/` package separates computational logic, state management, report generation, constants, and UI helpers. This reduces the size of individual Streamlit page modules and makes the application easier to maintain and extend.

## Dataset

The repository contains `DataSet/Student_Performance.csv`, which can be used to demonstrate the full workflow.

Input features include:

- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

Target variable:

- Performance Index

The application also supports uploading a compatible CSV file through the interface.

## Technologies

- Python 3.12
- Streamlit
- Pandas
- NumPy
- SciPy
- Scikit-learn
- XGBoost
- Bayesian Optimization
- Matplotlib
- Seaborn

## Local Installation

Clone the repository:

```bash
git clone https://github.com/savenkov21/student-performance-ensemble-streamlit.git
cd student-performance-ensemble-streamlit
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run main.py
```

By default, Streamlit opens the local application at `http://localhost:8501`.

## Streamlit Community Cloud

Recommended deployment settings:

```text
Repository: savenkov21/student-performance-ensemble-streamlit
Branch: main
Main file path: main.py
Python version: 3.12
```

`requirements.txt` must remain in the repository root next to `main.py`. After changes are committed and pushed to the tracked GitHub branch, Streamlit Community Cloud can rebuild the application from the updated repository state.

## Version 2.0

Compared with the initial public version, v2.0 introduces:

- refactoring from a flat module structure to `app_pages/` and `utils/` packages;
- separated core logic, UI, constants, reports, and state management;
- a dedicated statistical analysis stage;
- test-set fingerprinting and per-object prediction export;
- confidence interval estimation;
- McNemar tests and Holm multiple-comparison correction;
- paired bootstrap comparison of model metrics;
- expanded cross-strategy analysis and prediction workflow.

## Intended Use

This project is intended for academic, research, and demonstration purposes in student performance modeling and ensemble machine learning.

## Author

Valeriia Savenko
