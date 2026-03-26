# Development of a Student Rating Evaluation Model Based on Ensemble Machine Learning Methods

Streamlit-based application for student performance prediction, comparative analysis of ensemble learning strategies, and practical forecasting for new student records.

## Overview

This project implements an interactive web application for predicting student performance using ensemble machine learning methods. The system supports the full workflow of an applied machine learning study: data loading, preprocessing, visualization, hyperparameter optimization, training of first-level models, meta-feature generation, second-level ensemble learning, evaluation, comparison of alternative predictive strategies, and practical prediction for new students.

The application is built around two predictive strategies:

1. **Direct multiclass classification** – the target variable is discretized into classes before model training.
2. **Regression with subsequent categorization** – the model first predicts a continuous score, after which the result is transformed into categorical levels using threshold-based discretization.

The project is intended for academic, research, and demonstration use in the domain of student performance modeling.

## Main Features

- interactive data upload and primary dataset analysis;
- preprocessing pipeline with target selection and train/test split formation;
- exploratory data visualization;
- training of first-level base models;
- meta-feature generation for ensemble learning;
- training of second-level ensemble models;
- Bayesian optimization of hyperparameters with cross-validation;
- evaluation using classification and regression metrics;
- comparison of two predictive strategies in a unified interface;
- JSON export/import of experimental results;
- practical prediction for a new student record using the currently trained final model.

## Implemented Workflow

The Streamlit application consists of the following stages:

1. **Data loading and primary analysis**
2. **Data preprocessing and split formation**
3. **Visualization**
4. **Training first-level base models**
5. **Meta-feature generation**
6. **Training second-level ensemble models**
7. **Evaluation of model quality**
8. **Comparison of strategies and practical prediction**

These stages are connected through a multi-page interface defined in `main.py`.

## Project Structure

```text
.
├── main.py
├── module_1.py              # data loading and primary analysis
├── module_2.py              # preprocessing and train/test split formation
├── module_3.py              # visualization
├── module_4.py              # first-level model training
├── module_5.py              # meta-feature generation
├── module_6.py              # second-level ensemble training
├── module_7.py              # evaluation of results
├── module_8.py              # strategy comparison, JSON-based analysis, practical prediction
├── optimization_utils.py    # Bayesian optimization utilities
├── Student_Performance.csv  # sample dataset
└── README.md
```

## Predictive Strategies

### 1. Direct multiclass classification

The continuous target is discretized into classes before model training. Classification models are then trained directly on the derived labels.

### 2. Regression with subsequent categorization

Regression models first predict a continuous student score. The output is then categorized into performance levels using predefined thresholds such as `Q1` and `Q3`.

## Models and Methods

### First-level models

Depending on the selected strategy, the system uses base models such as:

- `RandomForestClassifier` / `RandomForestRegressor`
- `XGBClassifier` / `XGBRegressor`
- `RidgeClassifier` / `Ridge`

### Second-level ensemble models

The project supports meta-level ensemble learning, including:

- **Soft Voting / Voting**
- **Stacking**

### Hyperparameter optimization

Bayesian Optimization is used for automatic hyperparameter search. Cross-validation is integrated into the optimization procedure.

## Practical Prediction

The application includes a dedicated interface for practical prediction of student performance based on the currently trained models.

This functionality supports:

- fixing the final model for inference;
- manual input of a new student’s feature values;
- automatic preprocessing of new data using the fitted preprocessing pipeline;
- generation of final predictions:
  - predicted class for the classification strategy;
  - predicted score and categorized class for the regression strategy;
- display of class probabilities for classification models.

## Result Analysis

The application supports:

- comparison tables for base and ensemble models;
- classification reports;
- confusion matrices;
- regression diagnostics (`MAE`, `MSE`, `RMSE`, `R²`);
- `Actual vs Predicted` plots for regression models;
- cross-strategy comparison based on final metrics.

## Technologies

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib**
- **Seaborn**
- **bayesian-optimization**

## Dataset

The repository includes the dataset `Student_Performance.csv`, which is used for experiments on student performance prediction.

Typical input features include:

- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

Target variable:

- Performance Index

## Installation

Clone the repository:

```bash
git clone https://github.com/savenkov21/student-performance-ensemble-streamlit.git
cd student-performance-ensemble-streamlit
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run main.py
```

## Recommended `requirements.txt`

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
bayesian-optimization
```

## Usage Scenario

1. Upload and inspect the dataset.
2. Perform preprocessing and create train/test splits.
3. Train first-level models.
4. Generate meta-features.
5. Train second-level ensemble models.
6. Evaluate model quality.
7. Compare the two predictive strategies.
8. Select the final model and run practical prediction for a new student.

## Repository Description

**Streamlit-based machine learning project for student performance evaluation using ensemble methods, including preprocessing, visualization, Bayesian optimization, voting, stacking, strategy comparison, and practical prediction for new student records.**

## Author

**Valeriia Savenko**

## License

This project is intended for academic and research purposes.
