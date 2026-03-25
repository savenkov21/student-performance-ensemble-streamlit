# Development of a Student Rating Evaluation Model Based on Ensemble Machine Learning Methods

Streamlit-based application for student performance prediction and comparative analysis of ensemble machine learning strategies.

## Overview

This project implements an interactive web application for modeling student performance using ensemble machine learning methods. The system supports the full experimental workflow: data loading, preprocessing, visualization, hyperparameter optimization, training of first-level models, meta-feature generation, second-level ensemble learning, evaluation, and cross-strategy comparison.

The application is built around two predictive strategies:

1. **Direct multiclass classification** – the target variable is discretized into classes before model training.
2. **Regression with subsequent categorization** – continuous predictions are obtained first and then transformed into categories using threshold-based discretization.

The project is intended for educational and research use in the context of student performance analysis and ensemble modeling.

## Main Features

- interactive data upload and primary dataset analysis;
- preprocessing pipeline with target selection and train/test split formation;
- exploratory data visualization;
- training of first-level base models;
- generation of meta-features for stacked learning;
- training of second-level ensemble models;
- evaluation of model quality with classification and regression metrics;
- comparison of alternative forecasting strategies in a unified interface;
- support for Bayesian optimization of hyperparameters.

## Implemented Workflow

The Streamlit application consists of the following stages:

1. **Data loading and primary analysis**
2. **Data preprocessing and split formation**
3. **Visualization**
4. **Training first-level base models**
5. **Meta-feature formation**
6. **Training second-level models**
7. **Evaluation of model quality**
8. **Comparison of forecasting strategies**

These stages are connected through a multi-page navigation interface defined in `main.py`.

## Project Structure

```text
.
├── main.py
├── module_1.py              # data loading and primary analysis
├── module_2.py              # preprocessing and train/test split formation
├── module_3.py              # visualization
├── module_4.py              # first-level model training
├── module_5.py              # meta-feature generation
├── module_6.py              # second-level model training
├── module_7.py              # evaluation of results
├── module_8.py              # comparison of strategies
├── optimization_utils.py    # Bayesian optimization utilities
├── Student_Performance.csv  # sample dataset
└── README.md
```

## Models and Methods

### First-level models

Depending on the selected strategy, the system uses a set of base models such as:

- `RandomForestClassifier` / `RandomForestRegressor`
- `XGBClassifier` / `XGBRegressor`
- `RidgeClassifier` / `Ridge`

### Second-level ensemble models

The project supports ensemble learning at the meta-level, including:

- soft voting / weighted voting;
- stacking-based aggregation.

### Hyperparameter optimization

Bayesian Optimization is used for automatic hyperparameter search for selected models. Cross-validation is integrated into the optimization procedure.

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

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/student-rating-ensemble-model.git
cd student-rating-ensemble-model
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

## Run the application

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

## Possible Improvements

- refactoring repeated helper functions into dedicated `utils/` modules;
- separating data, core logic, UI helpers, and experiment artifacts into subdirectories;
- adding exportable experiment logs and configuration tracking;
- adding screenshots of the interface to the repository documentation;
- deploying the application via Streamlit Community Cloud or another hosting platform.

## Repository Description

**Streamlit-based machine learning project for student performance evaluation using ensemble methods, including stacking, voting, Bayesian optimization, preprocessing, visualization, and comparative analysis of classification and regression strategies.**

## Author

Valeriia Savenko

## License

This project is intended for academic and research purposes.
