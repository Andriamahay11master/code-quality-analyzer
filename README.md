# Code Quality Assessment Model

An AI/ML project that analyzes software metrics to predict code quality.  
This project explores how static code features and software metrics can be used to build a machine learning model that estimates the quality of source code.

---

## Project Overview

Software quality is an important part of software engineering, especially when trying to detect maintainability issues, code smells, and poor design patterns.  
This project aims to use machine learning to classify or estimate code quality based on measurable code metrics.

The workflow includes:

- Loading and exploring the dataset.
- Cleaning and preprocessing the data.
- Engineering relevant features.
- Training machine learning models.
- Evaluating performance with standard metrics.
- Saving the final model for reuse.

---

## Objectives

- Build an AI model for code quality assessment.
- Analyze the relationship between software metrics and code quality.
- Compare different machine learning algorithms.
- Create a reproducible pipeline for experimentation and evaluation.
- Produce a GitHub-ready project that demonstrates AI/ML and software engineering skills.

---

## Dataset

This project uses a software code quality dataset containing repository, version, and metric-related information.

Depending on the version of the project, the dataset may include:

- Repository-level information.
- Version-level information.
- Code quality metrics.
- Maintainability-related attributes.
- Structural and complexity indicators.

If the dataset is too large to store directly in GitHub, only a small sample or processed version should be included in the repository. The full dataset can be downloaded separately and placed in the `data/raw/` folder.

---

## Problem Type

This project can be formulated as:

- **Classification**, if the target is a quality class.
- **Regression**, if the target is a numeric quality score.

In the first version of the project, a classification approach has been chosen for simplicity and clearer interpretation.

---

## Project Structure

```text
code-quality-assessment-model/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── trained_model.pkl
│   └── label_encoder.pkl
├── reports/
│   ├── figures/
│   └── metrics/
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_prediction.py
└── docs/
    ├── project_overview.md
    └── dataset_description.md
```

---

## Methodology

### 1. Data Exploration

The dataset is loaded and inspected to understand:

- Column names.
- Missing values.
- Data types.
- Distributions of features.
- Target label availability.

### 2. Data Cleaning

The preprocessing step includes:

- Removing duplicates.
- Handling missing values.
- Converting data types.
- Encoding categorical variables.
- Normalizing or scaling features if needed.

### 3. Feature Engineering

Relevant software metrics are selected or derived to improve model learning.  
Examples may include:

- Complexity metrics.
- Object-oriented metrics.
- Size metrics.
- Coupling metrics.

### 4. Model Training

Several machine learning models can be trained and compared, such as:

- Logistic Regression.
- Random Forest.
- Decision Tree.
- XGBoost or Gradient Boosting.

### 5. Model Evaluation

The model is evaluated using metrics such as:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion Matrix.

### 6. Model Saving

The best-performing model is saved for future prediction and reuse.

---

## Tools and Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/code-quality-assessment-model.git
cd code-quality-assessment-model
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Place the dataset

Put the original dataset in:

```text
data/raw/
```

### 2. Run exploration

Open and run:

```text
notebooks/01_data_exploration.ipynb
```

### 3. Preprocess the data

Run:

```text
notebooks/02_preprocessing.ipynb
```

### 4. Train the model

Run:

```text
notebooks/03_model_training.ipynb
```

### 5. Evaluate the results

Run:

```text
notebooks/04_model_evaluation.ipynb
```

### 6. Make predictions

If a prediction script is included, run:

```bash
python src/predict.py
```

---

## Results

The project will generate:

- Cleaned and processed data.
- Trained machine learning model.
- Evaluation metrics.
- Visualizations of performance.
- Feature importance analysis.

Results will be stored in the `reports/` and `models/` folders.

---

## Future Improvements

- Add more datasets for better generalization.
- Extend the model to support multiple programming languages.
- Improve feature engineering with static analysis tools.
- Build a web interface for real-time code quality prediction.
- Deploy the model as an API.

---

## Repository Purpose

This repository is part of my AI/ML portfolio and demonstrates practical skills in:

- Machine learning.
- Data preprocessing.
- Model evaluation.
- Software quality analysis.
- Reproducible project organization.

---

## Author

**Henikaja Andriamahay IRIMANANA**
