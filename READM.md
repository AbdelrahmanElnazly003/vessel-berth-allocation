# Vessel Berth Allocation System

## Overview

This project addresses the vessel berth allocation problem in port operations.
The objective is to predict the most suitable berth for an incoming vessel **before arrival**,
based on historical vessel and terminal data.

An end-to-end machine learning pipeline was developed, covering data preprocessing,
model training, evaluation, and deployment through a RESTful API to support operational
decision-making.

---

## Dataset

The dataset consists of historical vessel arrival records and contains a mix of
categorical and numerical features related to vessel characteristics, berth properties,
and operational productivity metrics.

- **Target variable:** `BerthID`
- **Features:** vessel type, service information, berth length, productivity indicators, and operational attributes
- Post-arrival and leakage-prone features were removed to ensure realistic pre-arrival prediction

---
## Project Structure
```text
vessel-berth-allocation/
│
├── api/                    # FastAPI application
│├── app.py
│
├── model/                  # Trained model
│├── logistic_regression_pipeline.pkl
│
├── data/                   # Sample input for API testing
│├── sample_input.csv
│
├── notebooks/              # EDA & modeling notebooks
│   ├── notebook_1_data_cleaning.ipynb
│   └── notebook_2_modeling.ipynb
│
├── requirements.txt
└── README.md
```
## Methodology

The modeling pipeline follows these steps:

- Data cleaning and preprocessing
- Feature selection with leakage prevention
- Encoding of categorical variables
- Handling class imbalance using class-weighted loss functions
- Stratified train-test split
- Baseline modeling using a Dummy Classifier
- Model training and evaluation using Logistic Regression and Random Forest
- Cross-validation for model stability assessment

Macro F1-score was used as the primary evaluation metric to account for class imbalance.

## Model Evaluation & Results

Models were evaluated using:
- Accuracy
- Macro F1-score (primary metric due to class imbalance)
- Confusion Matrix
- Top-3 Accuracy (to reflect real-world decision support)

Both Logistic Regression and Random Forest achieved comparable performance,
with stable cross-validation results and consistent misclassification patterns.

## Final Model Selection

Logistic Regression was selected as the final model due to:

- Comparable performance to Random Forest
- Better interpretability
- Lower computational complexity
- Easier integration into production APIs

More complex models did not provide significant performance gains to justify
additional complexity.

## API Deployment

A RESTful API was built using FastAPI to simulate real-world deployment.

### Endpoint
POST /predict

### Input
CSV file containing vessel features (excluding target)

### Output
JSON response including:
- Top-1 predicted berth
- Top-3 recommended berths for operational flexibility

## Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Jupyter Notebook

## Future Improvements
- Integrate real-time berth availability data to ensure predictions consider current terminal capacity.
- Include additional operational features such as weather conditions or congestion indicators.
- Improve model explainability by highlighting the most influential features behind each prediction.
- Deploy the API using Docker to simplify setup and deployment across environments.
- Extend the API to support batch predictions for multiple vessels at once.
- Explore optimization-based or hybrid approaches to recommend the best berth assignment under operational constraints.


## Conclusion

This project demonstrates a complete machine learning workflow, from data preprocessing
and model evaluation to deployment through a production-ready API.

The system provides practical decision support for port operations and can be extended
to handle real-time constraints and optimization-based allocation strategies.

