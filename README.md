# Credit Default Prediction System
End-to-end machine learning system for predicting credit default risk, with a production-style Flask API for real-time inference.

## Overview

This project develops an end-to-end machine learning system to predict credit default risk using structured financial data. It integrates data preprocessing, model training, and deployment through a Flask-based web application and REST API.

## Problem

Credit default risk is a critical challenge in financial services. Lenders need reliable tools to identify high-risk borrowers and reduce potential losses.

## Approach

* Built a dynamic preprocessing pipeline handling numeric and categorical features
* Applied Target Encoding for high-cardinality categorical variables
* Trained Logistic Regression model with hyperparameter tuning (RandomizedSearchCV)
* Evaluated using recall-focused metrics to handle class imbalance

## System Architecture

* Machine learning pipeline (scikit-learn)
* Saved model pipeline (model.pkl)
* Flask web application (UI + API)
* Real-time prediction endpoint (/api/predict)

## Results

* Recall: 0.615
* Precision: 0.293
* Accuracy: 0.635

## Demo

The system supports both web interface and API-based prediction.

### Web UI
Users can input features through a form and receive real-time predictions.

### API Example
```bash
curl -X POST http://127.0.0.1:5001/api/predict \
-H "Content-Type: application/json" \
-d '{"loanAmnt": 10000, "interestRate": 12.5, ...}'
## Key Features

* End-to-end ML pipeline (training → deployment)
* Handles missing values and categorical encoding automatically
* Web interface for manual input prediction
* REST API for programmatic access

## Tech Stack

Python, pandas, scikit-learn, Flask

## How to Run

```bash
python train_model.py
python app.py
```

## Project Structure

```
credit-default-prediction/
├── app.py
├── train_model.py
├── custom_transformers.py
├── data_process.py
├── templates/
│   ├── index.html
│   └── result.html
```

## Business Impact

This system helps financial institutions identify high-risk customers and improve lending decisions by adjusting decision thresholds.

## Business Context

In real-world financial systems, recall is often prioritized over precision to minimize the risk of approving high-risk borrowers. This project reflects that trade-off by optimizing for recall and allowing flexible decision thresholds based on business needs.
