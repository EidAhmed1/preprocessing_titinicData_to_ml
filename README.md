# 🚢 DOT PY — Data Science & AI Program
## Module 01 & 02 · Data Preprocessing & ml
### In-Class Task Session — Titanic Dataset Pipeline

---

## 📋 Overview
This repository contains a comprehensive data preprocessing and exploratory data analysis (EDA) framework built for the classic **Titanic: Machine Learning from Disaster** dataset. The goal of this project is to construct a clean, structurally sound, and production-ready data pipeline that transforms raw historical information into highly optimized feature inputs for binary classification models.

> **Data Preprocessing Creed:** *Clean data in → Better model out. Garbage data in → Garbage model out. Preprocessing is not a formality; it is the absolute architectural foundation that determines the ceiling of your model's performance.*

---

## 🧠 Business Brief & Problem Statement
In April 1912, the RMS Titanic sank after colliding with an iceberg, resulting in the tragic loss of 1,502 out of 2,224 passengers and crew members. This dataset tracks specific demographics and boarding metrics for 891 individuals.

### The Objective
To isolate, clean, and engineer structural attributes capable of driving a statistical model to predict:
$$\text{Target Variable } (Y) = \begin{cases} 1, & \text{Survived} \\ 0, & \text{Did Not Survive} \end{cases}$$

---

## 🤖 Machine Learning Modeling & Evaluation

After completing the preprocessing and feature engineering pipeline, multiple Machine Learning models were trained and evaluated to predict survival outcomes.

The following algorithms were applied:

- Linear Regression
- Lasso Regression (L1 Regularization)
- Ridge Regression (L2 Regularization)
- Polynomial Regression (Non-linear feature expansion)

Each model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

### 📊 Evaluation Process
All models were trained on the processed training dataset and evaluated on unseen test data to measure generalization performance and detect overfitting or underfitting patterns.

A comparative evaluation was performed to analyze how linear, regularized, and non-linear models behave under the same preprocessing pipeline.

> This modeling phase is strictly for **educational purposes**, focusing on understanding model behavior rather than production deployment.

---

## 📊 Real-World Business Equivalents
While built on historical records, the same preprocessing and modeling logic is widely used in real-world AI systems:

| Titanic Problem | Real Industry Equivalent |
| :--- | :--- |
| Predict survival outcome | Predict customer churn |
| Feature importance analysis | Risk analysis in banking/insurance |
| Missing data handling | Incomplete customer records recovery |
| Encoding categorical variables | Converting business categories into ML features |
| Feature scaling | Normalizing financial transaction data |

---

## 🛠️ Core Tech Stack & Libraries
- pandas → data manipulation  
- numpy → numerical operations  
- matplotlib & seaborn → visualization  
- scikit-learn → preprocessing + ML models  

---

## 🚀 Pipeline Implementation Architecture
*(unchanged from original notebook section)*

---

## 📂 Project Structure
```bash
├── titanic-preprocessing-pipeline.ipynb
├── titanic.csv
└── README.md
