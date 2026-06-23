# 🚢 DOT PY — Data Science & AI Program
## Module 01 · Data Preprocessing
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

### Real-World Business Equivalents
While built on historical records, the data-wrangling patterns applied throughout this pipeline directly replicate technical workflows used by enterprise-level engineering teams across diverse sectors:

| Titanic Machine Learning Problem | Industry Enterprise Equivalent |
| :--- | :--- |
| Predict passenger survival based on profile telemetry | Predict customer churn/retention metrics based on user activity signatures |
| Rank features (`Age`, `Class`, `Sex`) driving survival | Isolate risk vectors causing loan defaults or financial fraud |
| Handle structural missing values in numerical `Age` fields | Reconstruct empty records in critical income or credit score fields |
| Convert text categories (`Sex`, `Embarked`) to numeric flags | Encode categorical vectors like global product groups or localized target zones |
| Scale bounded continuous metrics (`Fare`, `Age`) | Normalize skewed financial transaction volume ranges to uniform boundaries |

---

## 🛠️ Core Tech Stack & Libraries
The notebook leverages industry-standard open-source libraries to handle end-to-end data pipelines:
* **`pandas`**: Advanced structured data manipulation via tabular DataFrames.
* **`numpy`**: High-performance vectorized numerical operations and array handling.
* **`matplotlib` & `seaborn`**: Statistical visualizations, outlier mapping, and distribution checks.
* **`scikit-learn`**: Machine learning utilities, including robust transformers (`LabelEncoder`, `StandardScaler`, `MinMaxScaler`).

---

## 🚀 Pipeline Implementation Architecture

### 1. Data Understanding & Structural Validation
Exploratory calls (`df.info()` and `df.describe()`) provide clarity on features, data types, and structural health.
* **Missing Value Analysis:**

  * `Age`: **19.86%** missing records (177 rows).
  * `Cabin`: **77.10%** missing records (687 rows).
  * `Embarked`: **0.22%** missing records (2 rows).

### 2. Missing Value Imputation Strategy
* **`Age`:** Filled using **Median** values. The age distribution exhibits a noticeable right-skew; the median ensures robustness against extreme outliers.
* **`Embarked`:** Filled using the **Mode** (`'S'`). Since only 2 rows are empty, filling them with the highest-frequency category keeps data distributions stable without introducing structural noise.
* **`Cabin`:** **Dropped completely**. With over 77% of structural data missing, imputation would inject massive artificial bias into downstream models.

### 3. Outlier Identification & Treatment (Winsorization)
To prevent distance-based estimators (e.g., KNN, Linear Classifiers) from being skewed by extreme values, anomalies are evaluated using the Interquartile Range (IQR) method:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Boundaries} = [Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$$

* **Treatment Strategy:** Instead of dropping rows and losing data, outliers in `Age` and `Fare` are capped at their respective mathematical boundaries using the `.clip()` function (Winsorization), preserving dataset volume.

### 4. Categorical Feature Encoding
* **`Sex`:** Transformed via **Label Encoding** (Binary conversion into `0` and `1`).
* **`Embarked`:** Transformed via **One-Hot Encoding** with `drop_first=True`. This prevents the *dummy variable trap* and avoids introducing an artificial ordinal relationship (e.g., predicting $C < Q < S$).

### 5. Feature Engineering (Signal Enhancement)
New columns are derived from raw inputs to expose more explicit patterns to downstream models:
* **`FamilySize`** $= \text{SibSp} + \text{Parch} + 1$: Represents total family volume to capture family-oriented boarding behavior.
* **`IsAlone`**: A binary flag ($1$ if `FamilySize` $== 1$, else $0$) that isolates solo travelers.
* **`AgeGroup`**: Converts continuous, noisy age numbers into distinct lifecycle brackets.

---

## 📊 Sample Preprocessed Data Output

| Survived | Pclass | Sex | Age | SibSp | Parch | Fare | Embarked_Q | Embarked_S | FamilySize | IsAlone | AgeGroup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 3 | 1 (Male) | 22.0 | 1 | 0 | 7.2500 | 0 | 1 | 2 | 0 | Young Adult |
| **1** | 1 | 0 (Female) | 38.0 | 1 | 0 | 65.6344 | 0 | 0 | 2 | 0 | Adult |
| **1** | 3 | 0 (Female) | 26.0 | 0 | 0 | 7.9250 | 0 | 1 | 1 | 1 | Young Adult |

---

## 📂 Project Structure
```bash
├── titanic-preprocessing-pipeline.ipynb   # Complete interactive pipeline template
├── titanic.csv                            # Raw input source data
└── README.md                              # Technical repository blueprint
