# 🏦 Credit Scoring Based Loan Risk Assessment

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Data Science](https://img.shields.io/badge/Domain-FinTech-%23FF6F61)
![Machine Learning](https://img.shields.io/badge/Framework-Stacking%20Ensemble-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-92.18%25-brightgreen.svg?style=for-the-badge)

## 📌 Project Overview
Leveraging financial historical patterns, this project implements a highly optimized end-to-end Machine Learning pipeline to identify high-risk borrowers and predict loan defaulters. By analyzing a customer's credit attributes, debt-to-income metrics, and employment stability, this framework provides banking institutions with an automated, data-driven system to gauge default risk accurately before issuing capital.

---

## 🚀 Key Achievements & Core Highlights

* **🎯 Breakthrough Performance:** Achieved a peak classification **Accuracy of 92.18%** through advanced multi-tier ensembling.
* **🛠️ Robust Outlier Protection:** Integrated custom **IQR-based Winsorization (capping)** that safeguards information integrity while neutralizing heavy distribution skews without discarding incomplete data points.
* **⚖️ Imbalance Remediation:** Successfully resolved severe target class imbalance (`BAD=0` vs `BAD=1`) by utilizing **SMOTE** oversampling along continuous feature vector boundaries.

---

## 📊 Technical Specifications & Architecture

### Data Engineering & Preprocessing
1. **Missing Data Architecture:** Replaced simple statistical drops or broad mean-fills with a multi-variable **K-Nearest Neighbors (KNN) Imputer ($k=5$)** to logically capture non-linear feature interactions.
2. **Feature Standardisation:** Implemented **Z-Score Normalization** via `StandardScaler` to bring disjoint attributes (e.g., total asset values vs discrete inquiry tallies) to a common baseline scale:
   $$\mathbf{z} = \frac{x - \mu}{\sigma}$$
3. **Categorical Handling:** Safe vectorized processing using strict multi-track `OneHotEncoder` pipelines mapping disjoint elements cleanly.

### Ensemble Modeling Pipeline
The system utilizes a structured, cooperative model framework to handle complex tabular structures:

* **Base Estimator:** **XGBoost Classifier** (Gradient Boosting with optimized log-loss evaluation profiles) for structural interaction patterns with achieving of outstanding acuuracy of **92.18%** .
* **Voting Tier:** A soft-voting classifier blending probabilistic metrics.
* **Meta-Heuristic Estimator:** A final **Logistic Regression** layer acting as the meta-learner to make the definitive decision based on the combined probabilities of the underlying models.



---

## 📂 Project Directory Structure
```text
├── loan_dataset.csv          # Raw FinTech customer database records
├── Prediction_code.ipynb     # Complete end-to-end interactive Colab/Jupyter Notebook
├── README.md                 # Project documentation portfolio
