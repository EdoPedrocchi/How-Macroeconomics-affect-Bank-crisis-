# 🏦 Global Banking Crisis Prediction (1086–2030)

## 📘 Project Overview

This project aims to **predict banking crises** using a large-scale **panel dataset of macroeconomic variables** across **241 countries**, covering historical data from **1086 to 2024** and projections up to **2030**.

The target variable is **`BankingCrisis`** (binary classification), while the explanatory variables include a wide range of macroeconomic indicators.

The objective is to build interpretable and robust **machine learning models** capable of identifying early warning signals of financial crises.

---

## 🧩 Dataset Description

Dataset from: [Global Macro Data](https://www.globalmacrodata.com/?utm_source=chatgpt.com)


**Variables:**
countryname, ISO3, id, year,
rGDP_pc, rGDP_USD, cons_GDP, inv_GDP, finv_GDP,
exports_GDP, USDfx, CPI, infl, pop,
BankingCrisis, cons_USD, inv_USD, finv_USD, imports_USD



- **Target variable:** `BankingCrisis`  
- **Excluded identifiers:** `countryname`, `ISO3`, `id`, `year`

**Dataset type:** Panel (cross-country and time-series)  
**Period covered:** 1086–2030  
**Countries:** 241  

---
## 📖LIterature Review

..cecefecr
---
## 🚀 Project Pipeline

### **1. Data Loading & Understanding**
1. Load the dataset (`pandas.read_csv()`).
2. Inspect shape, types, and missingness (`df.info()`, `df.describe()`).
3. Verify that `BankingCrisis` is binary (0/1).
4. Confirm identifiers (`countryname`, `ISO3`, `id`, `year`).

---

### **2. Data Cleaning & Preprocessing**
1. **Handle Missing Values:**
   - Visualize (`missingno` or `sns.heatmap`).
   - Impute or drop (mean, median, interpolation, or removal).
2. **Fix Data Types:** Ensure correct numeric and categorical dtypes.
3. **Handle Outliers:** Winsorize or cap extreme values.
4. **Scale Numerical Variables:** Standardize or normalize (`StandardScaler`).
5. **Split Features and Target:**  
   - `X` = explanatory variables  
   - `y` = `BankingCrisis`

---

### **3. Exploratory Data Analysis (EDA)**
1. **Univariate Analysis:**  
   - Check feature distributions and class balance.  
2. **Bivariate Analysis:**  
   - Correlation heatmaps, pairplots, and mean comparisons (crisis vs. non-crisis).  
3. **Temporal & Cross-Country Trends:**  
   - Plot macroeconomic variables over time and across countries.  

---

### **4. Feature Engineering**
1. **Lag Features:** Create lagged versions of GDP, inflation, etc.  
2. **Growth Rates:** Compute percentage changes (GDP growth, inflation change).  
3. **Ratios:** Investment-to-GDP, consumption-to-GDP, export-import ratio.  
4. **Rolling Statistics:** Moving averages and volatility measures.  
5. **Country & Year Effects:** Add country dummies or year trends.

---

### **5. Train/Test Splitting (Panel-Aware)**
1. Sort by `country` and `year`.  
2. Use **time-series split** (train on past, test on future).  
3. Optionally apply **walk-forward validation** for robustness.  

---

### **6. Address Class Imbalance**
1. If crises are rare, use:
   - **SMOTE** / **ADASYN** (oversampling minority class), or  
   - **Class weights** in model training.  

---

### **7. Model Selection & Training**
1. **Baseline:** Logistic Regression (interpretable benchmark).  
2. **Tree-Based Models:** Random Forest, XGBoost, LightGBM, CatBoost.  
3. **Advanced Alternatives:** SVMs or LSTM/Transformers for sequential modeling.  

---

### **8. Model Evaluation**
1. **Metrics:**  
   - Precision, Recall, F1-Score, ROC-AUC.  
   - Prioritize Recall (detecting crises is critical).  
2. **Temporal Validation:** Evaluate performance on future time periods.  
3. **Cross-Country Validation:** Hold out countries for out-of-sample testing.  

---

### **9. Model Interpretation**
1. **Feature Importance:** Tree importances, SHAP, or LIME explanations.  
2. **Economic Insights:** Identify leading indicators of banking crises.  
3. **Compare across regions and time periods.**
4. Implement SAFE methods by Giudici and Rafinetti

---

### **10. Refinement & Deployment**
1. Tune hyperparameters (Optuna or GridSearchCV).  
2. Simplify and interpret final model.  
3. Create reproducible pipeline (`sklearn.pipeline.Pipeline`).  
4. Deploy or visualize predictions (Streamlit, Flask, or dashboard).

---
## 📊 Evaluation Metrics
| Metric | Description |
|:--|:--|
| **Precision** | Fraction of predicted crises that were correct |
| **Recall** | Fraction of actual crises detected |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Overall model discrimination ability |

---

## 🛠️ Tools & Libraries

- **Python** (3.10+)
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**, **XGBoost**, **LightGBM**
- **Imbalanced-learn** (SMOTE)
- **SHAP**, **LIME** (model explainability)
- **Optuna** (hyperparameter optimization)
- **Streamlit** / **Flask** (optional deployment)

---

## 📈 Potential Extensions

- Integrate **macroeconomic forecasting** (ARIMA, VAR, Prophet).  
- Develop a **country risk scoring system**.  
- Build an **interactive crisis dashboard** with real-time data updates.  

---

## 📚 References
- Laeven & Valencia (2020): *Systemic Banking Crises Database*.  
- IMF & World Bank macroeconomic datasets.  
- Reinhart & Rogoff (2009): *This Time Is Different: Eight Centuries of Financial Folly.*

---

## 🧠 Author
Developed by **[Your Name]**  
Focus areas: Economics • Finance • Machine Learning • Data Science  

---


