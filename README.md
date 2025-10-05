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


   Missing values
  
 | Variable      | Missing Count | Missing % |
| ------------- | ------------- | --------- |
| cons_USD      | 45,879        | 78.64%    |
| cons_GDP      | 45,574        | 78.11%    |
| finv_USD      | 44,630        | 76.50%    |
| finv_GDP      | 44,308        | 75.94%    |
| inv_USD       | 44,195        | 75.75%    |
| inv_GDP       | 43,087        | 73.85%    |
| rGDP_USD      | 41,918        | 71.85%    |
| exports_GDP   | 41,636        | 71.36%    |
| CPI           | 38,985        | 66.82%    |
| imports_USD   | 38,663        | 66.27%    |
| BankingCrisis | 38,410        | 65.83%    |
| rGDP_pc       | 36,110        | 61.89%    |
| USDfx         | 35,022        | 60.03%    |
| infl          | 33,105        | 56.74%    |
| pop           | 6,405         | 10.98%    |

  

## 🏦 Target Variable Analysis: *Banking Crisis*

### Class Counts

| BankingCrisis | Count  |
| ------------- | ------ |
| 0.0           | 19,571 |
| 1.0           | 362    |

### Class Distribution (%)

| BankingCrisis | Proportion (%) |
| ------------- | -------------- |
| 0.0           | 98.18%         |
| 1.0           | 1.82%          |

> ⚠️ **Observation:**
> The dataset is **highly imbalanced**, with banking crises representing only about **1.8%** of all observations.
> This imbalance must be addressed during model training (e.g., SMOTE, class weights, or undersampling).

---

## 📊 Descriptive Statistics for Numerical Features

| Feature       | Count  | Mean     | Std      | Min       | 25%      | 50%      | 75%      | Max      |
| ------------- | ------ | -------- | -------- | --------- | -------- | -------- | -------- | -------- |
| year          | 58,343 | 1880.58  | 128.68   | 1086.00   | 1835.00  | 1903.00  | 1969.00  | 2030.00  |
| rGDP_pc       | 22,233 | 1.72e+07 | 1.11e+08 | 4.51e-08  | 6.00e+03 | 2.53e+04 | 2.17e+05 | 1.77e+09 |
| rGDP_USD      | 16,425 | 2.07e+05 | 1.06e+06 | 7.72e-02  | 3.32e+03 | 1.40e+04 | 8.38e+04 | 2.42e+07 |
| cons_GDP      | 12,769 | 82.45    | 22.42    | 8.84      | 71.41    | 80.35    | 90.65    | 298.38   |
| inv_GDP       | 15,256 | 22.91    | 11.27    | -21.55    | 16.08    | 22.07    | 27.83    | 243.18   |
| finv_GDP      | 14,035 | 21.87    | 10.31    | -4.97     | 15.40    | 21.06    | 26.29    | 157.87   |
| exports_GDP   | 16,707 | 37.07    | 56.74    | 1.75e-11  | 14.55    | 26.01    | 43.83    | 1168.42  |
| USDfx         | 23,321 | 195.47   | 1596.12  | 3.49e-16  | 0.57     | 1.34     | 6.91     | 89500.00 |
| CPI           | 19,358 | 2.93e+09 | 2.40e+11 | 1.73e-15  | 0.53     | 19.61    | 94.52    | 2.75e+13 |
| infl          | 25,238 | 3.82e+22 | 6.06e+24 | -100.00   | 0.00     | 3.25     | 8.93     | 9.63e+26 |
| pop           | 51,938 | 13.40    | 67.91    | 0.00      | 0.16     | 1.41     | 5.72     | 1534.73  |
| BankingCrisis | 19,933 | 0.018    | 0.134    | 0.00      | 0.00     | 0.00     | 0.00     | 1.00     |
| cons_USD      | 12,464 | 1.55e+05 | 8.66e+05 | 5.84e-08  | 941.88   | 5380.10  | 37433.31 | 2.45e+07 |
| inv_USD       | 14,148 | 6.17e+04 | 5.03e+05 | -16281.44 | 185.74   | 1237.79  | 9452.12  | 1.80e+07 |
| finv_USD      | 13,713 | 5.69e+04 | 4.17e+05 | -748.22   | 179.95   | 1099.59  | 9306.72  | 1.35e+07 |
| imports_USD   | 19,680 | 3.43e+04 | 2.26e+05 | 8.88e-10  | 58.62    | 603.15   | 5409.95  | 2.11e+07 |

> 💡 **Insights:**
>
> * There’s a **wide variance** in GDP-related indicators — consistent with differences across countries and centuries.
> * Inflation (`infl`) and CPI show **extreme outliers**, suggesting potential preprocessing (e.g., winsorization or log-scaling).
> * Population (`pop`) ranges from near zero to over 1,500 million — consistent with historical data for small and large nations.
> * The dataset includes both **historical** and **projected** values (1086–2030), requiring temporal segmentation for modeling.

Excellent — this correlation matrix reveals a lot about the **structure and relationships** in your macroeconomic panel dataset.
Here’s a clear, **analytical summary** you can include in your project’s README or notebook:

---

## 🔍 Correlation Matrix Analysis

<img width="980" height="905" alt="image" src="https://github.com/user-attachments/assets/4c4f422b-b0bb-44a3-a273-2ae6af56e864" />


### 🧠 General Observations

* The correlations across variables are **generally low to moderate**, suggesting a diverse set of macroeconomic indicators without excessive multicollinearity — a good sign for machine learning modeling.
* However, there are a few clusters of **highly correlated variables** that likely represent overlapping economic concepts.

---

### 🧩 Key Insights by Variable Group

#### **1. GDP and Investment Relationships**

* **`inv_GDP` ↔ `finv_GDP` = 0.75** → Very strong positive correlation.
  These variables both represent investment activity, possibly capturing similar dynamics (e.g., gross vs. fixed investment).
  👉 Consider dropping one or combining them via PCA or feature averaging.
* **`inv_GDP` and `year` = 0.21**, **`finv_GDP` and `year` = 0.24** → Investment ratios tend to rise over time, likely reflecting global financial deepening and capital accumulation trends.

#### **2. Nominal vs. Real Measures**

* **`rGDP_USD` ↔ `cons_USD` = 0.95**, **`rGDP_USD` ↔ `inv_USD` = 0.51**, **`rGDP_USD` ↔ `finv_USD` = 0.58**, **`rGDP_USD` ↔ `imports_USD` = 0.63**
  → These are **extremely high correlations**, indicating all USD-denominated variables move together with GDP.
  This likely reflects **scale effects** — richer countries have higher GDP, consumption, investment, and imports in USD.
  👉 Suggestion: normalize USD variables by GDP or population to remove size effects.

#### **3. Population Effects**

* **`pop` ↔ `rGDP_USD` = 0.42**, **`pop` ↔ `cons_USD` = 0.31**, **`pop` ↔ `imports_USD` = 0.26**
  → Larger populations are associated with higher absolute economic activity, again indicating the need for per-capita scaling if comparing countries.

#### **4. Trade and External Variables**

* **`exports_GDP`** has weak correlations with most other variables — its relationship to `BankingCrisis` is slightly **negative (-0.02)**.
  This might suggest that open economies are marginally less prone to crises, but the effect is small.

#### **5. Price and Inflation Indicators**

* **`CPI`** and **`infl`** show almost **no correlation** with other macro variables (all < 0.01).
  This could reflect noisy or incomplete inflation data across long historical periods or differing base years in CPI measurement.
  👉 May require transformation (e.g., differencing or log-scaling).

#### **6. Time Trend (year)**

* **`year`** shows modest positive correlations with several economic variables (e.g., `inv_GDP`, `USDfx`, `pop`), which is expected — economies evolve and expand over time.
  👉 A clear **time trend** exists, so time-fixed effects or year dummies may be necessary to avoid spurious correlations.

---

### ⚠️ **Target Variable: `BankingCrisis`**

* **Correlations with all predictors are extremely low (|r| < 0.04)**.
  This confirms that **banking crises are complex, nonlinear phenomena** not linearly explained by single macro variables.
  👉 Machine learning models (e.g., tree-based methods or lagged features) will be essential to capture crisis precursors.

---

### 🧮 Multicollinearity Risks (High Correlations > 0.8)

| Variable Pair              | Correlation | Comment                                         |
| -------------------------- | ----------- | ----------------------------------------------- |
| `cons_USD` ↔ `rGDP_USD`    | 0.95        | Redundant scaling measure                       |
| `cons_USD` ↔ `finv_USD`    | 0.90        | Strong shared scale component                   |
| `inv_USD` ↔ `finv_USD`     | 0.98        | Almost identical — likely double representation |
| `inv_GDP` ↔ `finv_GDP`     | 0.75        | High, likely overlapping measures               |
| `cons_USD` ↔ `imports_USD` | 0.91        | Both scale with GDP                             |

👉 **Action:** Before modeling, remove or combine one variable from each highly correlated pair to reduce redundancy.

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


