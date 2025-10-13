# 🏦 Global Banking Crisis Prediction (1086–2030)

Domande per il prof:
- variabile target sbilanicata ( 0.0=19,571|
| 1.0=362)
- eliminare guerre? (1914 = 12 default)
- elimninare crisi del 2008? (=17 default)
- possibile approccio?
Target = 1.82% → molto sbilanciato.
Divido in train/test stratificato (mantengo proporzione crisi/non crisi nel test).
Applico SMOTE(da capire) sul training set per generare più casi di crisi.


- è un problema se le scrivo?
- accetta tesisisti?

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

WIP


---
## **Exploratory Data Analysis (EDA)**

### Dataset Shape
- **Rows:** 58,343  
- **Columns:** 19

### Data Types
| Column          | Type    |
|-----------------|---------|
| countryname     | object  |
| ISO3            | object  |
| id              | object  |
| year            | int64   |
| rGDP_pc         | float64 |
| rGDP_USD        | float64 |
| cons_GDP        | float64 |
| inv_GDP         | float64 |
| finv_GDP        | float64 |
| exports_GDP     | float64 |
| USDfx           | float64 |
| CPI             | float64 |
| infl            | float64 |
| pop             | float64 |
| BankingCrisis   | float64 |
| cons_USD        | float64 |
| inv_USD         | float64 |
| finv_USD        | float64 |
| imports_USD     | float64 |


### Descriptive Statistics
|       | year     | rGDP_pc       | rGDP_USD      | cons_GDP    | inv_GDP    | finv_GDP   | exports_GDP | USDfx       | CPI         | infl         | pop        | BankingCrisis | cons_USD     | inv_USD     | finv_USD    | imports_USD |
|-------|---------|---------------|---------------|------------|-----------|-----------|-------------|------------|------------|-------------|------------|---------------|-------------|------------|------------|-------------|
| count | 58343   | 22233         | 16425         | 12769      | 15256     | 14035     | 16707       | 23321      | 19358      | 25238       | 51938      | 19933         | 12464       | 14148      | 13713      | 19680       |
| mean  | 1880.58 | 1.719e+07     | 2.073e+05     | 82.45      | 22.91     | 21.87     | 37.07       | 195.47     | 2.93e+09   | 3.82e+22    | 13.40      | 0.0182        | 1.546e+05   | 6.166e+04  | 5.695e+04  | 3.430e+04   |
| std   | 128.68  | 1.112e+08     | 1.059e+06     | 22.42      | 11.27     | 10.31     | 56.74       | 1596.12    | 2.405e+11  | 6.065e+24   | 67.91      | 0.134         | 8.662e+05   | 5.030e+05  | 4.172e+05  | 2.261e+05   |
| min   | 1086    | 4.51e-08      | 0.0772        | 8.84       | -21.55    | -4.97     | 1.75e-11    | 3.49e-16   | 1.73e-15   | -100        | 0.000004   | 0.0           | 5.84e-08    | -1.628e+04 | -748.22    | 8.88e-10    |
| 25%   | 1835    | 6004          | 3315          | 71.41      | 16.08     | 15.40     | 14.55       | 0.5698     | 0.5301     | 0.0         | 0.158      | 0.0           | 941.88      | 185.74     | 179.95     | 58.62       |
| 50%   | 1903    | 25303         | 14039         | 80.35      | 22.07     | 21.06     | 26.01       | 1.3399     | 19.61      | 3.25        | 1.409      | 0.0           | 5380.10     | 1237.79    | 1099.59    | 603.15      |
| 75%   | 1969    | 216734        | 83831         | 90.65      | 27.83     | 26.29     | 43.83       | 6.911      | 94.52      | 8.93        | 5.724      | 0.0           | 37433.31    | 9452.12    | 9306.72    | 5409.95     |
| max   | 2030    | 1.767e+09     | 2.419e+07     | 298.38     | 243.18    | 157.87    | 1168.42     | 89500      | 2.750e+13  | 9.635e+26   | 1534.73    | 1.0           | 2.449e+07   | 1.804e+07  | 1.346e+07  | 2.106e+07   |


### Missing Value Analysis
| Column        | Missing Count | Missing % |
|---------------|---------------|-----------|
| cons_USD      | 45,879        | 78.64     |
| cons_GDP      | 45,574        | 78.11     |
| finv_USD      | 44,630        | 76.50     |
| finv_GDP      | 44,308        | 75.94     |
| inv_USD       | 44,195        | 75.75     |
| inv_GDP       | 43,087        | 73.85     |
| rGDP_USD      | 41,918        | 71.85     |
| exports_GDP   | 41,636        | 71.36     |
| CPI           | 38,985        | 66.82     |
| imports_USD   | 38,663        | 66.27     |
| BankingCrisis | 38,410        | 65.83     |
| rGDP_pc       | 36,110        | 61.89     |
| USDfx         | 35,022        | 60.03     |
| infl          | 33,105        | 56.74     |
| pop           | 6,405         | 10.98     |

### Target Variable Analysis: `BankingCrisis`
| Value | Count |
|-------|-------|
| 0.0   | 19,571|
| 1.0   | 362   |

**Crisis Class Distribution (%)**
| Value | Proportion |
|-------|------------|
| 0.0   | 98.18      |
| 1.0   | 1.82       

|<img width="708" height="545" alt="image" src="https://github.com/user-attachments/assets/7cc39a23-1b20-478c-9138-487a16acf386" />

--- Univariate Distributions (Histograms & KDEs) ---
<img width="1465" height="1105" alt="image" src="https://github.com/user-attachments/assets/2c883cdb-6b8b-489f-9b00-f73dcf67503f" />


### Mean Comparison (Crisis vs Non-Crisis)

| Variable      | Non-Crisis (0.0) | Crisis (1.0) |
|---------------|-----------------|---------------|
| year          | 1927.92         | 1944.41       |
| rGDP_pc       | 5,517,008       | 13,438,540    |
| rGDP_USD      | 227,221         | 336,069       |
| cons_GDP      | 80.80           | 81.20         |
| inv_GDP       | 22.38           | 20.93         |
| finv_GDP      | 21.25           | 20.02         |
| exports_GDP   | 33.35           | 26.93         |
| USDfx         | 169.27          | 139.62        |
| CPI           | 87.19           | 27.05         |
| infl          | 6.88e+22        | 76.64         |



<img width="1338" height="1275" alt="image" src="https://github.com/user-attachments/assets/6278fe03-0c3f-4235-acf3-b113a3316e71" />


<img width="979" height="903" alt="image" src="https://github.com/user-attachments/assets/fab9a2a9-d1d2-4c33-91a4-69c2fefc7b21" />



<img width="832" height="545" alt="image" src="https://github.com/user-attachments/assets/5e797822-fd7b-4db4-a98d-f5c8c145223c" />


<img width="843" height="545" alt="image" src="https://github.com/user-attachments/assets/6e2d8801-52f8-4d3f-af0d-2c1e58e702c8" />

<img width="839" height="545" alt="image" src="https://github.com/user-attachments/assets/722a1019-6b21-4dfc-8129-f1ced5f6b3e4" />

<img width="839" height="545" alt="image" src="https://github.com/user-attachments/assets/ad70dd96-d353-49e6-b3d5-e0ddbc5705f8" />


<img width="1078" height="545" alt="image" src="https://github.com/user-attachments/assets/e0ad4173-ee88-40c4-a677-926caebe765d" />



### Outliers Detected Using IQR Method (per variable)
| Variable      | Outlier Count |
|---------------|---------------|
| year          | 3,348         |
| rGDP_pc       | 3,592         |
| rGDP_USD      | 2,439         |
| cons_GDP      | 799           |
| inv_GDP       | 461           |
| finv_GDP      | 475           |
| exports_GDP   | 821           |
| USDfx         | 4,480         |
| CPI           | 774           |
| infl          | 3,200         |
| pop           | 7,321         |
| cons_USD      | 2,067         |
| inv_USD       | 2,438         |
| finv_USD      | 2,334         |
| imports_USD   | 3,383         |


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

  

### Correlation Matrix 

<img width="980" height="905" alt="image" src="https://github.com/user-attachments/assets/4c4f422b-b0bb-44a3-a273-2ae6af56e864" />


### General Observations

* The correlations across variables are **generally low to moderate**, suggesting a diverse set of macroeconomic indicators without excessive multicollinearity — a good sign for machine learning modeling.
* However, there are a few clusters of **highly correlated variables** that likely represent overlapping economic concepts.

---

### Key Insights by Variable Group

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




   
---

### **Feature Engineering**
1. **Lag Features:** Create lagged versions of GDP, inflation, etc.  
2. **Growth Rates:** Compute percentage changes (GDP growth, inflation change).  
3. **Ratios:** Investment-to-GDP, consumption-to-GDP, export-import ratio.  
4. **Rolling Statistics:** Moving averages and volatility measures.  
5. **Country & Year Effects:** Add country dummies or year trends.


---

### **Train/Test Splitting**
  
WIP
---

### **Address Class Imbalance**

WIP

---

### **Model Selection & Training**
1. **Logistic Regression** (interpretable benchmark).  
2. **Tree-Based Models:** Random Forest, XGBoost, 
3. **Advanced Alternatives:** SVMs and DeepLearning.  

---

### **Model Evaluation**

WIP
 

---

### **Model Interpretation** 
1. **Economic Insights:** Identify leading indicators of banking crises.  
2. **Compare across regions and time periods.**
3. **Implement SAFE** methods by Giudici and Rafinetti
---

## 📈 Potential Extensions

- Deploy or visualize predictions (Streamlit, Flask, or dashboard).
- Integrate **macroeconomic forecasting** (ARIMA, VAR, Prophet).  
- Develop a **country risk scoring system**.  
- Build an **interactive crisis dashboard** with real-time data updates.  




