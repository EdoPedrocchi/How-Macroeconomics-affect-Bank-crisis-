import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_path = "/Users/pedrocchiedoardo/Desktop/ESAME_Satistics/Dataset_macro.xlsx"
df = pd.read_excel(file_path)

# Make plots look nice
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-deep")

# ==============================================================================
# 1. Initial Data Overview & Preprocessing
# ==============================================================================

print("--- Initial Data Overview ---")
print("DataFrame Shape (Rows, Columns):", df.shape)
print("\nColumn Data Types:")
print(df.info())

print("\nDescriptive Statistics for Numerical Features:")
print(df.describe())

print("\nFirst 5 rows of the dataset:")
print(df.head())

# It's crucial to have 'country' and 'year' or a date column for time series analysis
# Ensure these columns exist and are in the correct format.
# If 'year' is not in datetime format, convert it.
# e.g., df['year'] = pd.to_datetime(df['year'], format='%Y')

# ==============================================================================
# 2. Handling Missing Values
# ==============================================================================

print("\n--- Missing Value Analysis ---")
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percentage = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_data = pd.concat([missing_values, missing_percentage], axis=1, keys=['Missing Count', 'Missing %'])
print(missing_data[missing_data['Missing Count'] > 0])

# Example of a simple missing value imputation (e.g., forward fill for time series)
# You should choose a method based on the data and column.
# df.fillna(method='ffill', inplace=True) 

# ==============================================================================
# 3. Univariate & Class Imbalance Analysis
# ==============================================================================

print("\n--- Target Variable Analysis: 'Bank crisis' ---")
# Count the occurrences of each class
crisis_counts = df['BankingCrisis'].value_counts()
print(crisis_counts)

# Calculate and print the percentage for each class
crisis_percentage = df['BankingCrisis'].value_counts(normalize=True) * 100
print("\nCrisis Class Distribution (%):")
print(crisis_percentage)

# Visualize class imbalance with a bar chart
plt.figure(figsize=(8, 6))
sns.countplot(x='BankingCrisis', data=df)
plt.title('Distribution of Bank Crisis (Target Variable)')
plt.xlabel('Bank Crisis (0 = No Crisis, 1 = Crisis)')
plt.ylabel('Count')
plt.show()

# ==============================================================================
# 4. Bivariate and Time-Series Analysis
# ==============================================================================

# ==============================================================================
# 4. Bivariate and Time-Series Analysis
# ==============================================================================

# Define the target variable and identify all potential input features
target_variable = 'BankingCrisis'
all_features = df.columns.tolist()
input_features = [col for col in all_features if col not in ['country', 'year', target_variable]]

# Filter for only numerical input features for plotting
numerical_input_features = df[input_features].select_dtypes(include=np.number).columns.tolist()

## Time-Series Plots for All Key Variables
# Note: Plotting all variables at once can be overwhelming. This code
# will create a separate plot for each numerical feature.
print("--- Generating Time-Series Plots for All Numerical Features ---")
for feature in numerical_input_features:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y=feature, hue='countryname', legend=False)
    plt.title(f'Time Plot of {feature} by Country')
    plt.xlabel('Year')
    plt.ylabel(feature)
    plt.show()

print("\n--- Generating Box Plots for All Numerical Features vs. Crisis ---")
## Box Plots to Compare Features Between Crisis and Non-Crisis Periods
# This loop generates a box plot for every numerical feature to compare distributions
for feature in numerical_input_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target_variable, y=feature, data=df)
    plt.title(f'{feature} Distribution vs. Bank Crisis')
    plt.xlabel('Bank Crisis (0 = No, 1 = Yes)')
    plt.ylabel(feature)
    plt.show()

print("\n--- Correlation Matrix ---")
## Correlation Matrix (for all numerical features)
# This helps identify features strongly correlated with the 'Bank crisis'
numerical_df = df.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of All Numerical Variables')
plt.show()
print(correlation_matrix)
