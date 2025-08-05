# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv("/content/Titanic-Dataset.csv")

# Display first 5 rows
print("🔍 First 5 rows of the dataset:")
print(df.head())

# 1. Summary Statistics
# -----------------------------
print("\n📊 Summary Statistics (Numerical):")
print(df.describe())
print("\n📊 Summary Statistics (Categorical):")
print(df.describe(include='O'))


# Check for missing values
print("\n❓ Missing values in each column:")
print(df.isnull().sum())

# 2. Univariate Analysis
# -----------------------------
# Histogram for numeric features
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("📉 Histograms for Numeric Features")
plt.tight_layout()
plt.show()


# Boxplots for numeric features
numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# 3. Correlation Matrix (Fix)
# -----------------------------
# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("🔗 Correlation Matrix (Numeric Features Only)")
plt.show()


# Pairplot for important variables
selected_cols = ['Survived', 'Pclass', 'Age', 'Fare']
sns.pairplot(df[selected_cols].dropna(), hue='Survived')
plt.suptitle("📊 Pairplot for Key Variables", y=1.02)
plt.show()


# 4. Categorical Features
# -----------------------------
# Count plots
categorical_cols = ['Sex', 'Pclass', 'Embarked']
for col in categorical_cols:
    sns.countplot(data=df, x=col, hue='Survived')
    plt.title(f"🎯 Survival count based on {col}")
    plt.show()


# 5. Skewness and Outliers
# -----------------------------
print("\n📐 Skewness of numeric columns:")
print(df[numeric_cols].skew())

# 5. Multicollinearity (Fix)
# -----------------------------
print("\n🔍 Checking for multicollinearity (correlation > 0.8):")
high_corr = correlation_matrix[(correlation_matrix > 0.8) & (correlation_matrix < 1)]
print(high_corr.dropna(how='all').dropna(axis=1, how='all'))


# 6. Observations / Inferences
# -----------------------------
print("\n🧠 Inferences:")
print("""
1. Female passengers had higher survival rates than males.
2. Passengers from 1st class were more likely to survive.
3. Younger passengers seemed to have slightly better chances.
4. Fare shows some outliers; some paid a very high amount.
5. Embarked location also correlates slightly with survival.
""")

