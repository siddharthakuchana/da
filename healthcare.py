import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


# Load dataset
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target

print("\n================ DATA OVERVIEW ================\n")

# Shape
print("Shape:", df.shape)

# Column info
print("\nColumns:\n", df.columns.tolist())

# Data types
print("\nData Types:\n", df.dtypes)

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

print("\n================ CENTRAL TENDENCY ================\n")

print("Mean:\n", df.mean())
print("\nMedian:\n", df.median())
print("\nMode:\n", df.mode().iloc[0])

print("\n================ DISPERSION ================\n")

print("Variance:\n", df.var())
print("\nStandard Deviation:\n", df.std())
print("\nRange:\n", df.max() - df.min())

print("\n================ DISTRIBUTION SHAPE ================\n")

print("Skewness:\n", df.skew())
print("\nKurtosis:\n", df.kurtosis())

print("\n================ QUARTILES ================\n")

print("Q1 (25%):\n", df.quantile(0.25))
print("\nQ2 (50%):\n", df.quantile(0.50))
print("\nQ3 (75%):\n", df.quantile(0.75))

print("\n================ OUTLIER DETECTION (IQR METHOD) ================\n")

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

print("Outliers per column:\n", outliers)

print("\n================ CORRELATION ================\n")

print(df.corr())

print("\n================ STATISTICAL SUMMARY (FULL) ================\n")

print(df.describe().T)
