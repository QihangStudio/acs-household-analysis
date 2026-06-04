"""
Exploratory Data Analysis (EDA) on the ACS PUMS dataset
<Qihang Cheng>
"""

# Project Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Data Overview
acs = pd.read_csv("pums_short.csv.gz")

print("===== Dataset Overview =====")
print(f"Rows: {acs.shape[0]}")
print(f"Columns: {acs.shape[1]}")
print()
print(acs.head())
print()

print("===== Column Names =====")
print(acs.columns.tolist())
print()

print("===== Column Types =====")
acs.info()
print()

# Select Variables of Interest
selected_cols = [
    "DIVISION", "REGION", "ST",
    "HINCP",   # household income
    "FINCP",   # family income
    "VALP",    # property value
    "RNTP",    # monthly rent
    "R65",     # presence of people 65+
    "TEN",     # tenure / ownership
    "NP",      # number of persons
    "VEH"      # vehicles
]

acs_selected = acs[selected_cols].copy()

print("===== Selected Columns =====")
print(acs_selected.head())
print()

# Missing Values Analysis
missing_summary = (
    acs_selected.isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

missing_summary.columns = ["column", "missing_proportion"]

print("===== Missing Value Summary =====")
print(missing_summary)
print()

plt.figure(figsize=(8, 5))
sb.barplot(
    data=missing_summary,
    x="missing_proportion",
    y="column",
    color="grey",
)
plt.title("Missing Value Proportion by Variable")
plt.xlabel("Missing Proportion")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# Household Income
income = acs_selected["HINCP"].dropna()
print("===== Household Income Summary =====")
print(income.describe())

plt.figure(figsize=(8, 5))
sb.histplot(income, bins=50)
plt.title("Distribution of Household Income")
plt.xlabel("Household Income")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

income_positive = income[income > 0]
plt.figure(figsize=(8, 5))
sb.histplot(np.log10(income_positive), bins=50)
plt.title("Distribution of Log Household Income")
plt.xlabel("log10(Household Income)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
