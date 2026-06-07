"""
Exploratory Data Analysis (EDA) on the ACS PUMS dataset
<Qihang Cheng>
"""

# Project Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# Import Dataset
acs = pd.read_csv("pums_short.csv.gz")


# Data Overview
print("===== Dataset Overview =====")
print(f"Rows: {acs.shape[0]}")
print(f"Columns: {acs.shape[1]}")
print()
print(acs.head())
print()

print("===== Column Names =====")
print(acs.columns.all)
col_name = acs.columns.tolist()
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

plt.figure(figsize=(10, 6))
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
print()


# Division (DIVISION)
div_name = {
    0: "Puerto Rico",
    1: "New England (Northeast region)",
    2: "Middle Atlantic (Northeast region)",
    3: "East North Central (Midwest region)",
    4: "West North Central (Midwest region)",
    5: "South Atlantic (South region)",
    6: "East South Central (South region)",
    7: "West South Central (South Region)",
    8: "Mountain (West region)",
    9: "Pacific (West region)",
}

acs_selected["DIVISION"].value_counts().sort_index()
acs_selected["DIVISION_name"] = acs_selected["DIVISION"].replace(div_name)
print("=== Division Name ===")
print(acs_selected["DIVISION_name"])
print()


# Monthly Rent (RNTP)
total_rntp = acs_selected["RNTP"].shape[0]
valid_rntp = acs_selected["RNTP"].count()
print(f"\"RNTP\" has", total_rntp, "rows in total.")
print(f"\"RNTP\" has", valid_rntp, "valid rows.")
print()

rntp = acs_selected["RNTP"].dropna()
print("=== Monthly Rent Head ===")
print(rntp.head(5))
print()

print("=== Monthly Rent Summary ===")
print(rntp.describe())
print(f"range\t", rntp.max()-rntp.min())
print()

plt.figure(figsize=(10, 6))
sb.boxplot(x=rntp)
plt.title("Distribution of Monthly Rent")
plt.xlabel("Monthly Rent")
plt.tight_layout()
plt.show()
print()

def rntp_proportion(less_than: int) -> None:
    total = rntp.count()
    p = len(rntp.loc[rntp <= less_than]) / total
    print(f"The proportion of the monthly rent that are have less than "
          f"{less_than} is {p: .3f}")

rntp_proportion(1000)
rntp_proportion(2000)
rntp_proportion(3000)
print()

plt.figure(figsize=(10, 6))
sb.histplot(rntp, bins=50)
plt.title("Distribution of Monthly Rent")
plt.xlabel("Monthly Rent")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
print()

# RNTP Monte Carlo Sampling
sample_size = 100
n_trials = 10000

simRNTP = pd.Series([
    rntp.sample(sample_size, replace=True).mean()
    for _ in range(n_trials)
])

print("=== Monte Carlo Sampling of 100 Samples and 10,000 Trials ===")
print(simRNTP.describe())
print()

simRNTP = pd.Series(simRNTP)

plt.figure(figsize=(10, 6))
sb.histplot(simRNTP, bins=50)
plt.title("Monte Carlo Simulation of Mean Monthly Rent, n = 100, trials = 10,000")
plt.xlabel("Sample Mean of Monthly Rent")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
print()

print("===== Monte Carlo Simulation: Monthly Rent =====")
print(f"Sample size: {sample_size}")
print(f"Number of trials: {n_trials}")
print(f"Simulation mean: {simRNTP.mean():.2f}")
print(f"Simulation standard deviation: {simRNTP.std():.2f}")
print(f"Population mean: {rntp.mean():.2f}")
print(f"Theoretical standard error: {rntp.std() / np.sqrt(sample_size):.2f}")
print()


# Monthly Rent by Region
rent_division = acs_selected.loc[rntp.index, "DIVISION_name"]
print(rent_division.value_counts().sort_index())
print()

rent_data = acs_selected.loc[acs_selected["RNTP"].notna(), 
                             ["RNTP", "DIVISION_name"]].copy()
print(rent_data.head())
print()

plt.figure(figsize=(10, 6))

sb.boxplot(data=rent_data, x="RNTP", y="DIVISION_name")

plt.title("Monthly Rent by Census Division")
plt.xlabel("Monthly Rent")
plt.ylabel("Census Division")
plt.tight_layout()
plt.show()
print()


# Household Income (HINCP)
income = acs_selected["HINCP"].dropna()
print("===== Household Income Summary =====")
print(income.describe())
print()

