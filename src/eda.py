"""
EDA.py
Exploratory Data Analysis (EDA) on the ACS PUMS dataset

<Qihang Cheng>
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import Dataset
acs = pd.read_csv("pums_short.csv.gz")


# ========== Data Overview ==========
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
# ========== end of Data Overview ==========


# ========== Select Variables of Interest ==========
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
# ========== end of Variables of Interest ==========


# ========== Missing Values Analysis ==========
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

# Graph for Missing Summary
order = missing_summary["column"].tolist()

palette = sns.color_palette(
    "Reds",
    n_colors=len(missing_summary)
)[::-1]

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=missing_summary,
    x="missing_proportion",
    y="column",
    order=order,
    hue="column",
    palette=palette,
    legend=False
)

plt.title("Missing Value Proportion by Variable")
plt.xlabel("Missing Proportion")
plt.ylabel("Variable")

for i, row in missing_summary.iterrows():
    value = row["missing_proportion"]
    ax.text(
        value + 0.01,
        i,
        f"{value:.1%}",
        va="center"
    )

plt.xlim(0, missing_summary["missing_proportion"].max() + 0.1)
plt.tight_layout()
plt.show()
print()
# ========== end of Missing Values Analysis ==========


# ========== Division (DIVISION) ==========
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
# ========== end of Division (DIVISION) ==========


# ========== Monthly Rent (RNTP) ==========
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

# Graph for RNTP counts
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

# Ordered RNTP
st_summary = (
    acs_selected["ST_name"].value_counts()
    .reset_index()
)

st_summary.columns = ["state", "count"]

st_summary = st_summary.sort_values("count")
print(st_summary.head)
print()

st_palette = sns.color_palette(
    "Blues",
    n_colors=len(st_summary)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=st_summary,
    x="state",
    y="count",
    hue="state",
    palette=st_palette,
    legend=False
)

plt.title("Household Count by State")
plt.xlabel("State")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
print()

# Graph for Distribution of RNTP
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
sns.histplot(simRNTP, bins=50)
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
# ========== End of Monthly Rent (RNTP) ==========


# ========== Household Income (HINCP) ==========
income = acs_selected["HINCP"].dropna()
print("===== Household Income Summary =====")
print(income.describe())
print()

# Boxplot for Distribution of Household Income
plt.figure(figsize=(10, 6))
sns.boxplot(x=income)
plt.title("Distribution of Household Income")

# Histogram for Distribution of Household Income
plt.figure(figsize=(10, 6))
sns.histplot(income, bins=50)
plt.title("Distribution of Household Income")
plt.xlabel("Household Income")
plt.ylabel("Count")
plt.tight_layout()

# Aujusted Household Income
income_adj = np.log10(income[income > 0])

plt.figure(figsize=(10, 6))
sns.boxplot(x=income_adj)
plt.xlabel("log10(Household Income)")
plt.ylabel("Count")

plt.figure(figsize=(10, 6))
sns.histplot(income_adj, bins=50)
plt.title("Distribution of Log Household Income")
plt.xlabel("log10(Household Income)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# ========== end of Household Income (HINCP) ==========
