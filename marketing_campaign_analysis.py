#############################################
# MARKET CAMPAIGN PROJECT
# EDA & CRM + ML
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_validate
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


#############################################
# SETTINGS
#############################################

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

sns.set_theme(style="whitegrid")

DATA_PATH = r"C:\Users\omerkuru\Desktop\marketing_campaign.csv"
RANDOM_STATE = 42
RUN_SEARCH = False


#############################################
# FUNCTIONS
#############################################

def add_value_labels(ax, fmt="{:.2f}", fontsize=9):
    for p in ax.patches:
        height = p.get_height()
        if pd.notnull(height):
            ax.annotate(
                fmt.format(height),
                (p.get_x() + p.get_width() / 2., height),
                ha="center",
                va="bottom",
                fontsize=fontsize
            )


def add_percent_labels(ax, fontsize=9):
    for p in ax.patches:
        height = p.get_height()
        if pd.notnull(height):
            ax.annotate(
                f"{height * 100:.1f}%",
                (p.get_x() + p.get_width() / 2., height),
                ha="center",
                va="bottom",
                fontsize=fontsize
            )


def summary_table(dataframe, group_col):
    table = dataframe.groupby(group_col).agg({
        "ID": "count",
        "Age": "mean",
        "Income": "mean",
        "Total_Spending": "mean",
        "Total_Purchases": "mean",
        "Average_Basket_Value": "mean",
        "Response": "mean"
    }).round(2)

    table.rename(columns={"ID": "Customer_Count"}, inplace=True)

    return table


#############################################
# EDA & CRM - DATA LOADING
#############################################

df = pd.read_csv(DATA_PATH, sep=";")

print("\n##################### DATA PREVIEW #####################")
print(df.head())

print("\n##################### DATA INFO #####################")
print(df.info())

print("\n##################### DESCRIPTIVE STATISTICS #####################")
print(df.describe().T)


#############################################
# EDA & CRM - DATA PREPARATION
#############################################

df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True,
    errors="coerce"
)

print("\n##################### MISSING VALUES #####################")
print(df.isnull().sum())

df = df[df["Income"].notnull()]

df["Age"] = 2024 - df["Year_Birth"]

df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

df["Has_Children"] = np.where(df["Total_Children"] > 0, 1, 0)

df["Children_Status"] = df["Has_Children"].map({
    0: "No Children",
    1: "Has Children"
})

df["Customer_Tenure_Days"] = (
    df["Dt_Customer"].max() - df["Dt_Customer"]
).dt.days

df["Customer_Tenure_Years"] = (df["Customer_Tenure_Days"] / 365).round(2)


#############################################
# EDA & CRM - FEATURE ENGINEERING
#############################################

product_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

purchase_cols = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]

campaign_cols = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5"
]

df["Total_Spending"] = df[product_cols].sum(axis=1)

df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

df["Average_Basket_Value"] = df["Total_Spending"] / df["Total_Purchases"]

df["Average_Basket_Value"] = (
    df["Average_Basket_Value"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

df["Total_Accepted_Campaigns"] = df[campaign_cols].sum(axis=1)

df["Has_Accepted_Previous_Campaign"] = np.where(
    df["Total_Accepted_Campaigns"] > 0,
    1,
    0
)

df["Previous_Campaign_Status"] = df["Has_Accepted_Previous_Campaign"].map({
    0: "No Previous Acceptance",
    1: "Accepted Previous Campaign"
})

df["Age_Segment"] = pd.cut(
    df["Age"],
    bins=[0, 30, 45, 60, 100],
    labels=["Young", "Adult", "Middle Age", "Senior"]
)

df["Income_Segment"] = pd.qcut(
    df["Income"].rank(method="first"),
    q=4,
    labels=["Low Income", "Medium Income", "High Income", "Premium Income"]
)

df["Spending_Segment"] = pd.qcut(
    df["Total_Spending"].rank(method="first"),
    q=4,
    labels=["Low", "Medium", "High", "VIP"]
)

df["Purchase_Segment"] = pd.qcut(
    df["Total_Purchases"].rank(method="first"),
    q=4,
    labels=["Low Frequency", "Medium Frequency", "High Frequency", "Very High Frequency"]
)


#############################################
# EDA & CRM - GENERAL SUMMARY
#############################################

general_summary = pd.DataFrame({
    "Metric": [
        "Customer Count",
        "Average Age",
        "Average Income",
        "Average Spending",
        "Average Purchases",
        "Average Basket Value",
        "Response Rate",
        "Complaint Rate"
    ],
    "Value": [
        df["ID"].nunique(),
        df["Age"].mean(),
        df["Income"].mean(),
        df["Total_Spending"].mean(),
        df["Total_Purchases"].mean(),
        df["Average_Basket_Value"].mean(),
        df["Response"].mean(),
        df["Complain"].mean()
    ]
})

print("\n##################### GENERAL CUSTOMER PROFILE #####################")
print(general_summary)


#############################################
# EDA & CRM - NUMERIC ANALYSIS
#############################################

numeric_cols = [
    "Age",
    "Income",
    "Recency",
    "Total_Spending",
    "Total_Purchases",
    "Average_Basket_Value",
    "Customer_Tenure_Days",
    "Total_Accepted_Campaigns"
]

numeric_summary = df[numeric_cols].describe().T.round(2)

print("\n##################### NUMERIC VARIABLE SUMMARY #####################")
print(numeric_summary)

plt.figure(figsize=(10, 5))
sns.histplot(df["Age"], bins=25, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df["Income"], bins=30, kde=True)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df["Total_Spending"], bins=30, kde=True)
plt.title("Total Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - CORRELATION ANALYSIS
#############################################

corr_cols = [
    "Age",
    "Income",
    "Recency",
    "Total_Children",
    "Total_Spending",
    "Total_Purchases",
    "Average_Basket_Value",
    "Total_Accepted_Campaigns",
    "Response"
]

correlation_matrix = df[corr_cols].corr().round(2)

print("\n##################### CORRELATION MATRIX #####################")
print(correlation_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - PRODUCT CATEGORY ANALYSIS
#############################################

product_summary = df[product_cols].mean().sort_values(ascending=False)

print("\n##################### PRODUCT CATEGORY ANALYSIS #####################")
print(product_summary)

plt.figure(figsize=(10, 5))
ax = product_summary.plot(kind="bar")
plt.title("Average Spending by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Spending")
plt.xticks(rotation=45)
add_value_labels(ax, fmt="{:.0f}", fontsize=9)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - PURCHASE CHANNEL ANALYSIS
#############################################

channel_summary = df[purchase_cols].mean().sort_values(ascending=False)

print("\n##################### PURCHASE CHANNEL ANALYSIS #####################")
print(channel_summary)

plt.figure(figsize=(9, 5))
ax = channel_summary.plot(kind="bar")
plt.title("Average Purchases by Channel")
plt.xlabel("Purchase Channel")
plt.ylabel("Average Purchases")
plt.xticks(rotation=45)
add_value_labels(ax, fmt="{:.2f}", fontsize=9)
plt.tight_layout()
plt.show()

channel_by_age = df.groupby("Age_Segment")[[
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]].mean().round(2)

print("\n##################### CHANNEL ANALYSIS BY AGE SEGMENT #####################")
print(channel_by_age)

ax = channel_by_age.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Purchase Channels by Age Segment")
plt.xlabel("Age Segment")
plt.ylabel("Average Purchases")
plt.xticks(rotation=45)
plt.legend(title="Channel")
add_value_labels(ax, fmt="{:.2f}", fontsize=8)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - EDUCATION ANALYSIS
#############################################

education_summary = summary_table(df, "Education")

education_summary = education_summary.sort_values(
    by="Total_Spending",
    ascending=False
)

print("\n##################### EDUCATION ANALYSIS #####################")
print(education_summary)

plt.figure(figsize=(10, 5))
ax = sns.barplot(
    x="Education",
    y="Total_Spending",
    data=df,
    estimator=np.mean
)
plt.title("Average Spending by Education")
plt.xlabel("Education")
plt.ylabel("Average Spending")
plt.xticks(rotation=45)
add_value_labels(ax, fmt="{:.0f}", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
ax = sns.barplot(
    x="Education",
    y="Response",
    data=df,
    estimator=np.mean
)
plt.title("Response Rate by Education")
plt.xlabel("Education")
plt.ylabel("Response Rate")
plt.xticks(rotation=45)
add_percent_labels(ax, fontsize=9)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - MARITAL STATUS ANALYSIS
#############################################

marital_summary = summary_table(df, "Marital_Status")

marital_summary = marital_summary.sort_values(
    by="Response",
    ascending=False
)

print("\n##################### MARITAL STATUS ANALYSIS #####################")
print(marital_summary)

plt.figure(figsize=(10, 5))
ax = sns.barplot(
    x="Marital_Status",
    y="Response",
    data=df,
    estimator=np.mean
)
plt.title("Response Rate by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Response Rate")
plt.xticks(rotation=45)
add_percent_labels(ax, fontsize=9)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - CHILDREN STATUS ANALYSIS
#############################################

children_summary = summary_table(df, "Children_Status")

print("\n##################### CHILDREN STATUS ANALYSIS #####################")
print(children_summary)

plt.figure(figsize=(7, 5))
ax = sns.barplot(
    x="Children_Status",
    y="Total_Spending",
    data=df,
    estimator=np.mean
)
plt.title("Average Spending by Children Status")
plt.xlabel("Children Status")
plt.ylabel("Average Spending")
add_value_labels(ax, fmt="{:.0f}", fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
ax = sns.barplot(
    x="Children_Status",
    y="Response",
    data=df,
    estimator=np.mean
)
plt.title("Response Rate by Children Status")
plt.xlabel("Children Status")
plt.ylabel("Response Rate")
add_percent_labels(ax, fontsize=10)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - AGE SEGMENT ANALYSIS
#############################################

age_summary = summary_table(df, "Age_Segment")

print("\n##################### AGE SEGMENT ANALYSIS #####################")
print(age_summary)

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x="Age_Segment",
    y="Response",
    data=df,
    estimator=np.mean,
    order=["Young", "Adult", "Middle Age", "Senior"]
)
plt.title("Response Rate by Age Segment")
plt.xlabel("Age Segment")
plt.ylabel("Response Rate")
add_percent_labels(ax, fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x="Age_Segment",
    y="Total_Spending",
    data=df,
    estimator=np.mean,
    order=["Young", "Adult", "Middle Age", "Senior"]
)
plt.title("Average Spending by Age Segment")
plt.xlabel("Age Segment")
plt.ylabel("Average Spending")
add_value_labels(ax, fmt="{:.0f}", fontsize=9)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - INCOME SEGMENT ANALYSIS
#############################################

income_summary = summary_table(df, "Income_Segment")

print("\n##################### INCOME SEGMENT ANALYSIS #####################")
print(income_summary)

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    x="Income_Segment",
    y="Total_Spending",
    data=df,
    estimator=np.mean,
    order=["Low Income", "Medium Income", "High Income", "Premium Income"]
)
plt.title("Average Spending by Income Segment")
plt.xlabel("Income Segment")
plt.ylabel("Average Spending")
plt.xticks(rotation=15)
add_value_labels(ax, fmt="{:.0f}", fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    x="Income_Segment",
    y="Response",
    data=df,
    estimator=np.mean,
    order=["Low Income", "Medium Income", "High Income", "Premium Income"]
)
plt.title("Response Rate by Income Segment")
plt.xlabel("Income Segment")
plt.ylabel("Response Rate")
plt.xticks(rotation=15)
add_percent_labels(ax, fontsize=9)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - SPENDING SEGMENT ANALYSIS
#############################################

spending_summary = summary_table(df, "Spending_Segment")

print("\n##################### SPENDING SEGMENT ANALYSIS #####################")
print(spending_summary)

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x="Spending_Segment",
    y="Response",
    data=df,
    estimator=np.mean,
    order=["Low", "Medium", "High", "VIP"]
)
plt.title("Response Rate by Spending Segment")
plt.xlabel("Spending Segment")
plt.ylabel("Response Rate")
add_percent_labels(ax, fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x="Spending_Segment",
    y="Total_Purchases",
    data=df,
    estimator=np.mean,
    order=["Low", "Medium", "High", "VIP"]
)
plt.title("Average Purchases by Spending Segment")
plt.xlabel("Spending Segment")
plt.ylabel("Average Purchases")
add_value_labels(ax, fmt="{:.2f}", fontsize=10)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - CAMPAIGN ANALYSIS
#############################################

campaign_success = (df[campaign_cols].mean() * 100).sort_values(ascending=False)

print("\n##################### CAMPAIGN SUCCESS ANALYSIS #####################")
print(campaign_success.round(2))

plt.figure(figsize=(8, 5))
ax = campaign_success.plot(kind="bar")
plt.title("Campaign Acceptance Rates")
plt.xlabel("Campaign")
plt.ylabel("Acceptance Rate (%)")
plt.xticks(rotation=45)
add_value_labels(ax, fmt="{:.1f}%", fontsize=10)
plt.tight_layout()
plt.show()

previous_campaign_summary = summary_table(df, "Previous_Campaign_Status")

print("\n##################### PREVIOUS CAMPAIGN ANALYSIS #####################")
print(previous_campaign_summary)

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x="Previous_Campaign_Status",
    y="Response",
    data=df,
    estimator=np.mean
)
plt.title("Response Rate by Previous Campaign Status")
plt.xlabel("Previous Campaign Status")
plt.ylabel("Response Rate")
plt.xticks(rotation=15)
add_percent_labels(ax, fontsize=10)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - RFM ANALYSIS
#############################################

rfm = df[[
    "ID",
    "Recency",
    "Total_Purchases",
    "Total_Spending"
]].copy()

rfm.columns = [
    "Customer_ID",
    "Recency",
    "Frequency",
    "Monetary"
]

rfm["Recency_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)

rfm["Frequency_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["Monetary_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["RF_SCORE"] = (
    rfm["Recency_Score"].astype(str) +
    rfm["Frequency_Score"].astype(str)
)

rfm["RFM_SCORE"] = (
    rfm["Recency_Score"].astype(str) +
    rfm["Frequency_Score"].astype(str) +
    rfm["Monetary_Score"].astype(str)
)

seg_map = {
    r"[1-2][1-2]": "Hibernating",
    r"[1-2][3-4]": "At_Risk",
    r"[1-2]5": "Cant_Lose",
    r"3[1-2]": "About_To_Sleep",
    r"33": "Need_Attention",
    r"[3-4][4-5]": "Loyal_Customers",
    r"41": "Promising",
    r"51": "New_Customers",
    r"[4-5][2-3]": "Potential_Loyalists",
    r"5[4-5]": "Champions"
}

rfm["RFM_Segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

df = df.merge(
    rfm[[
        "Customer_ID",
        "Recency_Score",
        "Frequency_Score",
        "Monetary_Score",
        "RFM_SCORE",
        "RFM_Segment"
    ]],
    left_on="ID",
    right_on="Customer_ID",
    how="left"
)

df.drop("Customer_ID", axis=1, inplace=True)

rfm_summary = df.groupby("RFM_Segment").agg({
    "ID": "count",
    "Recency": "mean",
    "Total_Purchases": "mean",
    "Total_Spending": "mean",
    "Average_Basket_Value": "mean",
    "Response": "mean"
}).round(2)

rfm_summary.rename(columns={"ID": "Customer_Count"}, inplace=True)

rfm_summary = rfm_summary.sort_values(
    by="Response",
    ascending=False
)

print("\n##################### RFM SEGMENT SUMMARY #####################")
print(rfm_summary)

plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=rfm_summary.index,
    y=rfm_summary["Response"]
)
plt.title("Response Rate by RFM Segment")
plt.xlabel("RFM Segment")
plt.ylabel("Response Rate")
plt.xticks(rotation=45)
add_percent_labels(ax, fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=rfm_summary.index,
    y=rfm_summary["Total_Spending"]
)
plt.title("Average Spending by RFM Segment")
plt.xlabel("RFM Segment")
plt.ylabel("Average Spending")
plt.xticks(rotation=45)
add_value_labels(ax, fmt="{:.0f}", fontsize=8)
plt.tight_layout()
plt.show()


#############################################
# EDA & CRM - COMPLAINT ANALYSIS
#############################################

complaint_summary = df.groupby("Complain").agg({
    "ID": "count",
    "Total_Spending": "mean",
    "Total_Purchases": "mean",
    "Average_Basket_Value": "mean",
    "Response": "mean"
}).round(2)

complaint_summary.rename(columns={"ID": "Customer_Count"}, inplace=True)

print("\n##################### COMPLAINT ANALYSIS #####################")
print(complaint_summary)

plt.figure(figsize=(6, 5))
ax = sns.barplot(
    x="Complain",
    y="Response",
    data=df,
    estimator=np.mean
)
plt.title("Response Rate by Complaint Status")
plt.xlabel("Complaint Status")
plt.ylabel("Response Rate")
add_percent_labels(ax, fontsize=10)
plt.tight_layout()
plt.show()


#############################################
# ML - DATA LOADING
#############################################

df = pd.read_csv(DATA_PATH, sep=";")

df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True,
    errors="coerce"
)

before = len(df)
df = df.dropna(subset=["Income"])
print(f"Rows with missing Income removed   : {before - len(df)}")


#############################################
# ML - OUTLIER CLEANING
#############################################

df["Age"] = 2026 - df["Year_Birth"]

n = len(df)
df = df[df["Income"] < 200_000]
print(f"Income outliers removed     : {n - len(df)} rows")

n = len(df)
df = df[df["Age"] < 90]
print(f"Age outliers removed        : {n - len(df)} rows")
print(f"Cleaned dataset size    : {len(df)}")


#############################################
# ML - FEATURE ENGINEERING
#############################################

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
df["Has_Children"] = np.where(df["Total_Children"] > 0, 1, 0)

df["Total_Spending"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
    df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)

df["Total_Purchases"] = (
    df["NumDealsPurchases"] + df["NumWebPurchases"] +
    df["NumCatalogPurchases"] + df["NumStorePurchases"]
)

df["Average_Basket_Value"] = (
    df["Total_Spending"] / df["Total_Purchases"]
).replace([np.inf, -np.inf], 0).fillna(0)

df["Total_Accepted_Campaigns"] = (
    df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] +
    df["AcceptedCmp4"] + df["AcceptedCmp5"]
)

df["Has_Accepted_Previous_Campaign"] = np.where(
    df["Total_Accepted_Campaigns"] > 0,
    1,
    0
)

df["Customer_Days"] = (
    pd.Timestamp("2014-12-31") - df["Dt_Customer"]
).dt.days.fillna(0).astype(int)

df["Income_per_Person"] = df["Income"] / (df["Total_Children"] + 1)
df["Spending_to_Income"] = df["Total_Spending"] / (df["Income"] + 1)
df["Web_to_Store_Ratio"] = df["NumWebPurchases"] / (df["NumStorePurchases"] + 1)
df["Deal_Rate"] = df["NumDealsPurchases"] / (df["Total_Purchases"] + 1)
df["Meat_Rate"] = df["MntMeatProducts"] / (df["Total_Spending"] + 1)
df["Wines_Rate"] = df["MntWines"] / (df["Total_Spending"] + 1)

df["Age_Segment"] = pd.cut(
    df["Age"],
    bins=[0, 30, 45, 60, 100],
    labels=["Young", "Adult", "Middle Age", "Senior"]
)

df["Spending_Segment"] = pd.qcut(
    df["Total_Spending"].rank(method="first"),
    q=4,
    labels=["Low", "Medium", "High", "VIP"]
)


#############################################
# ML - RFM SEGMENT
#############################################

rfm = df[["ID", "Recency", "Total_Purchases", "Total_Spending"]].copy()

rfm.columns = ["Customer_ID", "Recency", "Frequency", "Monetary"]

rfm["Recency_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)

rfm["Frequency_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["RF_SCORE"] = (
    rfm["Recency_Score"].astype(str) +
    rfm["Frequency_Score"].astype(str)
)

seg_map = {
    r"[1-2][1-2]": "Hibernating",
    r"[1-2][3-4]": "At_Risk",
    r"[1-2]5": "Cant_Lose",
    r"3[1-2]": "About_To_Sleep",
    r"33": "Need_Attention",
    r"[3-4][4-5]": "Loyal_Customers",
    r"41": "Promising",
    r"51": "New_Customers",
    r"[4-5][2-3]": "Potential_Loyalists",
    r"5[4-5]": "Champions"
}

rfm["RFM_Segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

df = df.merge(
    rfm[["Customer_ID", "RFM_Segment"]],
    left_on="ID",
    right_on="Customer_ID",
    how="left"
)

df.drop("Customer_ID", axis=1, inplace=True)


#############################################
# ML - DATA PREPARATION
#############################################

drop_cols = ["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue"]

df_ml = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df_ml.drop("Response", axis=1)
y = df_ml["Response"]

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

print(f"\nSample count    : {X.shape[0]}")
print(f"Feature count  : {X.shape[1]}  ({len(num_cols)} numerical + {len(cat_cols)} categorical)")
print(f"Categorical       : {cat_cols}")
print(f"Positive rate    : %{y.mean() * 100:.1f}")


#############################################
# ML - TRAIN TEST SPLIT
#############################################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain set      : {X_train.shape[0]} samples")
print(f"Test set       : {X_test.shape[0]} samples")


#############################################
# ML - PREPROCESSING
#############################################

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", encoder, cat_cols)
    ],
    remainder="drop"
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


#############################################
# ML - MODELS AND PARAMETERS
#############################################

BEST_PARAMS = {
    "Logistic Regression": {
        "C": 0.1,
        "solver": "lbfgs"
    },
    "Random Forest": {
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "max_samples": 0.8
    },
    "XGBoost": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0
    },
    "LightGBM": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 20,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_samples": 40,
        "reg_alpha": 0.3,
        "reg_lambda": 1.0
    }
}

SEARCH_PARAMS = {
    "Logistic Regression": {
        "model__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        "model__solver": ["lbfgs", "saga"]
    },
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_samples_split": [5, 10, 20],
        "model__min_samples_leaf": [2, 4, 8],
        "model__max_features": ["sqrt", "log2"],
        "model__max_samples": [0.7, 0.8, 0.9]
    },
    "XGBoost": {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__max_depth": [3, 4, 5],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__min_child_weight": [3, 5, 10],
        "model__gamma": [0, 0.1, 0.2, 0.3],
        "model__reg_alpha": [0, 0.1, 0.3, 0.5],
        "model__reg_lambda": [0.5, 1.0, 1.5, 2.0]
    },
    "LightGBM": {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__num_leaves": [15, 20, 31, 40],
        "model__max_depth": [4, 6, 8, -1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__min_child_samples": [20, 30, 40, 50],
        "model__reg_alpha": [0, 0.1, 0.3, 0.5],
        "model__reg_lambda": [0.5, 1.0, 1.5, 2.0]
    }
}

BASE_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=pos_weight
    ),
    "LightGBM": LGBMClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbose=-1
    )
}


#############################################
# ML - MODEL TRAINING
#############################################

results = []
best_models = {}

for model_name, base_model in BASE_MODELS.items():

    print("\n" + "=" * 60)
    print(f"{model_name} {'is being optimized' if RUN_SEARCH else 'is being trained'}")
    print("=" * 60)

    if RUN_SEARCH:
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", base_model)
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=SEARCH_PARAMS[model_name],
            n_iter=50,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0
        )

        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_

        used_params = {
            k.replace("model__", ""): v
            for k, v in search.best_params_.items()
        }

    else:
        base_model.set_params(**BEST_PARAMS[model_name])

        best_pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", base_model)
            ]
        )

        best_pipe.fit(X_train, y_train)

        used_params = BEST_PARAMS[model_name]

    best_models[model_name] = best_pipe

    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    cv_res = cross_validate(
        best_pipe,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall"
        },
        n_jobs=-1
    )

    test_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Test ROC-AUC": round(test_auc, 4),
        "Test F1": round(f1_score(y_test, y_pred), 4),
        "Test Precision": round(precision_score(y_test, y_pred), 4),
        "Test Recall": round(recall_score(y_test, y_pred), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "CV ROC-AUC Mean": round(cv_res["test_roc_auc"].mean(), 4),
        "CV ROC-AUC Std": round(cv_res["test_roc_auc"].std(), 4),
        "CV F1 Mean": round(cv_res["test_f1"].mean(), 4),
        "CV-Test Gap": round(cv_res["test_roc_auc"].mean() - test_auc, 4),
        "Best Params": used_params
    })

    print(f"Params       : {used_params}")
    print(f"Test ROC-AUC : {results[-1]['Test ROC-AUC']}")
    print(f"Test F1      : {results[-1]['Test F1']}")
    print(f"CV ROC-AUC   : {results[-1]['CV ROC-AUC Mean']} +/- {results[-1]['CV ROC-AUC Std']}")
    print(f"CV-Test Gap  : {results[-1]['CV-Test Gap']:+.4f}")


#############################################
# ML - SOFT VOTING ENSEMBLE
#############################################

print("\n" + "=" * 60)
print("Creating Soft Voting Ensemble")
print("=" * 60)

lgbm_pipe = best_models.get("LightGBM")
xgb_pipe = best_models.get("XGBoost")
lr_pipe = best_models.get("Logistic Regression")

voting_clf = VotingClassifier(
    estimators=[
        ("lgbm", lgbm_pipe),
        ("xgb", xgb_pipe),
        ("lr", lr_pipe)
    ],
    voting="soft",
    weights=[2, 2, 1]
)

voting_clf.fit(X_train, y_train)

y_pred_v = voting_clf.predict(X_test)
y_prob_v = voting_clf.predict_proba(X_test)[:, 1]

cv_voting = cross_validate(
    voting_clf,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "roc_auc": "roc_auc",
        "f1": "f1"
    },
    n_jobs=-1
)

results.append({
    "Model": "Soft Voting Ensemble",
    "Test ROC-AUC": round(roc_auc_score(y_test, y_prob_v), 4),
    "Test F1": round(f1_score(y_test, y_pred_v), 4),
    "Test Precision": round(precision_score(y_test, y_pred_v), 4),
    "Test Recall": round(recall_score(y_test, y_pred_v), 4),
    "Test Accuracy": round(accuracy_score(y_test, y_pred_v), 4),
    "CV ROC-AUC Mean": round(cv_voting["test_roc_auc"].mean(), 4),
    "CV ROC-AUC Std": round(cv_voting["test_roc_auc"].std(), 4),
    "CV F1 Mean": round(cv_voting["test_f1"].mean(), 4),
    "CV-Test Gap": round(
        cv_voting["test_roc_auc"].mean() - roc_auc_score(y_test, y_prob_v),
        4
    ),
    "Best Params": {
        "weights": "lgbm:2, xgb:2, lr:1",
        "voting": "soft"
    }
})

best_models["Soft Voting Ensemble"] = voting_clf

print(f"Test ROC-AUC : {results[-1]['Test ROC-AUC']}")
print(f"Test F1      : {results[-1]['Test F1']}")
print(f"CV ROC-AUC   : {results[-1]['CV ROC-AUC Mean']} +/- {results[-1]['CV ROC-AUC Std']}")
print(f"CV-Test Gap  : {results[-1]['CV-Test Gap']:+.4f}")


#############################################
# ML - MODEL COMPARISON
#############################################

comparison_df = (
    pd.DataFrame(results)
    .drop(columns=["Best Params"])
    .sort_values("Test ROC-AUC", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(comparison_df.to_string(index=False))


#############################################
# ML - BEST MODEL DETAILS
#############################################

best_model_name = comparison_df.iloc[0]["Model"]
best_model = best_models[best_model_name]

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name}")
print("=" * 60)

if best_model_name == "Soft Voting Ensemble":
    y_pred_best = y_pred_v
    y_prob_best = y_prob_v
else:
    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_best,
        target_names=["No Response", "Response"]
    )
)

print(f"Test ROC-AUC: {round(roc_auc_score(y_test, y_prob_best), 4)}")


#############################################
# ML - FEATURE IMPORTANCE
#############################################

if best_model_name == "Soft Voting Ensemble":
    source_pipe = best_models["LightGBM"]
    print("\nFeature importance was taken from the LightGBM component.")
else:
    source_pipe = best_model

final_preprocess = source_pipe.named_steps["preprocess"]
final_model = source_pipe.named_steps["model"]

all_features = final_preprocess.get_feature_names_out()

clean_names = [
    f.replace("num__", "").replace("cat__", "")
    for f in all_features
]

if hasattr(final_model, "feature_importances_"):
    fi_df = pd.DataFrame({
        "Feature": clean_names,
        "Importance": final_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    fi_col = "Importance"

else:
    fi_df = pd.DataFrame({
        "Feature": clean_names,
        "Coefficient": np.abs(final_model.coef_[0])
    }).sort_values(
        "Coefficient",
        ascending=False
    ).reset_index(drop=True)

    fi_col = "Coefficient"

fi_df["Rank"] = fi_df.index + 1

fi_df = fi_df[[
    "Rank",
    "Feature",
    fi_col
]]

print("\n" + "=" * 60)
print(f"TOP 20 FEATURES - {best_model_name}")
print("=" * 60)
print(fi_df.head(20).to_string(index=False))