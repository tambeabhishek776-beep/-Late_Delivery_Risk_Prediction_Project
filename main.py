# =====================================================
# MACHINE LEARNING BASED LATE DELIVERY RISK PREDICTION
# ORGANIZATION: UNIFIED MENTOR
# CLIENT: APL LOGISTICS
# =====================================================


# =====================================================
# 1. IMPORT REQUIRED LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')


# =====================================================
# 2. CREATE PROJECT DIRECTORIES
# =====================================================

os.makedirs("graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("predictions", exist_ok=True)


print("\n======================================")
print("LATE DELIVERY PREDICTION PROJECT")
print("======================================")


# =====================================================
# 3. LOAD DATASET
# =====================================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv("APL_Logistics.csv", encoding="cp1252")

print("Dataset Loaded Successfully!")

print("\nDataset Shape:")
print(df.shape)


# =====================================================
# 4. DISPLAY FIRST FIVE RECORDS
# =====================================================

print("\n========== FIRST FIVE RECORDS ==========")

print(df.head())


# =====================================================
# 5. DATASET INFORMATION
# =====================================================

print("\n========== DATASET INFORMATION ==========")

print(df.info())


# =====================================================
# 6. COLUMN NAMES
# =====================================================

print("\n========== COLUMN NAMES ==========")

print(df.columns.tolist())


# =====================================================
# 7. STATISTICAL SUMMARY
# =====================================================

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())


# =====================================================
# 8. CHECK MISSING VALUES
# =====================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

print("\nTotal Missing Values:")

print(df.isnull().sum().sum())


# =====================================================
# 9. CHECK DUPLICATE RECORDS
# =====================================================

print("\n========== DUPLICATE RECORDS ==========")

duplicates = df.duplicated().sum()

print("Total Duplicate Records:", duplicates)


# =====================================================
# 10. CHECK TARGET VARIABLE
# =====================================================

print("\n========== TARGET VARIABLE ANALYSIS ==========")

print(df['Late_delivery_risk'].value_counts())

print("\nTarget Variable Percentage:")

print(
    df['Late_delivery_risk']
    .value_counts(normalize=True) * 100
)


# =====================================================
# 11. CHECK DATA TYPES
# =====================================================

print("\n========== DATA TYPES ==========")

print(df.dtypes)


# =====================================================
# 12. CHECK UNIQUE VALUES
# =====================================================

print("\n========== UNIQUE VALUES ==========")

for column in df.columns:

    print(
        column,
        ":",
        df[column].nunique()
    )


# =====================================================
# 13. SAVE ORIGINAL DATA INFORMATION
# =====================================================

df.to_csv("cleaned_data.csv", index=False)

print("\nDataset Inspection Completed!")

print("Initial Dataset Saved Successfully.")

print("\n======================================")
print("STEP 1 COMPLETED SUCCESSFULLY")
print("======================================")
# =====================================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# =====================================================

print("\n======================================")
print("STEP 2: DATA CLEANING")
print("======================================")


# =====================================================
# 1. CREATE DATASET COPY
# =====================================================

cleaned_df = df.copy()


# =====================================================
# 2. HANDLE DUPLICATE RECORDS
# =====================================================

print("\n========== DUPLICATE HANDLING ==========")

before = cleaned_df.shape[0]

cleaned_df = cleaned_df.drop_duplicates()

after = cleaned_df.shape[0]

print("Records Before:", before)
print("Records After:", after)
print("Duplicates Removed:", before - after)


# =====================================================
# 3. HANDLE MISSING VALUES
# =====================================================

print("\n========== MISSING VALUES HANDLING ==========")

print("Before Cleaning:")

print(cleaned_df.isnull().sum()[cleaned_df.isnull().sum() > 0])


# Customer Lname

cleaned_df['Customer Lname'] = (
    cleaned_df['Customer Lname'].fillna('Unknown')
)


# Customer Zipcode

cleaned_df['Customer Zipcode'] = (
    cleaned_df['Customer Zipcode'].fillna(
        cleaned_df['Customer Zipcode'].mode()[0]
    )
)


print("\nAfter Cleaning:")

print(cleaned_df.isnull().sum().sum())

print("Missing Values Handled Successfully!")


# =====================================================
# 4. HANDLE INCONSISTENT CATEGORICAL VALUES
# =====================================================

print("\n========== CATEGORICAL CLEANING ==========")

categorical_columns = cleaned_df.select_dtypes(
    include=['object']
).columns

for column in categorical_columns:

    cleaned_df[column] = (
        cleaned_df[column]
        .astype('string')
        .str.strip()
    )

print("Categorical Values Cleaned Successfully!")


# =====================================================
# 5. CHECK TARGET VARIABLE
# =====================================================

print("\n========== TARGET VALIDATION ==========")

print(cleaned_df['Late_delivery_risk'].value_counts())

valid_target = cleaned_df[
    'Late_delivery_risk'
].isin([0, 1]).all()

print("Target Variable Valid:", valid_target)

if not valid_target:
    raise ValueError("Invalid Target Values Found!")


# =====================================================
# 6. NUMERICAL DATA VALIDATION
# =====================================================

print("\n========== NUMERICAL VALIDATION ==========")

numerical_columns = cleaned_df.select_dtypes(
    include=['number']
).columns

print("Numerical Columns:", len(numerical_columns))

print("Infinite Values:")

print(
    np.isinf(
        cleaned_df[numerical_columns].to_numpy()
    ).sum()
)


# =====================================================
# 7. CHECK DATA TYPES
# =====================================================

print("\n========== FINAL DATA TYPES ==========")

print(cleaned_df.dtypes)


# =====================================================
# 8. FINAL MISSING VALUE CHECK
# =====================================================

print("\n========== FINAL DATA QUALITY ==========")

print("Total Rows:", cleaned_df.shape[0])

print("Total Columns:", cleaned_df.shape[1])

print("Missing Values:", cleaned_df.isnull().sum().sum())

print("Duplicate Records:", cleaned_df.duplicated().sum())


# =====================================================
# 9. SAVE CLEANED DATASET
# =====================================================

cleaned_df.to_csv(
    "cleaned_data.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")


# =====================================================
# STEP 2 COMPLETED
# =====================================================

print("\n======================================")
print("STEP 2 COMPLETED SUCCESSFULLY")
print("======================================")

# =====================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================

print("\n======================================")
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("======================================")


# =====================================================
# 1. PREPARE EDA DATA
# =====================================================

eda_df = cleaned_df.copy()

sns.set_theme(style="whitegrid")

print("\nTotal Records:", len(eda_df))


# =====================================================
# 2. LATE VS ON-TIME DELIVERY
# =====================================================

print("\n========== DELIVERY DISTRIBUTION ==========")

delivery_counts = eda_df['Late_delivery_risk'].value_counts()

print(delivery_counts)

plt.figure(figsize=(7, 6))

plt.pie(
    delivery_counts,
    labels=['Late Delivery', 'On-Time Delivery'],
    autopct='%1.2f%%',
    startangle=90
)

plt.title("Late vs On-Time Delivery Distribution")

plt.savefig("graphs/01_delivery_distribution.png")

plt.show()


# =====================================================
# 3. SHIPPING MODE ANALYSIS
# =====================================================

print("\n========== SHIPPING MODE ANALYSIS ==========")

shipping_analysis = eda_df.groupby(
    'Shipping Mode'
)['Late_delivery_risk'].mean() * 100

print(shipping_analysis.sort_values(ascending=False))

plt.figure(figsize=(9, 6))

plt.bar(
    shipping_analysis.index,
    shipping_analysis.values
)

plt.title("Late Delivery Rate by Shipping Mode")

plt.xlabel("Shipping Mode")

plt.ylabel("Late Delivery Rate (%)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("graphs/02_shipping_mode_analysis.png")

plt.show()


# =====================================================
# 4. SCHEDULED SHIPPING DAYS
# =====================================================

print("\n========== SCHEDULED SHIPPING DAYS ==========")

scheduled_analysis = eda_df.groupby(
    'Days for shipment (scheduled)'
)['Late_delivery_risk'].mean() * 100

print(scheduled_analysis)

plt.figure(figsize=(8, 5))

plt.bar(
    scheduled_analysis.index.astype(str),
    scheduled_analysis.values
)

plt.title("Late Delivery Rate by Scheduled Shipping Days")

plt.xlabel("Scheduled Shipping Days")

plt.ylabel("Late Delivery Rate (%)")

plt.tight_layout()

plt.savefig("graphs/03_scheduled_days.png")

plt.show()


# =====================================================
# 5. MARKET-WISE ANALYSIS
# =====================================================

print("\n========== MARKET-WISE ANALYSIS ==========")

market_analysis = eda_df.groupby(
    'Market'
)['Late_delivery_risk'].mean() * 100

print(market_analysis.sort_values(ascending=False))

plt.figure(figsize=(9, 6))

plt.bar(
    market_analysis.index,
    market_analysis.values
)

plt.title("Market-wise Late Delivery Rate")

plt.xlabel("Market")

plt.ylabel("Late Delivery Rate (%)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("graphs/04_market_analysis.png")

plt.show()


# =====================================================
# 6. REGION-WISE ANALYSIS
# =====================================================

print("\n========== REGION-WISE ANALYSIS ==========")

region_analysis = (
    eda_df.groupby('Order Region')['Late_delivery_risk']
    .agg(['mean', 'count'])
)

region_analysis['Late_Rate'] = region_analysis['mean'] * 100

region_analysis = region_analysis.sort_values(
    by='Late_Rate',
    ascending=False
)

print(region_analysis[['count', 'Late_Rate']])

plt.figure(figsize=(13, 7))

plt.barh(
    region_analysis.index,
    region_analysis['Late_Rate']
)

plt.title("Region-wise Late Delivery Rate")

plt.xlabel("Late Delivery Rate (%)")

plt.ylabel("Order Region")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("graphs/05_region_analysis.png")

plt.show()


# =====================================================
# 7. CUSTOMER SEGMENT ANALYSIS
# =====================================================

print("\n========== CUSTOMER SEGMENT ANALYSIS ==========")

segment_analysis = eda_df.groupby(
    'Customer Segment'
)['Late_delivery_risk'].mean() * 100

print(segment_analysis)

plt.figure(figsize=(8, 5))

plt.bar(
    segment_analysis.index,
    segment_analysis.values
)

plt.title("Late Delivery Rate by Customer Segment")

plt.xlabel("Customer Segment")

plt.ylabel("Late Delivery Rate (%)")

plt.tight_layout()

plt.savefig("graphs/06_customer_segment.png")

plt.show()


# =====================================================
# 8. PRODUCT CATEGORY ANALYSIS
# =====================================================

print("\n========== PRODUCT CATEGORY ANALYSIS ==========")

category_analysis = (
    eda_df.groupby('Category Name')['Late_delivery_risk']
    .agg(['mean', 'count'])
)

category_analysis['Late_Rate'] = category_analysis['mean'] * 100

top_categories = category_analysis.sort_values(
    by='count',
    ascending=False
).head(10)

print(top_categories[['count', 'Late_Rate']])

plt.figure(figsize=(12, 6))

plt.bar(
    top_categories.index,
    top_categories['Late_Rate']
)

plt.title("Top 10 Product Categories - Late Delivery Rate")

plt.xlabel("Product Category")

plt.ylabel("Late Delivery Rate (%)")

plt.xticks(rotation=45, ha='right')

plt.tight_layout()

plt.savefig("graphs/07_category_analysis.png")

plt.show()


# =====================================================
# 9. ORDER QUANTITY ANALYSIS
# =====================================================

print("\n========== ORDER QUANTITY ANALYSIS ==========")

quantity_analysis = eda_df.groupby(
    'Order Item Quantity'
)['Late_delivery_risk'].mean() * 100

print(quantity_analysis)

plt.figure(figsize=(8, 5))

plt.bar(
    quantity_analysis.index.astype(str),
    quantity_analysis.values
)

plt.title("Late Delivery Rate by Order Quantity")

plt.xlabel("Order Quantity")

plt.ylabel("Late Delivery Rate (%)")

plt.tight_layout()

plt.savefig("graphs/08_quantity_analysis.png")

plt.show()


# =====================================================
# 10. PAYMENT TYPE ANALYSIS
# =====================================================

print("\n========== PAYMENT TYPE ANALYSIS ==========")

payment_analysis = eda_df.groupby(
    'Type'
)['Late_delivery_risk'].mean() * 100

print(payment_analysis)

plt.figure(figsize=(8, 5))

plt.bar(
    payment_analysis.index,
    payment_analysis.values
)

plt.title("Late Delivery Rate by Payment Type")

plt.xlabel("Payment Type")

plt.ylabel("Late Delivery Rate (%)")

plt.tight_layout()

plt.savefig("graphs/09_payment_analysis.png")

plt.show()


# =====================================================
# 11. CORRELATION ANALYSIS
# =====================================================

print("\n========== CORRELATION ANALYSIS ==========")

correlation_columns = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Sales per customer',
    'Late_delivery_risk',
    'Order Item Discount',
    'Order Item Discount Rate',
    'Order Item Product Price',
    'Order Item Profit Ratio',
    'Order Item Quantity',
    'Sales',
    'Order Item Total',
    'Order Profit Per Order',
    'Product Price'
]

correlation_matrix = eda_df[
    correlation_columns
].corr()

print(correlation_matrix['Late_delivery_risk'].sort_values(
    ascending=False
))

plt.figure(figsize=(13, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("graphs/10_correlation_heatmap.png")

plt.show()


# =====================================================
# 12. ACTUAL VS SCHEDULED SHIPPING DAYS
# =====================================================

print("\n========== ACTUAL VS SCHEDULED SHIPPING ==========")

shipping_comparison = eda_df.groupby(
    'Days for shipment (scheduled)'
)['Days for shipping (real)'].mean()

print(shipping_comparison)

plt.figure(figsize=(8, 5))

plt.plot(
    shipping_comparison.index,
    shipping_comparison.values,
    marker='o',
    label='Average Actual Shipping Days'
)

plt.plot(
    shipping_comparison.index,
    shipping_comparison.index,
    marker='o',
    linestyle='--',
    label='Scheduled Shipping Days'
)

plt.title("Actual vs Scheduled Shipping Days")

plt.xlabel("Scheduled Shipping Days")

plt.ylabel("Shipping Days")

plt.legend()

plt.tight_layout()

plt.savefig("graphs/11_shipping_comparison.png")

plt.show()


# =====================================================
# 13. DELIVERY STATUS ANALYSIS
# =====================================================

print("\n========== DELIVERY STATUS ANALYSIS ==========")

status_analysis = eda_df[
    'Delivery Status'
].value_counts()

print(status_analysis)

plt.figure(figsize=(9, 6))

plt.bar(
    status_analysis.index,
    status_analysis.values
)

plt.title("Delivery Status Distribution")

plt.xlabel("Delivery Status")

plt.ylabel("Number of Orders")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("graphs/12_delivery_status.png")

plt.show()


# =====================================================
# 14. SAVE EDA RESULTS
# =====================================================

market_analysis.to_csv(
    "reports/market_analysis.csv"
)

region_analysis.to_csv(
    "reports/region_analysis.csv"
)

shipping_analysis.to_csv(
    "reports/shipping_mode_analysis.csv"
)

segment_analysis.to_csv(
    "reports/customer_segment_analysis.csv"
)

category_analysis.to_csv(
    "reports/category_analysis.csv"
)


# =====================================================
# 15. EDA COMPLETION
# =====================================================

print("\n======================================")

print("EDA COMPLETED SUCCESSFULLY")

print("Total Visualizations: 12")

print("Graphs Saved in graphs Folder")

print("EDA Reports Saved in reports Folder")

print("======================================")

# =====================================================
# STEP 4: FEATURE ENGINEERING
# =====================================================

print("\n======================================")
print("STEP 4: FEATURE ENGINEERING")
print("======================================")


# =====================================================
# 1. CREATE FEATURE ENGINEERING DATASET
# =====================================================

feature_df = cleaned_df.copy()


# =====================================================
# 2. SHIPPING PRESSURE INDEX
# =====================================================

print("\n========== SHIPPING PRESSURE INDEX ==========")

feature_df['Shipping Pressure Index'] = (
    feature_df['Order Item Quantity'] /
    feature_df['Days for shipment (scheduled)'].clip(lower=1)
)

print(feature_df['Shipping Pressure Index'].describe())


# =====================================================
# 3. MODE RISK FLAGS
# =====================================================

print("\n========== MODE RISK FLAGS ==========")

feature_df['Express Mode Flag'] = (
    feature_df['Shipping Mode']
    .str.contains('First|Second', case=False)
    .astype(int)
)

feature_df['Standard Mode Flag'] = (
    feature_df['Shipping Mode']
    .str.contains('Standard', case=False)
    .astype(int)
)

feature_df['Same Day Mode Flag'] = (
    feature_df['Shipping Mode']
    .str.contains('Same Day', case=False)
    .astype(int)
)

print(
    feature_df[
        [
            'Shipping Mode',
            'Express Mode Flag',
            'Standard Mode Flag',
            'Same Day Mode Flag'
        ]
    ].head()
)


# =====================================================
# 4. REGIONAL CONGESTION INDICATOR
# =====================================================

print("\n========== REGIONAL CONGESTION INDICATOR ==========")

# Historical regional risk will be calculated
# from training data only during model development.
# Here we create the regional category for later use.

feature_df['Regional Congestion Indicator'] = (
    feature_df['Order Region'].astype(str)
)

print(
    feature_df[
        ['Order Region', 'Regional Congestion Indicator']
    ].head()
)


# =====================================================
# 5. ORDER COMPLEXITY SCORE
# =====================================================

print("\n========== ORDER COMPLEXITY SCORE ==========")

feature_df['Order Complexity Score'] = (
    feature_df['Order Item Quantity'] *
    feature_df['Order Item Product Price']
)

print(
    feature_df['Order Complexity Score'].describe()
)


# =====================================================
# 6. ADDITIONAL FEATURE ENGINEERING
# =====================================================

print("\n========== ADDITIONAL FEATURES ==========")


# Discount Impact

feature_df['Discount Impact'] = (
    feature_df['Order Item Discount Rate'] *
    feature_df['Order Item Quantity']
)


# Product Value

feature_df['Product Value'] = (
    feature_df['Order Item Product Price'] *
    feature_df['Order Item Quantity']
)


# Scheduled Shipping Category

feature_df['Scheduled Shipping Category'] = pd.cut(
    feature_df['Days for shipment (scheduled)'],
    bins=[-1, 1, 2, 3, 10],
    labels=[
        'Very Short',
        'Short',
        'Medium',
        'Long'
    ]
)


# Order Quantity Category

feature_df['Order Quantity Category'] = pd.cut(
    feature_df['Order Item Quantity'],
    bins=[0, 1, 3, 10],
    labels=[
        'Low',
        'Medium',
        'High'
    ]
)


# =====================================================
# 7. DISPLAY NEW FEATURES
# =====================================================

print("\n========== ENGINEERED FEATURES ==========")

new_features = [
    'Shipping Pressure Index',
    'Express Mode Flag',
    'Standard Mode Flag',
    'Same Day Mode Flag',
    'Regional Congestion Indicator',
    'Order Complexity Score',
    'Discount Impact',
    'Product Value',
    'Scheduled Shipping Category',
    'Order Quantity Category'
]

print(feature_df[new_features].head(10))


# =====================================================
# 8. FEATURE SUMMARY
# =====================================================

print("\n========== FEATURE SUMMARY ==========")

print("Original Columns:", cleaned_df.shape[1])

print("New Features:", len(new_features))

print("Total Columns:", feature_df.shape[1])


# =====================================================
# 9. SAVE FEATURE ENGINEERED DATASET
# =====================================================

feature_df.to_csv(
    "feature_engineered_data.csv",
    index=False
)

print("\nFeature Engineered Dataset Saved Successfully!")


# =====================================================
# STEP 4 COMPLETED
# =====================================================

print("\n======================================")

print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")

print("======================================")

# ==========================================
# STEP 5: MACHINE LEARNING MODEL BUILDING
# ==========================================

print("\n==========================================")
print("STEP 5: MACHINE LEARNING MODEL BUILDING")
print("==========================================")

# ==========================================
# STEP 5.1: FEATURE SELECTION
# ==========================================

print("\n========== FEATURE SELECTION ==========")

import pandas as pd
from sklearn.model_selection import train_test_split

# Use Feature Engineered Dataset
model_df = feature_df.copy()

# Numerical Features
numerical_features = [
    'Days for shipment (scheduled)',
    'Category Id',
    'Department Id',
    'Latitude',
    'Longitude',
    'Order Item Discount',
    'Order Item Discount Rate',
    'Order Item Product Price',
    'Order Item Quantity',
    'Product Price',
    'Shipping Pressure Index',
    'Express Mode Flag',
    'Standard Mode Flag',
    'Same Day Mode Flag',
    'Order Complexity Score',
    'Discount Impact',
    'Product Value'
]

# Categorical Features
categorical_features = [
    'Type',
    'Category Name',
    'Customer Country',
    'Customer Segment',
    'Department Name',
    'Market',
    'Order Country',
    'Order Region',
    'Product Name',
    'Shipping Mode',
    'Scheduled Shipping Category',
    'Order Quantity Category',
    'Regional Congestion Indicator'
]

# Combine Features
selected_features = numerical_features + categorical_features

# Input Variables
X = model_df[selected_features].copy()

# Target Variable
y = model_df['Late_delivery_risk'].copy()

print("\nTotal Selected Features:", len(selected_features))
print("Numerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))

print("\nInput Data Shape:", X.shape)
print("Target Data Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

print("\n========== TRAIN TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nSTEP 5.1 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 5.2: ENCODING AND SCALING
# ==========================================

print("\n========== ENCODING AND SCALING ==========")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

# Move ID columns into categorical features
numerical_features.remove('Category Id')
numerical_features.remove('Department Id')

categorical_features.extend([
    'Category Id',
    'Department Id'
])

# Rebuild X with updated feature types
selected_features = numerical_features + categorical_features

X = model_df[selected_features].copy()
y = model_df['Late_delivery_risk'].copy()

# Train-Test Split again
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# One-Hot Encoder compatibility
try:
    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse=True
    )

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(with_mean=False), numerical_features),
        ('cat', encoder, categorical_features)
    ],
    remainder='drop'
)

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

print("\nOriginal Training Shape:", X_train.shape)
print("Processed Training Shape:", X_train_processed.shape)

print("\nOriginal Testing Shape:", X_test.shape)
print("Processed Testing Shape:", X_test_processed.shape)

print("\nSparse Matrix:", sparse.issparse(X_train_processed))

print("\nENCODING AND SCALING COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 5.3: LOGISTIC REGRESSION MODEL
# ==========================================

print("\n==========================================")
print("LOGISTIC REGRESSION MODEL")
print("==========================================")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Create Model
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# Train Model
print("\nTraining Logistic Regression Model...")

lr_model.fit(X_train_processed, y_train)

print("Model Training Completed!")

# Predictions
y_pred_lr = lr_model.predict(X_test_processed)

# Prediction Probabilities
y_prob_lr = lr_model.predict_proba(X_test_processed)[:, 1]

# Model Evaluation
print("\n========== MODEL EVALUATION ==========")

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_roc_auc = roc_auc_score(y_test, y_prob_lr)

print("\nAccuracy:", round(lr_accuracy, 4))
print("Precision:", round(lr_precision, 4))
print("Recall:", round(lr_recall, 4))
print("F1 Score:", round(lr_f1, 4))
print("ROC-AUC:", round(lr_roc_auc, 4))

# Classification Report
print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred_lr))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Late', 'Late'],
    yticklabels=['Not Late', 'Late']
)

plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('graphs/logistic_regression_confusion_matrix.png')
plt.close()

print("\nConfusion Matrix Saved Successfully!")

# Save Results
model_results = {
    'Logistic Regression': {
        'Accuracy': lr_accuracy,
        'Precision': lr_precision,
        'Recall': lr_recall,
        'F1 Score': lr_f1,
        'ROC-AUC': lr_roc_auc
    }
}

print("\nSTEP 5.3 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 5.4: RANDOM FOREST MODEL
# ==========================================

print("\n==========================================")
print("RANDOM FOREST MODEL")
print("==========================================")

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Train Model
print("\nTraining Random Forest Model...")

rf_model.fit(X_train_processed, y_train)

print("Random Forest Training Completed!")

# Predictions
y_pred_rf = rf_model.predict(X_test_processed)

# Prediction Probabilities
y_prob_rf = rf_model.predict_proba(X_test_processed)[:, 1]

# Evaluation
print("\n========== RANDOM FOREST EVALUATION ==========")

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_prob_rf)

print("\nAccuracy:", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1 Score:", round(rf_f1, 4))
print("ROC-AUC:", round(rf_roc_auc, 4))

# Classification Report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Not Late', 'Late'],
    yticklabels=['Not Late', 'Late']
)

plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('graphs/random_forest_confusion_matrix.png')
plt.close()

print("\nRandom Forest Confusion Matrix Saved!")

# Update Model Results
model_results['Random Forest'] = {
    'Accuracy': rf_accuracy,
    'Precision': rf_precision,
    'Recall': rf_recall,
    'F1 Score': rf_f1,
    'ROC-AUC': rf_roc_auc
}

print("\nSTEP 5.4 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 5.5: GRADIENT BOOSTING MODEL
# ==========================================

print("\n==========================================")
print("GRADIENT BOOSTING MODEL")
print("==========================================")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Convert Sparse Matrix to Dense Float32
print("\nPreparing Data for Gradient Boosting...")

X_train_gb = X_train_processed.toarray().astype("float32")
X_test_gb = X_test_processed.toarray().astype("float32")

print("Training Data Shape:", X_train_gb.shape)
print("Testing Data Shape:", X_test_gb.shape)

# Balanced Sample Weights
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

# Create Model
gb_model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=42
)

# Train Model
print("\nTraining Gradient Boosting Model...")

gb_model.fit(
    X_train_gb,
    y_train,
    sample_weight=sample_weights
)

print("Gradient Boosting Training Completed!")

# Predictions
y_pred_gb = gb_model.predict(X_test_gb)

# Prediction Probabilities
y_prob_gb = gb_model.predict_proba(X_test_gb)[:, 1]

# Evaluation
print("\n========== GRADIENT BOOSTING EVALUATION ==========")

gb_accuracy = accuracy_score(y_test, y_pred_gb)
gb_precision = precision_score(y_test, y_pred_gb)
gb_recall = recall_score(y_test, y_pred_gb)
gb_f1 = f1_score(y_test, y_pred_gb)
gb_roc_auc = roc_auc_score(y_test, y_prob_gb)

print("\nAccuracy:", round(gb_accuracy, 4))
print("Precision:", round(gb_precision, 4))
print("Recall:", round(gb_recall, 4))
print("F1 Score:", round(gb_f1, 4))
print("ROC-AUC:", round(gb_roc_auc, 4))

# Classification Report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred_gb))

# Confusion Matrix
cm_gb = confusion_matrix(y_test, y_pred_gb)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_gb,
    annot=True,
    fmt='d',
    cmap='Oranges',
    xticklabels=['Not Late', 'Late'],
    yticklabels=['Not Late', 'Late']
)

plt.title('Gradient Boosting - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('graphs/gradient_boosting_confusion_matrix.png')
plt.close()

print("\nGradient Boosting Confusion Matrix Saved!")

# Update Model Results
model_results['Gradient Boosting'] = {
    'Accuracy': gb_accuracy,
    'Precision': gb_precision,
    'Recall': gb_recall,
    'F1 Score': gb_f1,
    'ROC-AUC': gb_roc_auc
}

print("\nSTEP 5.5 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 5.6: FINAL MODEL COMPARISON
# ==========================================

print("\n==========================================")
print("FINAL MODEL COMPARISON")
print("==========================================")

comparison_df = pd.DataFrame(model_results).T

comparison_df = comparison_df.sort_values(
    by='ROC-AUC',
    ascending=False
)

print("\n========== MODEL PERFORMANCE ==========")
print(comparison_df.round(4))

# Save Comparison
comparison_df.to_csv(
    'reports/model_comparison.csv'
)

# Plot Comparison
comparison_df.plot(
    kind='bar',
    figsize=(12, 6)
)

plt.title('Machine Learning Model Comparison')
plt.ylabel('Score')
plt.xlabel('Models')
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig('graphs/model_comparison.png')
plt.close()

print("\nModel Comparison Saved Successfully!")

print("\nBest Model Based on ROC-AUC:")
print(comparison_df.index[0])

print("\nSTEP 5.6 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 6 - FEATURE IMPORTANCE ANALYSIS
# ==========================================

print("\n==========================================")
print("STEP 6 - FEATURE IMPORTANCE ANALYSIS")
print("==========================================")

import pandas as pd
import matplotlib.pyplot as plt

# Get Feature Importance from Random Forest
importance = rf_model.feature_importances_

# Get Processed Feature Names
feature_names = preprocessor.get_feature_names_out()

# Create DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# Sort Features
feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

# Display Top 20 Features
print("\n========== TOP 20 IMPORTANT FEATURES ==========")
print(feature_importance.head(20).to_string(index=False))

# Save Complete Feature Importance Report
feature_importance.to_csv(
    'reports/feature_importance.csv',
    index=False
)

# Plot Top 20 Features
top_features = feature_importance.head(20)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features['Feature'][::-1],
    top_features['Importance'][::-1]
)

plt.title('Top 20 Features - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.tight_layout()

plt.savefig(
    'graphs/feature_importance.png',
    dpi=300
)

plt.close()

print("\nFeature Importance Report Saved Successfully!")
print("Graph Saved Successfully!")

print("\nSTEP 6 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 7 - ORDER LEVEL RISK PREDICTION
# ==========================================

print("\n==========================================")
print("STEP 7 - ORDER LEVEL RISK PREDICTION")
print("==========================================")

import numpy as np
import pandas as pd

# Predict Late Delivery Probability
risk_probability = rf_model.predict_proba(
    X_test_processed
)[:, 1]

# Predicted Class
predicted_class = (risk_probability >= 0.50).astype(int)

# Create Risk Category
def assign_risk(probability):

    if probability < 0.40:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    else:
        return "High"


risk_category = [
    assign_risk(probability)
    for probability in risk_probability
]

# Get Original Test Data
order_risk_df = X_test.copy()

# Add Prediction Details
order_risk_df["Actual Late Risk"] = y_test.values

order_risk_df["Late Delivery Probability"] = risk_probability

order_risk_df["Predicted Late Risk"] = predicted_class

order_risk_df["Risk Category"] = risk_category

# Add useful order information
order_risk_df["Shipping Mode"] = cleaned_df.loc[
    X_test.index, "Shipping Mode"
].values

order_risk_df["Order Region"] = cleaned_df.loc[
    X_test.index, "Order Region"
].values

order_risk_df["Market"] = cleaned_df.loc[
    X_test.index, "Market"
].values

order_risk_df["Customer Segment"] = cleaned_df.loc[
    X_test.index, "Customer Segment"
].values

# Display Results
print("\n========== FIRST 10 ORDER PREDICTIONS ==========")

print(
    order_risk_df[
        [
            "Actual Late Risk",
            "Late Delivery Probability",
            "Predicted Late Risk",
            "Risk Category",
            "Shipping Mode",
            "Order Region"
        ]
    ].head(10).to_string(index=False)
)

# Risk Category Count
print("\n========== RISK CATEGORY DISTRIBUTION ==========")

print(order_risk_df["Risk Category"].value_counts())

# Save Report
order_risk_df.to_csv(
    "reports/order_risk_predictions.csv",
    index=False
)

print("\nOrder Risk Prediction Report Saved Successfully!")

print("\nSTEP 7 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 8 - HIGH RISK ORDERS ANALYSIS
# ==========================================

print("\n==========================================")
print("STEP 8 - HIGH RISK ORDERS ANALYSIS")
print("==========================================")

# Select High Risk Orders
high_risk_orders = order_risk_df[
    order_risk_df["Risk Category"] == "High"
].copy()

# Sort by Probability
high_risk_orders = high_risk_orders.sort_values(
    by="Late Delivery Probability",
    ascending=False
)

# Display Top 20 High Risk Orders
print("\n========== TOP 20 HIGH RISK ORDERS ==========")

display_columns = [
    "Late Delivery Probability",
    "Shipping Mode",
    "Order Region",
    "Market",
    "Customer Segment",
    "Days for shipment (scheduled)",
    "Order Item Quantity"
]

print(
    high_risk_orders[display_columns]
    .head(20)
    .to_string(index=False)
)

# High Risk Count
print("\nTotal High Risk Orders:", len(high_risk_orders))

# Shipping Mode Analysis
print("\n========== HIGH RISK BY SHIPPING MODE ==========")

print(
    high_risk_orders["Shipping Mode"]
    .value_counts()
)

# Region Analysis
print("\n========== HIGH RISK BY REGION ==========")

print(
    high_risk_orders["Order Region"]
    .value_counts()
    .head(10)
)

# Save High Risk Report
high_risk_orders.to_csv(
    "reports/high_risk_orders.csv",
    index=False
)

print("\nHigh Risk Orders Report Saved Successfully!")

print("\nSTEP 8 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 9 - OPERATIONS ACTION PANEL
# ==========================================

print("\n==========================================")
print("STEP 9 - OPERATIONS ACTION PANEL")
print("==========================================")

# Create Operations Action Report
operations_df = high_risk_orders.copy()

# Assign Priority
def assign_priority(probability):

    if probability >= 0.90:
        return "Critical"

    elif probability >= 0.80:
        return "High"

    else:
        return "Medium"


operations_df["Priority"] = operations_df[
    "Late Delivery Probability"
].apply(assign_priority)


# Assign Recommended Action
def recommend_action(row):

    if row["Shipping Mode"] == "First Class":
        return "Check express carrier capacity and dispatch immediately"

    elif row["Shipping Mode"] == "Second Class":
        return "Review dispatch schedule and transit route"

    else:
        return "Review shipment and monitor delivery progress"


operations_df["Recommended Action"] = operations_df.apply(
    recommend_action,
    axis=1
)


# Add Dataset Row Reference
operations_df["Dataset Row Reference"] = (
    operations_df.index + 1
)


# Display Top 15 Priority Orders
print("\n========== IMMEDIATE ATTENTION ORDERS ==========")

print(
    operations_df[
        [
            "Dataset Row Reference",
            "Late Delivery Probability",
            "Priority",
            "Shipping Mode",
            "Order Region",
            "Recommended Action"
        ]
    ].head(15).to_string(index=False)
)


# Priority Distribution
print("\n========== PRIORITY DISTRIBUTION ==========")

print(
    operations_df["Priority"].value_counts()
)


# Save Operations Report
operations_df.to_csv(
    "reports/operations_action_report.csv",
    index=False
)

print("\nOperations Action Report Saved Successfully!")

print("\nSTEP 9 COMPLETED SUCCESSFULLY!")

# ==========================================
# STEP 10.8 - MODEL COMPARISON REPORT
# ==========================================

import pandas as pd
import os

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [
        0.6959,
        0.6979,
        0.6977
    ],
    "Precision": [
        0.8466,
        0.8491,
        0.8490
    ],
    "Recall": [
        0.5438,
        0.5462,
        0.5457
    ],
    "F1 Score": [
        0.6623,
        0.6647,
        0.6644
    ],
    "ROC-AUC": [
        0.7423,
        0.7553,
        0.7547
    ]
})

os.makedirs("reports", exist_ok=True)

model_comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON REPORT ==========")
print(model_comparison)

print("\nModel Comparison CSV Saved Successfully!")