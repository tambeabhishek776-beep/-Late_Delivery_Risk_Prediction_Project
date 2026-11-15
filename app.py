import streamlit as st
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Late Delivery Risk Dashboard",
    page_icon="🚚",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    orders = pd.read_csv("reports/order_risk_predictions.csv")

    high_risk = pd.read_csv("reports/high_risk_orders.csv")

    operations = pd.read_csv("reports/operations_action_report.csv")

    importance = pd.read_csv("reports/feature_importance.csv")

    return orders, high_risk, operations, importance


try:
    orders, high_risk, operations, importance_df = load_data()

except Exception as e:
    st.error(f"Data Loading Error: {e}")
    st.stop()


# ==========================================
# DASHBOARD TITLE
# ==========================================

st.title("🚚 Late Delivery Risk Prediction Dashboard")

st.markdown(
    "Machine Learning-Based Late Delivery Risk Prediction "
    "in Global Supply Chain Operations"
)

st.divider()


# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

shipping_modes = st.sidebar.multiselect(
    "Shipping Mode",
    options=sorted(orders["Shipping Mode"].dropna().unique()),
    default=sorted(orders["Shipping Mode"].dropna().unique())
)

regions = st.sidebar.multiselect(
    "Order Region",
    options=sorted(orders["Order Region"].dropna().unique()),
    default=sorted(orders["Order Region"].dropna().unique())
)

markets = st.sidebar.multiselect(
    "Market",
    options=sorted(orders["Market"].dropna().unique()),
    default=sorted(orders["Market"].dropna().unique())
)

segments = st.sidebar.multiselect(
    "Customer Segment",
    options=sorted(orders["Customer Segment"].dropna().unique()),
    default=sorted(orders["Customer Segment"].dropna().unique())
)

risk_threshold = st.sidebar.slider(
    "High Risk Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05
)


# ==========================================
# APPLY FILTERS
# ==========================================

filtered_orders = orders[
    (orders["Shipping Mode"].isin(shipping_modes)) &
    (orders["Order Region"].isin(regions)) &
    (orders["Market"].isin(markets)) &
    (orders["Customer Segment"].isin(segments))
].copy()

filtered_orders["Risk Category"] = filtered_orders[
    "Late Delivery Probability"
].apply(
    lambda x: "High" if x >= risk_threshold
    else "Low" if x < 0.40
    else "Medium"
)


# ==========================================
# KPI OVERVIEW
# ==========================================

total_orders = len(filtered_orders)

high_risk_count = (
    filtered_orders["Late Delivery Probability"]
    >= risk_threshold
).sum()

average_probability = (
    filtered_orders["Late Delivery Probability"].mean()
    if total_orders > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", f"{total_orders:,}")

col2.metric("High Risk Orders", f"{high_risk_count:,}")

col3.metric(
    "Average Late Risk",
    f"{average_probability * 100:.2f}%"
)

col4.metric(
    "Risk Threshold",
    f"{risk_threshold * 100:.0f}%"
)

st.divider()


# ==========================================
# 1. DELAY RISK OVERVIEW
# ==========================================

st.subheader("1. Delay Risk Overview")

if total_orders > 0:

    risk_counts = (
        filtered_orders["Risk Category"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )

    st.bar_chart(risk_counts)

    st.write("Risk Category Distribution")

    st.dataframe(
        risk_counts,
        use_container_width=True
    )

else:

    st.warning("No orders match the selected filters.")


# ==========================================
# 2. ORDER LEVEL RISK PREDICTION
# ==========================================

st.divider()

st.subheader("2. Order-Level Risk Prediction")

if total_orders > 0:

    st.dataframe(
        filtered_orders[
            [
                "Late Delivery Probability",
                "Predicted Late Risk",
                "Risk Category",
                "Shipping Mode",
                "Order Region",
                "Market",
                "Customer Segment"
            ]
        ].sort_values(
            by="Late Delivery Probability",
            ascending=False
        ),
        use_container_width=True
    )

else:

    st.info("No order predictions available.")


# ==========================================
# 3. REGION & SHIPPING MODE ANALYSIS
# ==========================================

st.divider()

st.subheader("3. Region & Shipping Mode Risk Analysis")

if total_orders > 0:

    region_risk = (
        filtered_orders.groupby("Order Region")[
            "Late Delivery Probability"
        ].mean() * 100
    )

    st.write("Average Risk by Region (%)")

    st.bar_chart(
        region_risk.sort_values(ascending=False)
    )

    mode_risk = (
        filtered_orders.groupby("Shipping Mode")[
            "Late Delivery Probability"
        ].mean() * 100
    )

    st.write("Average Risk by Shipping Mode (%)")

    st.bar_chart(
        mode_risk.sort_values(ascending=False)
    )


# ==========================================
# 4. OPERATIONS ACTION PANEL
# ==========================================

st.divider()

st.subheader("4. Operations Action Panel")

action_orders = filtered_orders[
    filtered_orders["Late Delivery Probability"]
    >= risk_threshold
].copy()


def get_action(mode):

    if mode == "First Class":

        return "Check express carrier capacity and dispatch immediately"

    elif mode == "Second Class":

        return "Review dispatch schedule and transit route"

    elif mode == "Same Day":

        return "Verify immediate dispatch and delivery capacity"

    else:

        return "Review shipment and monitor delivery progress"


action_orders["Recommended Action"] = (
    action_orders["Shipping Mode"].apply(get_action)
)

if len(action_orders) > 0:

    st.dataframe(
        action_orders[
            [
                "Late Delivery Probability",
                "Shipping Mode",
                "Order Region",
                "Market",
                "Recommended Action"
            ]
        ].sort_values(
            by="Late Delivery Probability",
            ascending=False
        ),
        use_container_width=True
    )

else:

    st.info("No high-risk orders found for current filters.")


# ==========================================
# 5. DOWNLOAD REPORTS
# ==========================================

st.divider()

st.subheader("5. Download Reports")

csv_data = filtered_orders.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Order Risk Report",
    data=csv_data,
    file_name="order_risk_report.csv",
    mime="text/csv",
    key="order_report_download"
)


# ==========================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ==========================================

st.divider()

st.subheader("6. Machine Learning Feature Importance")

try:

    importance_df["Importance"] = pd.to_numeric(
        importance_df["Importance"],
        errors="coerce"
    )

    importance_df = importance_df.dropna(
        subset=["Importance"]
    )

    top_features = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(15)

    if not top_features.empty:

        st.write("Top 15 Important Features - Random Forest")

        # Feature Importance Table

        st.dataframe(
            top_features,
            use_container_width=True,
            hide_index=True
        )

        # Feature Importance Graph

        chart_data = top_features.set_index("Feature")[
            "Importance"
        ]

        st.bar_chart(
            chart_data,
            horizontal=True,
            x_label="Importance Score"
        )

    else:

        st.warning("Feature Importance data is empty.")

    st.download_button(
        label="Download Feature Importance Report",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name="feature_importance.csv",
        mime="text/csv",
        key="feature_importance_download"
    )

except Exception as e:

    st.error(f"Feature Importance Error: {e}")


# ==========================================
# 7. INDIVIDUAL ORDER RISK EXPLANATION
# ==========================================

st.divider()

st.subheader("7. Individual Order Risk Explanation")

if len(filtered_orders) > 0:

    selected_record = st.selectbox(
        "Select Prediction Record",
        options=filtered_orders.index.tolist(),
        format_func=lambda x: f"Prediction Record {x + 1}"
    )

    selected_order = filtered_orders.loc[selected_record]

    probability = selected_order["Late Delivery Probability"]

    risk_category = selected_order["Risk Category"]

    # Risk Information

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Late Delivery Probability",
        f"{probability * 100:.2f}%"
    )

    col2.metric(
        "Risk Category",
        risk_category
    )

    col3.metric(
        "Predicted Late Risk",
        int(selected_order["Predicted Late Risk"])
    )

    # Order Details

    st.markdown("### Order Details")

    details = {

        "Shipping Mode": selected_order["Shipping Mode"],

        "Order Region": selected_order["Order Region"],

        "Market": selected_order["Market"],

        "Customer Segment": selected_order["Customer Segment"],

        "Scheduled Shipping Days": selected_order.get(
            "Days for shipment (scheduled)",
            "Not Available"
        ),

        "Order Quantity": selected_order.get(
            "Order Item Quantity",
            "Not Available"
        )
    }

    details_df = pd.DataFrame(
        list(details.items()),
        columns=["Parameter", "Value"]
    )

    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True
    )

    # Risk Interpretation

    st.markdown("### Risk Interpretation")

    if probability >= 0.90:

        st.error(
            "Critical Risk: This order needs immediate operational review."
        )

    elif probability >= 0.70:

        st.warning(
            "High Risk: Review shipment planning and carrier arrangements."
        )

    elif probability >= 0.40:

        st.info(
            "Medium Risk: Monitor shipment progress and dispatch planning."
        )

    else:

        st.success(
            "Low Risk: Current model prediction indicates comparatively lower risk."
        )

    st.caption(
        "The order details describe the selected record. "
        "Feature importance shows global model drivers, not proven "
        "causes or individual SHAP explanations."
    )

else:

    st.warning("No orders available for the selected filters.")


# ==========================================
# DASHBOARD FOOTER
# ==========================================

st.divider()

st.success("Dashboard Loaded Successfully!")

st.caption(
    "Machine Learning-Based Late Delivery Risk Prediction "
    "| Unified Mentor Internship Project"
)

# ==========================================
# 8. MODEL PERFORMANCE COMPARISON
# ==========================================

st.header("8. Model Performance Comparison")

try:
    model_df = pd.read_csv("reports/model_comparison.csv")

    st.subheader("Model Evaluation Metrics")

    st.dataframe(
        model_df.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC-AUC": "{:.2%}"
        }),
        use_container_width=True
    )

    st.subheader("Model Performance Chart")

    chart_df = model_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ]

    st.bar_chart(chart_df)

    best_model = model_df.loc[
        model_df["ROC-AUC"].idxmax()
    ]

    st.success(
        f"Best Performing Model: {best_model['Model']} "
        f"| ROC-AUC: {best_model['ROC-AUC']:.2%}"
    )

except Exception as e:
    st.error(f"Model comparison error: {e}")

    # ==========================================
# STEP 11 - CONFUSION MATRIX VISUALIZATION
# ==========================================

st.header("9. Confusion Matrix Visualization")

import os
from PIL import Image

confusion_files = {
    "Logistic Regression": "graphs/logistic_regression_confusion_matrix.png",
    "Random Forest": "graphs/random_forest_confusion_matrix.png",
    "Gradient Boosting": "graphs/gradient_boosting_confusion_matrix.png"
}

for model_name, file_path in confusion_files.items():

    st.subheader(model_name)

    if os.path.exists(file_path):

        img = Image.open(file_path)
        st.image(img, caption=f"{model_name} Confusion Matrix",
                 use_container_width=True)

    else:
        st.warning(f"Confusion Matrix not found: {file_path}")

st.success("Confusion Matrix Visualization Loaded Successfully!")