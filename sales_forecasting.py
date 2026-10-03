
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import date


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sales Forecasting & Profit Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "sales_forecasting_dataset.csv"
MODEL_FILE = BASE_DIR / "Sales_Model2.pkl"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Only Year is required by the current model
    df["year"] = df["Date"].dt.year

    return df


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


# =========================================================
# LOAD FILES
# =========================================================

try:

    df = load_data()
    pipe = load_model()

except FileNotFoundError as e:

    st.error(f"Required file not found: {e.filename}")

    st.info(
        "Keep sales_forecasting_dataset.csv and "
        "Sales_Model2.pkl in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Could not load the application files: {e}"
    )

    st.stop()


# =========================================================
# PRODUCT → CATEGORY
# =========================================================

product_category = (
    df[["Product", "Category"]]
    .drop_duplicates()
    .set_index("Product")["Category"]
    .to_dict()
)


products = sorted(
    df["Product"]
    .dropna()
    .unique()
    .tolist()
)


regions = sorted(
    df["Region"]
    .dropna()
    .unique()
    .tolist()
)


# =========================================================
# HEADER
# =========================================================

st.title("📊 Sales Forecasting & Profit Prediction")

st.caption(
    "Predict Total Sales using the trained Machine Learning "
    "model and estimate profit using the entered Cost Price."
)

st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

left, right = st.columns(2)


# ---------------------------------------------------------
# PRODUCT INFORMATION
# ---------------------------------------------------------

with left:

    st.subheader("📦 Product Information")

    forecast_date = st.date_input(
        "Forecast Date",
        value=date.today(),
        min_value=date(2016, 1, 1),
        max_value=date(2030, 12, 31)
    )

    product = st.selectbox(
        "Product",
        products
    )

    # Automatically select category
    category = product_category[product]

    st.text_input(
        "Category",
        value=category,
        disabled=True
    )

    region = st.selectbox(
        "Region",
        regions
    )


# ---------------------------------------------------------
# SALES & COST INFORMATION
# ---------------------------------------------------------

with right:

    st.subheader("💰 Sales & Cost Information")

    quantity_sold = st.number_input(
        "Quantity Sold",
        min_value=1,
        max_value=100000,
        value=10,
        step=1
    )

    selling_price = st.number_input(
        "Selling Price / Unit (₹)",
        min_value=0.0,
        value=500.0,
        step=10.0,
        format="%.2f"
    )

    discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        format="%.1f"
    )

    cost_price = st.number_input(
        "Cost Price / Unit (₹)",
        min_value=0.0,
        value=300.0,
        step=10.0,
        format="%.2f"
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Sales & Calculate Profit",
    type="primary",
    use_container_width=False
):

    # -----------------------------------------------------
    # DATE → YEAR
    # -----------------------------------------------------

    selected_date = pd.Timestamp(
        forecast_date
    )

    year = selected_date.year


    # -----------------------------------------------------
    # UNIT PRICE LOG TRANSFORMATION
    # -----------------------------------------------------

    unit_price_log = np.log1p(
        selling_price
    )


    # -----------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------
    #
    # Current model features:
    #
    # year
    # Product
    # Category
    # Region
    # Quantity_Sold
    # Unit_Price_log
    # Discount
    #
    # Month is intentionally NOT included.
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "year": [year],

        "Product": [product],

        "Category": [category],

        "Region": [region],

        "Quantity_Sold": [quantity_sold],

        "Unit_Price_log": [unit_price_log],

        "Discount": [discount]

    })


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    try:

        prediction_log = float(
            pipe.predict(input_data)[0]
        )

        # Convert log prediction back
        # to original Total_Sales scale

        predicted_sales = np.expm1(
            prediction_log
        )

    except Exception as e:

        st.error(
            "Prediction failed. Make sure Sales_Model2.pkl "
            "was trained with these features:"
        )

        st.code(
            """
year
Product
Category
Region
Quantity_Sold
Unit_Price_log
Discount
"""
        )

        st.exception(e)

        st.stop()


    # -----------------------------------------------------
    # PREVENT NEGATIVE BUSINESS VALUE
    # -----------------------------------------------------

    predicted_sales = max(
        0.0,
        predicted_sales
    )


    # =====================================================
    # PROFIT CALCULATION
    # =====================================================

    total_cost = (
        quantity_sold *
        cost_price
    )

    net_profit = (
        predicted_sales -
        total_cost
    )


    if predicted_sales > 0:

        profit_margin = (
            net_profit /
            predicted_sales
        ) * 100

    else:

        profit_margin = 0.0


    # =====================================================
    # SALES CALCULATION CHECK
    # =====================================================

    discounted_unit_price = (
        selling_price *
        (1 - discount / 100)
    )


    expected_sales_from_input = (
        discounted_unit_price *
        quantity_sold
    )


    # =====================================================
    # RESULTS
    # =====================================================

    st.subheader("📊 Prediction Results")

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Predicted Total Sales",
            f"₹{predicted_sales:,.2f}"
        )


    with c2:

        st.metric(
            "Total Cost",
            f"₹{total_cost:,.2f}"
        )


    with c3:

        st.metric(
            "Net Profit"
            if net_profit >= 0
            else "Net Loss",

            f"₹{net_profit:,.2f}"
        )


    st.divider()


    # =====================================================
    # SALES CALCULATION
    # =====================================================

    st.subheader("🧮 Sales Calculation")

    a, b, c = st.columns(3)


    with a:

        st.write(
            "**Discounted Price / Unit**"
        )

        st.write(
            f"₹{discounted_unit_price:,.2f}"
        )


    with b:

        st.write(
            "**Sales from Input Values**"
        )

        st.write(
            f"₹{expected_sales_from_input:,.2f}"
        )


    with c:

        st.write(
            "**ML Predicted Sales**"
        )

        st.write(
            f"₹{predicted_sales:,.2f}"
        )


    st.caption(
        "Sales from Input Values = Quantity × Selling Price × "
        "(1 − Discount/100). This is shown as a calculation "
        "check; the ML result comes from your trained model."
    )


    # =====================================================
    # PROFIT ANALYSIS
    # =====================================================

    st.subheader("💰 Profit Analysis")

    p1, p2, p3 = st.columns(3)


    with p1:

        st.write("**Total Cost**")

        st.write(
            f"₹{quantity_sold:,.0f} × "
            f"₹{cost_price:,.2f} = "
            f"₹{total_cost:,.2f}"
        )


    with p2:

        st.write(
            "**Net Profit / Loss**"
        )

        st.write(
            f"₹{predicted_sales:,.2f} − "
            f"₹{total_cost:,.2f} = "
            f"₹{net_profit:,.2f}"
        )


    with p3:

        st.write(
            "**Profit Margin**"
        )

        st.write(
            f"{profit_margin:.2f}%"
        )


    if net_profit >= 0:

        st.success(
            f"Estimated Net Profit: "
            f"₹{net_profit:,.2f}"
        )

    else:

        st.warning(
            f"Estimated Net Loss: "
            f"₹{abs(net_profit):,.2f}"
        )


    # =====================================================
    # PREDICTION DETAILS
    # =====================================================

    with st.expander(
        "🔎 Prediction Details"
    ):

        st.write(
            "**Model Input:**"
        )

        st.dataframe(
            input_data,
            use_container_width=True
        )

