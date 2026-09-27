import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from datetime import date

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Profit Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "sales_forecasting_dataset.csv"
MODEL_FILE = BASE_DIR / "Sales_Model.pkl"

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Same feature engineering used in the notebook
    df["year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    return df


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


try:
    df = load_data()
    pipe = load_model()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}")
    st.info(
        "Keep sales_forecasting_dataset.csv and Sales_Model.pkl "
        "in the same folder as sales_forecasting.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the application files: {e}")
    st.stop()

# ---------------------------------------------------------
# PRODUCT -> CATEGORY
# ---------------------------------------------------------
product_category = (
    df[["Product", "Category"]]
    .drop_duplicates()
    .set_index("Product")["Category"]
    .to_dict()
)

products = sorted(df["Product"].dropna().unique().tolist())
regions = sorted(df["Region"].dropna().unique().tolist())

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("📊 Sales Forecasting & Profit Prediction")
st.caption(
    "Predict Total Sales using the trained ML model and estimate profit "
    "using the entered Cost Price."
)

st.divider()

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("📦 Product Information")

    # Date is used to automatically create year + month.
    # Month is NOT shown as a separate input.
    forecast_date = st.date_input(
        "Forecast Date",
        value=date.today(),
        min_value=date(2024, 1, 1),
        max_value=date(2030, 12, 31)
    )

    product = st.selectbox(
        "Product",
        products
    )

    # Category automatically follows the selected product.
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

    # IMPORTANT:
    # The notebook dataset stores discount as 5.0 for 5%, NOT 0.05.
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

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button(
    "🔮 Predict Sales & Calculate Profit",
    type="primary",
    use_container_width=False
):

    selected_date = pd.Timestamp(forecast_date)

    year = selected_date.year
    month = selected_date.month

    # This exactly matches the notebook feature names/order.
    input_data = pd.DataFrame({
        "year": [year],
        "Month": [month],
        "Product": [product],
        "Category": [category],
        "Region": [region],
        "Quantity_Sold": [quantity_sold],
        "Unit_Price": [selling_price],
        "Discount": [discount]
    })

    try:
        predicted_sales = float(pipe.predict(input_data)[0])
    except Exception as e:
        st.error(
            "Prediction failed. The saved Sales_Model.pkl must be trained "
            "with the same features as the notebook:\n"
            "year, Month, Product, Category, Region, Quantity_Sold, "
            "Unit_Price, Discount"
        )
        st.exception(e)
        st.stop()

    # Prevent a regression model from displaying an impossible negative
    # sales value in the business UI.
    predicted_sales = max(0.0, predicted_sales)

    # -----------------------------------------------------
    # PROFIT CALCULATION
    # -----------------------------------------------------
    total_cost = quantity_sold * cost_price
    net_profit = predicted_sales - total_cost

    if predicted_sales > 0:
        profit_margin = (net_profit / predicted_sales) * 100
    else:
        profit_margin = 0.0

    # Actual customer selling value after discount.
    discounted_unit_price = selling_price * (1 - discount / 100)
    expected_sales_from_input = discounted_unit_price * quantity_sold

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------
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
            "Net Profit" if net_profit >= 0 else "Net Loss",
            f"₹{net_profit:,.2f}"
        )

    st.divider()

    # -----------------------------------------------------
    # SALES CALCULATION CHECK
    # -----------------------------------------------------
    st.subheader("🧮 Sales Calculation")

    a, b, c = st.columns(3)

    with a:
        st.write("**Discounted Price / Unit**")
        st.write(f"₹{discounted_unit_price:,.2f}")

    with b:
        st.write("**Sales from Input Values**")
        st.write(f"₹{expected_sales_from_input:,.2f}")

    with c:
        st.write("**ML Predicted Sales**")
        st.write(f"₹{predicted_sales:,.2f}")

    st.caption(
        "Sales from Input Values = Quantity × Selling Price × "
        "(1 − Discount/100). This is shown as a calculation check; "
        "the ML result comes from your trained model."
    )

    # -----------------------------------------------------
    # PROFIT DETAILS
    # -----------------------------------------------------
    st.subheader("💰 Profit Analysis")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.write("**Total Cost**")
        st.write(
            f"₹{quantity_sold:,.0f} × ₹{cost_price:,.2f}"
            f" = ₹{total_cost:,.2f}"
        )

    with p2:
        st.write("**Net Profit / Loss**")
        st.write(
            f"₹{predicted_sales:,.2f} − ₹{total_cost:,.2f}"
            f" = ₹{net_profit:,.2f}"
        )

    with p3:
        st.write("**Profit Margin**")
        st.write(f"{profit_margin:.2f}%")

    if net_profit >= 0:
        st.success(f"Estimated Net Profit: ₹{net_profit:,.2f}")
    else:
        st.warning(f"Estimated Net Loss: ₹{abs(net_profit):,.2f}")

    # -----------------------------------------------------
    # INPUT SUMMARY
    # -----------------------------------------------------
    with st.expander("🔎 Prediction Details"):
        st.dataframe(input_data, use_container_width=True)
