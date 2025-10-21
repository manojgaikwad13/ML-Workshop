import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np


st.set_page_config(page_title="Swiggy Delivery Time Prediction", page_icon="=´", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-color: #ffe5e5;
        border-radius: 15px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("=´B Swiggy Delivery Time Prediction")
st.write("Predict estimated delivery time for Swiggy orders based on key features.")


df = pd.read_csv(r'swiggy_cleaned (1).csv')
df.dropna(inplace=True)


input_columns = [
    'age', 'ratings', 'weather', 'traffic', 'vehicle_condition',
    'type_of_order', 'type_of_vehicle', 'multiple_deliveries', 'festival',
    'city_type', 'city_name', 'order_day_of_week', 'is_weekend',
    'pickup_time_minutes', 'order_time_hour', 'order_time_of_day', 'distance'
]


st.subheader("Enter Delivery Details:")

Age = st.number_input(' Age of Delivery Partner:', min_value=18, max_value=60, value=25)
Rating = st.number_input('P Average Rating:', min_value=1, max_value=5, value=4)
weather = st.selectbox(' Weather Condition:', df['weather'].unique())
traffic = st.selectbox(' Traffic Condition:', df['traffic'].unique())
vehicle_condition = st.selectbox(' Vehicle Condition:', df['vehicle_condition'].unique())
type_of_order = st.selectbox(' Type of Order:', df['type_of_order'].unique())
type_of_vehicle = st.selectbox('Type of Vehicle:', df['type_of_vehicle'].unique())
multiple_deliveries = st.number_input('Number of Deliveries:', min_value=1, max_value=3, value=1)
festival = st.selectbox(' Festival:', df['festival'].unique())
city_type = st.selectbox('City Type:', df['city_type'].unique())
city_name = st.selectbox('City Name:', df['city_name'].unique())
order_day_of_week = st.selectbox(' Order Day of Week:', df['order_day_of_week'].unique())
is_weekend = st.number_input(' Is it Weekend? (1=Yes, 0=No):', min_value=0, max_value=1, value=0)
pickup_time_minutes = st.number_input(' Pickup Time (minutes):', min_value=1, max_value=100, value=15)
order_time_hour = st.number_input('Order Time (hour):', min_value=0, max_value=23, value=12)
order_time_of_day = st.selectbox(' Order Time of Day:', df['order_time_of_day'].unique())
distance = st.number_input(' Distance (km):', min_value=1, max_value=100, value=5)


MODEL_PATH = r'model.pkl'

if not os.path.exists(MODEL_PATH):
    st.error("L Model file not found! Please make sure 'model.pkl' exists.")
    st.stop()
else:
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"  Error loading model: {e}")
        st.stop()

# ----------------------------
# >þ Prepare DataFrame
# ----------------------------
data = pd.DataFrame([[
    Age, Rating, weather, traffic, vehicle_condition, type_of_order,
    type_of_vehicle, multiple_deliveries, festival, city_type, city_name,
    order_day_of_week, is_weekend, pickup_time_minutes,
    order_time_hour, order_time_of_day, distance
]], columns=input_columns)

# ----------------------------
# =' Auto-Fix Missing Columns
# ----------------------------
expected_cols = getattr(model, 'feature_names_in_', None)
if expected_cols is not None:
    # Add missing columns
    missing_cols = set(expected_cols) - set(data.columns)
    for col_name in missing_cols:
        if col_name == 'order_day' and 'order_day_of_week' in data.columns:
            data[col_name] = data['order_day_of_week']
        else:
            data[col_name] = 0

    # Reorder columns to match model
    data = data[list(expected_cols)]

# ----------------------------
# =' Handle Unknown Categories
# ----------------------------
# Check for categorical transformers
if hasattr(model.named_steps['preprocessor'], 'named_transformers_'):
    cat_transformer = model.named_steps['preprocessor'].named_transformers_.get('cat', None)
    if cat_transformer and hasattr(cat_transformer, 'categories_'):
        cat_cols = model.named_steps['preprocessor'].transformers_[1][2]  # categorical column names
        for i, col_name in enumerate(cat_cols):
            known_categories = cat_transformer.categories_[i]
            if data[col_name].iloc[0] not in known_categories:
                # Replace unknown category with first known category to avoid crash
                data[col_name] = known_categories[0]

# ----------------------------
# =. Predict Button
# ----------------------------
if st.button("=. Predict Delivery Time"):
    try:
        result = model.predict(data)[0]
        st.success(f"=P Estimated Delivery Time: **{round(result, 2)} minutes**")
    except Exception as e:
        st.error(f"  Prediction failed: {e}")


