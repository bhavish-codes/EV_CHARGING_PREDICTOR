import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import os

st.set_page_config(page_title="EV Charging Planner", layout="wide")

# Configuration for data and models
MODEL_PATH = "models/rf_demand.pkl"
STATION_INFO_PATH = "data/raw/UrbanEVDataset/UrbanEVDataset/20220901-20230228_zone-cleaned-aggregated/station_information.csv"

@st.cache_resource
def load_rf_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_station_info():
    if os.path.exists(STATION_INFO_PATH):
        return pd.read_csv(STATION_INFO_PATH)
    return None

model = load_rf_model()
stations = load_station_info()

st.title("Intelligent EV Charging Demand Prediction")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Demand Forecasting", "About"])

if page == "Dashboard":
    st.subheader("Station Network Overview")
    if stations is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.map(stations, latitude="latitude", longitude="longitude")
        with col2:
            st.metric("Total Stations", len(stations))
            avg_piles = stations['charge_count'].mean()
            st.metric("Avg Piles / Station", f"{avg_piles:.1f}")
            st.write("Top Stations by Capacity:")
            st.dataframe(stations.sort_values(by='charge_count', ascending=False).head(10))

elif page == "Demand Forecasting":
    st.subheader("Forecast Charging Demand")
    
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Select Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Select Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    with col2:
        s_price = st.number_input("Service Fee (CNY/kWh)", value=0.5)
        e_price = st.number_input("Electricity Price (CNY/kWh)", value=1.0)

    # Process inputs for model prediction
    day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
    is_weekend = 1 if day_idx >= 5 else 0
    
    input_df = pd.DataFrame([[hour, day_idx, is_weekend, s_price, e_price]], 
                           columns=['hour', 'day_of_week', 'is_weekend', 's_price', 'e_price'])
    
    if model:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Charging Volume: **{prediction:.2f} kWh**")
        
        # Display hourly trends
        st.write("---")
        st.write("Hourly Demand Trend (24h)")
        hours = list(range(24))
        trend_X = pd.DataFrame({
            'hour': hours,
            'day_of_week': [day_idx]*24,
            'is_weekend': [is_weekend]*24,
            's_price': [s_price]*24,
            'e_price': [e_price]*24
        })
        trend_preds = model.predict(trend_X)
        fig = px.line(x=hours, y=trend_preds, labels={'x': 'Hour of Day', 'y': 'Predicted Demand (kWh)'}, 
                     title=f"Predicted Demand Cycle for {day_of_week}")
        st.plotly_chart(fig, use_container_width=True)

elif page == "About":

    st.markdown("""
    ### Project Objective
    Design and implement an analytics system that predicts EV charging demand using historical data.
    
    ### Tech Stack
    - **ML Framework**: Scikit-learn (Random Forest)
    - **Dashboard**: Streamlit, Plotly
    - **Dataset**: UrbanEV Dataset (Open Benchmark)
    """)
