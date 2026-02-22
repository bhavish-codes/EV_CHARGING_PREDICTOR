import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import os

st.set_page_config(page_title="EV-ChargePlan AI", layout="wide", page_icon="⚡")

# Load Resources
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

st.title("⚡ Intelligent EV Charging Demand Prediction")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Demand Forecasting", "Infrastructure Planning", "About"])

if page == "Dashboard":
    st.subheader("📍 Station Network Overview")
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
    st.subheader("📊 Forecast Charging Demand")
    
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Select Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Select Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    with col2:
        s_price = st.number_input("Service Fee (CNY/kWh)", value=0.5)
        e_price = st.number_input("Electricity Price (CNY/kWh)", value=1.0)

    # Prepare input
    day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
    is_weekend = 1 if day_idx >= 5 else 0
    
    input_df = pd.DataFrame([[hour, day_idx, is_weekend, s_price, e_price]], 
                           columns=['hour', 'day_of_week', 'is_weekend', 's_price', 'e_price'])
    
    if model:
        prediction = model.predict(input_df)[0]
        st.success(f"📈 Predicted Charging Volume: **{prediction:.2f} kWh**")
        
        # Trend Visualization
        st.write("---")
        st.write("🕒 Hourly Demand Trend (24h)")
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

elif page == "Infrastructure Planning":
    st.subheader("🤖 Agentic AI Infrastructure Planning Assistant")
    st.markdown("This assistant reasons about charging demand patterns and generates structured growth recommendations.")
    
    if st.button("🚀 Run Planning Analysis"):
        with st.spinner("Agentic Assistant is reasoning..."):
            try:
                # 1. Gather Demand Insights (Simplified for UI flow)
                # In a real scenario, we'd pass actual model predictions for different zones
                demand_summary = "Peak demand of 25.4 kWh observed between 5 PM and 8 PM on weekdays. Current station capacity (12 piles) is entering 85% occupancy during these windows."
                
                # 2. Run LangGraph Workflow (Mocked for now to show flow)
                from app.agent import build_agent_graph
                agent_app = build_agent_graph()
                initial_state = {
                    "demand_summary": demand_summary,
                    "hot_zones": ["Station 1001 Area"],
                    "guidelines": "",
                    "recommendations": "",
                    "history": []
                }
                
                result = agent_app.invoke(initial_state)
                
                # 3. Display Results
                st.success("Analysis Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.info("🔍 Demand Insights")
                    st.write(result['demand_summary'])
                    st.write("**Identified Hot Zones:**")
                    for zone in result['hot_zones']:
                        st.write(f"- {zone}")
                
                with col2:
                    st.info("💡 Recommendations")
                    st.write(result['recommendations'])
                
                # 4. Generate and Export PDF
                from app.report import generate_pdf_report
                report_path = generate_pdf_report(result['demand_summary'], result['recommendations'])
                
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📄 Download Planning Report (PDF)",
                        data=f,
                        file_name="EV_Planning_Report.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Error during agentic analysis: {e}")

elif page == "About":

    st.markdown("""
    ### Project Objective
    Design and implement an AI-driven analytics system that predicts EV charging demand using historical data and generates structured planning recommendations.
    
    ### Tech Stack
    - **ML**: Scikit-learn (Random Forest)
    - **UI**: Streamlit, Plotly
    - **Data**: UrbanEV Dataset (Dryad)
    """)

