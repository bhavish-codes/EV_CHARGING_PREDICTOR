import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import os
import requests
import json
from huggingface_hub import InferenceClient
from groq import Groq
from dotenv import load_dotenv

# Configuration and environment loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

st.set_page_config(page_title="EV Charging Planner", layout="wide")

# Configuration for data and models (Root level)
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_demand.pkl")
STATION_INFO_PATH = os.path.join(BASE_DIR, "data", "raw", "UrbanEVDataset", "UrbanEVDataset", "20220901-20230228_zone-cleaned-aggregated", "station_information.csv")

@st.cache_resource
def load_rf_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            raise e
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
page = st.sidebar.radio("Go to", ["Dashboard", "Demand Forecasting", "AI Infrastructure Planner", "Ask AI", "About"])

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

elif page == "AI Infrastructure Planner":
    st.subheader("AI-Driven Infrastructure Recommendation")
    
    st.write("Generate intelligent recommendations for infrastructure planning using a HuggingFace LLM.")
    
    # Using environment variables or Streamlit secrets for deployment
    token = os.getenv("HUGGINGFACE_API_KEY") or st.secrets.get("HUGGINGFACE_API_KEY", "")

    if st.button("Generate Planning Report"):
        if not token:
            st.error("Missing HuggingFace API Key. Please set HUGGINGFACE_API_KEY in your environment or Streamlit Secrets.")
        else:
            with st.spinner("Analyzing demand patterns and communicating with LLM..."):
                try:
                    # Construct Data Summary
                    summary = ""
                    if stations is not None:
                        total_stations = len(stations)
                        total_piles = stations['charge_count'].sum()
                        avg_piles = stations['charge_count'].mean()
                        top_stations = stations.sort_values(by='charge_count', ascending=False).head(3)['station_id'].tolist()
                        
                        summary = f"Total Stations: {total_stations}. Total Charging Piles: {total_piles}. Average Piles per Station: {avg_piles:.1f}. High-capacity stations (Top 3 IDs): {top_stations}."
                    
                    # We create a generic prompt for the LLM
                    client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=token)
                    
                    messages = [
                        {"role": "system", "content": "You are an expert AI urban infrastructure planner. Based on the following data analysis of an EV charging network, please provide a structured recommendation report. Please provide your output exactly with these 4 sections: 1. Demand Summary, 2. High-load Locations, 3. Suggestions for New Charging Stations, 4. Load Balancing Recommendations"},
                        {"role": "user", "content": f"Network Analysis: {summary}\n\nGenerate the planning report."}
                    ]
                    
                    response = client.chat_completion(
                        messages,
                        max_tokens=512,
                        temperature=0.3
                    )
                    
                    generated_text = response.choices[0].message.content
                    
                    if generated_text:
                        st.success("Report Generated Successfully!")
                        st.markdown(generated_text)
                    else:
                        st.error("HuggingFace API returned an empty response.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

elif page == "Ask AI":
    st.subheader("💬 Ask AI about the Dataset")
    
    st.write("Query the EV charging dataset using natural language via Groq Cloud.")
    
    # Initialize Groq Client
    # Using environment variables or Streamlit secrets for Groq
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar clear button for chat
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Data Context Preparation
    def get_context():
        if stations is None:
            return "No dataset loaded."
        
        stats = stations['charge_count'].describe()
        top_10 = stations.sort_values(by='charge_count', ascending=False).head(10)
        
        context = f"""
        EV CHARGING DATASET CONTEXT:
        - Total Stations: {len(stations)}
        - Total Charging Piles: {stations['charge_count'].sum()}
        - Avg Piles per Station: {stats['mean']:.2f}
        - Max Piles at a single station: {stats['max']}
        - Top 10 Stations by Capacity: 
        {top_10[['station_id', 'charge_count']].to_string(index=False)}
        
        The dataset includes station_id, longitude, latitude, and charge_count.
        """
        return context

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask something about the charging stations..."):
        if not api_key:
            st.error("Missing Groq API Key. Please set GROQ_API_KEY in your environment or Streamlit Secrets.")
        else:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    client = Groq(api_key=api_key)
                    
                    data_context = get_context()
                    
                    system_prompt = f"""
                    You are an assistant specialized in analyzed EV charging data.
                    You have access to the following dataset summary:
                    {data_context}
                    
                    Answer the user's questions accurately based on this data. If you don't know the answer, say so.
                    """
                    
                    # Prepare messages for API
                    api_messages = [{"role": "system", "content": system_prompt}] + [
                        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                    ]
                    
                    completion = client.chat.completions.create(
                        model="llama-3-70b-8192",
                        messages=api_messages,
                        temperature=0.3,
                        max_tokens=1024,
                        stream=False
                    )
                    
                    full_response = completion.choices[0].message.content
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Groq API Error: {e}")

elif page == "About":

    st.markdown("""
    ### Project Objective
    Design and implement an analytics system that predicts EV charging demand using historical data.
    
    ### Tech Stack
    - **ML Framework**: Scikit-learn (Random Forest)
    - **Dashboard**: Streamlit, Plotly
    - **Dataset**: UrbanEV Dataset (Open Benchmark)
    """)