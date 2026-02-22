# EV Charging Planner: Demand Prediction & Infrastructure System

An analytics system that predicts electric vehicle (EV) charging demand and provides automated recommendations for infrastructure planning.

## Overview
This project addresses urban EV infrastructure challenges by:
1.  **Predicting Demand**: Forecasting hourly load using historical session data.
2.  **Infrastructure Planning**: Analyzing load patterns to suggest station expansions and efficiency improvements.
3.  **Visualization**: An interactive dashboard for monitoring and generating planning reports.

## Tech Stack
- **Dashboard**: Streamlit, Plotly
- **Machine Learning**: Scikit-Learn
- **System Logic**: LangGraph
- **Reporting**: FPDF2
- **Data**: UrbanEV Dataset (Open Benchmark)

## Project Structure
```text
.
├── app/
│   ├── main.py          # Dashboard Application
│   ├── agent.py         # Planning Logic
│   ├── preprocess.py    # Data Engineering
│   ├── model.py         # Model Utilities
│   └── report.py        # PDF Generation
├── data/                # Station Metadata
├── models/              # Trained Model
├── requirements.txt     # Dependencies
├── setup.sh             # Setup Script
└── train_model.py       # Training Pipeline
```

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip3

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/bhavish-codes/EV_CHARGING_PREDICTOR.git
    cd ev.project
    ```
2.  **Run the setup script**:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

## Accessing the Dashboard
Once the setup is complete, launch the dashboard using the following command:
```bash
streamlit run app/main.py
```
By default, the application will be accessible at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip-address:8501

## Methodology
- **Model**: Random Forest Regressor trained on seasonal charging patterns.
- **Planner**: Automated decision-making based on occupancy thresholds and urban planning guidelines.

