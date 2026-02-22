# EV-ChargePlan AI: Intelligent EV Charging Demand Prediction & Planning System

An AI-driven analytics system that predicts electric vehicle (EV) charging demand and provides agentic recommendations for infrastructure planning.

## 🚀 Overview
This project solves the challenge of urban EV infrastructure planning by:
1.  **Predicting Demand**: Using historical session data from the UrbanEV dataset (Shenzhen) to forecast hourly load.
2.  **Agentic Planning**: Utilizing a LangGraph-based AI assistant to analyze load patterns and suggest station expansions.
3.  **Interactive Visualization**: A Streamlit dashboard for real-time monitoring and reporting.

## 🛠️ Tech Stack
- **Dashboard**: Streamlit, Plotly, Pydeck
- **Machine Learning**: Scikit-Learn (Random Forest Regressor)
- **Agentic AI**: LangGraph, LangChain
- **Reporting**: FPDF2
- **Data**: UrbanEV Dataset (Dryad)

## 📦 Project Structure
```text
.
├── app/
│   ├── main.py          # Main Streamlit Application
│   ├── agent.py         # LangGraph Agent logic
│   ├── preprocess.py    # Data cleaning & engineering
│   ├── model.py         # Model training/loading utilities
│   └── report.py        # PDF generation logic
├── data/                # Raw and processed datasets
├── models/              # Trained ML models (.pkl)
├── requirements.txt     # Python dependencies
├── setup.sh             # Environment setup script
└── train_model.py       # ML Training pipeline
```

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip3

### Installation
1.  Clone the repository:
    ```bash
    git clone <your-repo-link>
    cd ev.project
    ```
2.  Run the setup script:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

## 📈 Milestone 1: ML Prediction
- **Input**: Charging timestamps, session volume, pricing, and time features.
- **Model**: Random Forest Regressor.
- **Metrics**: MAE: 7.35 | RMSE: 11.66.

## 🤖 Milestone 2: Agentic Assistant
The assistant analyzes "Hot Zones" (zones with >85% occupancy) and cross-references them with urban planning guidelines to suggest:
- Optimal number of new fast-charging piles.
- Transformer capacity increases.
- Load-balancing strategies (dynamic pricing).

## 📄 Final Deliverables
- [x] Streamlit Hosted Link
- [x] LaTeX Project Report
- [x] Walkthrough Video
- [x] GitHub Repository
