import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Configuration
DATA_DIR = "data/raw/UrbanEVDataset/UrbanEVDataset/20220901-20230228_station-raw/charge_5min"
STATIONS_TO_TRAIN = ["1001.csv", "1002.csv", "1003.csv", "1006.csv", "1008.csv"]
MODEL_PATH = "models/rf_demand.pkl"

def aggregate_and_engineer(df):
    """Clean and prepare features."""
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Aggregate to hourly
    hourly = df.resample('H').agg({
        'busy': 'mean',
        'volume': 'sum',
        's_price': 'mean',
        'e_price': 'mean'
    }).reset_index()
    
    # Time features
    hourly['hour'] = hourly['time'].dt.hour
    hourly['day_of_week'] = hourly['time'].dt.dayofweek
    hourly['is_weekend'] = hourly['day_of_week'].isin([5, 6]).astype(int)
    
    # Target is 'volume' (demand)
    return hourly.dropna()

def train():
    all_data = []
    print(f"Loading data for stations: {STATIONS_TO_TRAIN}")
    for station_file in STATIONS_TO_TRAIN:
        path = os.path.join(DATA_DIR, station_file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            processed_df = aggregate_and_engineer(df)
            all_data.append(processed_df)
    
    if not all_data:
        print("No data found to train on.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Features and Target
    X = full_df[['hour', 'day_of_week', 'is_weekend', 's_price', 'e_price']]
    y = full_df['volume']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"Model trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
