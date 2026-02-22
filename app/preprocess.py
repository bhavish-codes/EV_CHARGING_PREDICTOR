import pandas as pd
import numpy as np

def load_charging_data(filepath):
    """Loads a single station's 5-min charging data."""
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    return df

def aggregate_to_hourly(df):
    """Aggregates 5-min data to hourly intervals."""
    df.set_index('time', inplace=True)
    hourly_df = df.resample('H').agg({
        'busy': 'mean',
        'idle': 'mean',
        's_price': 'mean',
        'e_price': 'mean',
        'duration': 'sum',
        'volume': 'sum'
    }).reset_index()
    return hourly_df

def engineer_features(df):
    """Extracts time-based features from the timestamp."""
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

if __name__ == "__main__":
    pass

