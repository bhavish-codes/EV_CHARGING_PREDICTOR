import logging
import pandas as pd
import numpy as np
import warnings

# Configure local module logger
logger = logging.getLogger(__name__)

def load_charging_data(filepath: str) -> pd.DataFrame:
    """
    Safely loads and validates raw 5-minute station charging telemetry.
    Drops inherently duplicate telemetry scans to preserve aggregate sums.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"Data file missing at path: {filepath}")
        raise

    if 'time' not in df.columns:
        raise ValueError(f"CRITICAL: Raw data at {filepath} lacks required 'time' column.")

    # Drop explicit duplicates caused by network multi-transmits
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_len:
        logger.debug(f"Removed {initial_len - len(df)} duplicate telemetry rows from {filepath}")

    # Standardize time dimension
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Drop rows where timestamp parsing completely failed
    df.dropna(subset=['time'], inplace=True)
    return df

def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downsamples the volatile 5-minute telemetry into stable 1-hour chunks.
    Averages categorical hardware states (busy/idle) but SUMs consumption (volume).
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to aggregate_to_hourly. Returning empty subset.")
        return df

    # Set index for resampler
    df.set_index('time', inplace=True)
    
    # Not all telemetry stations report 'idle' or 'duration'. Use safe-get aggregation.
    agg_funcs = {}
    if 'busy' in df.columns: agg_funcs['busy'] = 'mean'
    if 'idle' in df.columns: agg_funcs['idle'] = 'mean'
    if 's_price' in df.columns: agg_funcs['s_price'] = 'mean'
    if 'e_price' in df.columns: agg_funcs['e_price'] = 'mean'
    if 'duration' in df.columns: agg_funcs['duration'] = 'sum'
    if 'volume' in df.columns: agg_funcs['volume'] = 'sum'
    
    # Execute the aggregation and restore index to a normal column
    hourly_df = df.resample('h').agg(agg_funcs).reset_index()
    
    # Forward-fill minor gaps (1-2 hours) where a station dropped offline,
    # but strictly drop spans larger than that to prevent hallucinated data.
    hourly_df.ffill(limit=2, inplace=True)
    hourly_df.dropna(inplace=True)
    
    return hourly_df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Injects critical temporal dimensions into the dataset.
    Re-maps linear time variables into cyclical equivalents so the Regression model
    understands that 23:00 (11PM) is temporally adjacent to 00:00 (Midnight).
    """
    if df.empty or 'time' not in df.columns:
        return df

    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Create isolated weekend flag (usage spikes drastically on Sat/Sun)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # --- CYCLICAL ENCODING ---
    # Traditional ML trees split recursively on thresholds (e.g. hour < 12). 
    # If we leave 'hour' linear, the model thinks hour 0 and 23 are extremely far apart.
    # Mapping to sine/cosine coordinates essentially maps the hours onto a continuous clock circle.
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

def process_station_pipeline(filepath: str) -> pd.DataFrame:
    """End-to-end wrapper executing the unified preprocessing pipeline for a single station."""
    df = load_charging_data(filepath)
    hourly_df = aggregate_to_hourly(df)
    final_df = engineer_features(hourly_df)
    return final_df
