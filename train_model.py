import os
import sys
import logging
import pandas as pd

# Directly import the unified structural pipeline rather than reinventing it here
from app.preprocess import process_station_pipeline
from app.model import train_demand_model, save_model

# Establish production-grade console logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration Constants
DATA_DIR = "data/raw/UrbanEVDataset/UrbanEVDataset/20220901-20230228_station-raw/charge_5min"
STATIONS_TO_TRAIN = ["1001.csv", "1002.csv", "1003.csv", "1006.csv", "1008.csv"]
MODEL_PATH = "models/rf_demand.pkl"

def main():
    logger.info("Initializing EV Demand Model Training Pipeline...")
    all_data = []
    
    # 1. Ingest & Engineer Data Space
    for station_filename in STATIONS_TO_TRAIN:
        path = os.path.join(DATA_DIR, station_filename)
        if not os.path.exists(path):
            logger.warning(f"Target dataset {station_filename} skipped (File not found).")
            continue
            
        try:
            # Leverage our newly refactored, robust engineering pipeline
            processed_df = process_station_pipeline(path)
            all_data.append(processed_df)
            logger.info(f"Successfully digested {len(processed_df)} hourly shards from {station_filename}")
        except Exception as e:
            logger.error(f"Failed to process {station_filename} due to data corruption/schema mismatch. Err: {e}")

    if not all_data:
        logger.critical("Pipeline aborted. Zero datasets successfully loaded.")
        sys.exit(1)

    # Compile the ultimate master table
    full_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Unified dataset compiled. Absolute row count: {len(full_df)}")
    
    # 2. Extract Vectors
    # Note: We now utilize the cyclical time features to respect physical clock boundaries
    target = 'volume'
    features = ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 's_price', 'e_price']
    
    # Validate feature integrity before model fit
    missing_cols = [f for f in features if f not in full_df.columns]
    if missing_cols:
        logger.critical(f"Dataframe missing engineered features necessary for prediction: {missing_cols}")
        sys.exit(1)
        
    X = full_df[features]
    y = full_df[target]
    
    # 3. Fit & Evaluate
    logger.info("Engaging Random Forest Regressor architecture...")
    model, metrics = train_demand_model(X, y)
    
    logger.info("--- Model Efficacy Report ---")
    logger.info(f"Mean Absolute Error : {metrics['MAE']:.3f} kWh")
    logger.info(f"Root Mean Sq. Error : {metrics['RMSE']:.3f} kWh")
    logger.info(f"Validation Sample Sz: {metrics['Test_Samples']}")
    
    # 4. Export Artifact
    save_model(model, metrics, MODEL_PATH)

if __name__ == "__main__":
    main()
