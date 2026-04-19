import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def train_demand_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Trains the core Random Forest Regressor targeting EV energy volume demand.
    Limits tree depth to intrinsically prevent overfitting on sparse temporal shards.
    """
    if X.empty or y.empty:
        raise ValueError("Cannot train model on empty feature/target arrays.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We restrict max_depth to 10. While deeper trees lower training error, 
    # EV cycles possess intrinsic noise (weather/traffic spikes) that deep trees 
    # mistakenly memorize. Depth 10 forces generalization.
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,  # Utilize all CPU cores for training speed
        random_state=42
    )
    
    logger.info(f"Initiated model fitting on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    # Validate against holdout set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    metrics = {"MAE": mae, "RMSE": rmse, "Test_Samples": len(y_test)}
    return model, metrics

def save_model(model: RandomForestRegressor, metrics: Dict[str, float], path: str = "models/rf_demand.pkl") -> None:
    """
    Safely serializes the model state to disk alongside its validation metrics,
    allowing inference servers to verify historical performance bounds.
    """
    target_path = Path(path)
    
    # Robustly guarantee output directory exists before dumping binary
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "model": model, 
        "metrics": metrics,
        "version": "1.0",
        "description": "Hourly generic load prediction logic"
    }
    
    with open(target_path, "wb") as f:
        pickle.dump(payload, f)
        
    logger.info(f"Model and telemetry safely encoded to {target_path}")

def load_model(path: str = "models/rf_demand.pkl") -> Optional[Any]:
    """
    Loads pre-trained model. Gracefully returns None instead of throwing 
    OS-level crashes if the file has not been built yet.
    """
    if not os.path.exists(path):
        logger.warning(f"Attempted to load missing model architecture at {path}.")
        return None
        
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            
            # Legacy support: Some early scripts might just dump raw model instances
            # instead of our dictionary wrapper. Handle both.
            if isinstance(data, dict) and "model" in data:
                return data["model"]
            return data
    except Exception as e:
        logger.error(f"Corrupted binary pickle stream at {path}. Exception: {e}")
        return None
