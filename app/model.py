from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pickle
import os

def train_demand_model(X, y):
    """Fits a Random Forest regressor for charging demand forecasting."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return model, {"MAE": mae, "RMSE": rmse}

def save_model(model, metrics, path="models/rf_demand.pkl"):
    """Serializes the model and associated validation metrics."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics}, f)

def load_model(path="models/rf_demand.pkl"):
    """Loads a pre-trained model from the local directory."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

