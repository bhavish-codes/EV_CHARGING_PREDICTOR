import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pickle
import os

def train_demand_model(X, y):
    """Trains an XGBoost regressor on the provided data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return model, {"MAE": mae, "RMSE": rmse}

def save_model(model, metrics, path="models/xgboost_demand.pkl"):
    """Saves the trained model and its metrics."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics}, f)

def load_model(path="models/xgboost_demand.pkl"):
    """Loads a previously trained model."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
