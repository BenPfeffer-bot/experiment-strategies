# File: src/models/ml_model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trained = False
        
    def train(self, X, y):
        """
        Train a RandomForest regression model with given features.
        """
        self.model.fit(X, y)
        self.trained = True
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        print(f"Model trained. MSE: {mse:.4f}, R2: {r2:.4f}")
        
    def predict(self, X):
        """
        Return predictions if model is trained.
        """
        if not self.trained:
            raise ValueError("Model is not trained yet!")
        return self.model.predict(X)

