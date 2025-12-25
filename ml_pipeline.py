# ml_pipeline.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb

class ForexMLPredictor:
    """Machine learning for Forex and gold prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def create_features(self, data: pd.DataFrame, 
                       lookback: int = 20) -> pd.DataFrame:
        """Create comprehensive feature set for Forex/Gold"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = \
                data['Close'].pct_change().rolling(window).std()
            features[f'atr_{window}'] = self.calculate_atr(data, window)
        
        # Trend features
        features['sma_ratio'] = data['Close'] / data['Close'].rolling(20).mean()
        features['ema_ratio'] = data['Close'] / data['Close'].ewm(span=20).mean()
        
        # Momentum indicators
        features['rsi'] = self.calculate_rsi(data['Close'], 14)
        features['macd'] = self.calculate_macd(data['Close'])
        features['stochastic'] = self.calculate_stochastic(data, 14)
        
        # Support/Resistance
        features['pivot_points'] = self.calculate_pivot_points(data)
        
        # Time-based features (important for Forex)
        features['hour_of_day'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month
        
        # Gold-specific features
        if 'XAU' in data.columns or 'GOLD' in data.columns:
            features['gold_usd_correlation'] = \
                self.calculate_rolling_correlation(
                    data['Close'], 
                    self.get_usd_index(), 
                    window=20
                )
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            
        return features.dropna()
    
    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series,
                            train_size: float = 0.8) -> Dict:
        """Train ensemble of ML models"""
        # Split data
        split_idx = int(len(X) * train_size)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        self.scalers['X'] = StandardScaler()
        X_train_scaled = self.scalers['X'].fit_transform(X_train)
        X_test_scaled = self.scalers['X'].transform(X_test)
        
        self.scalers['y'] = StandardScaler()
        y_train_scaled = self.scalers['y'].fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        # Initialize models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            ),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train_scaled)
            self.models[name] = model
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        # Ensemble prediction (weighted average)
        predictions = self.ensemble_predict(X_test_scaled)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, predictions)
        
        return {
            'models': self.models,
            'feature_importance': self.feature_importance,
            'metrics': metrics,
            'predictions': predictions
        }
    
    def ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction"""
        predictions = []
        weights = {'random_forest': 0.3, 'xgboost': 0.3, 
                  'lightgbm': 0.3, 'svr': 0.1}
        
        for name, model in self.models.items():
            pred = model.predict(X)
            # Inverse transform if scaler exists
            if 'y' in self.scalers:
                pred = self.scalers['y'].inverse_transform(
                    pred.reshape(-1, 1)
                ).ravel()
            predictions.append(pred * weights.get(name, 0.25))
        
        return np.sum(predictions, axis=0)