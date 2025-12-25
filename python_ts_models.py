# python_ts_models.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats

class AdvancedTimeSeriesFX:
    """Forex-specific time series analysis"""
    
    def __init__(self):
        self.fx_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        
    def calculate_rollinger_features(self, data: pd.DataFrame, 
                                    window: int = 20) -> pd.DataFrame:
        """Bollinger Bands with FX-adapted parameters"""
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        
        features = pd.DataFrame({
            'bb_upper': sma + (std * 2),
            'bb_lower': sma - (std * 2),
            'bb_width': (sma + (std * 2) - (sma - (std * 2))) / sma,
            'bb_position': (data['Close'] - sma) / (2 * std),
            'volatility': std / sma * 100  # Percentage volatility
        })
        return features
    
    def hurst_exponent(self, prices: pd.Series, 
                      max_lag: int = 20) -> float:
        """Calculate Hurst exponent for market regime detection"""
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) 
               for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    def forex_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """FX-specific features"""
        features = pd.DataFrame(index=data.index)
        
        # Carry trade signals (interest rate differentials)
        # Assuming we have interest rate data
        features['carry_signal'] = self.calculate_carry_signal(data)
        
        # Momentum across multiple timeframes (FX trends)
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = \
                data['Close'].pct_change(period)
            
        # Volatility ratio (important for Forex)
        features['vol_ratio'] = \
            data['Close'].rolling(5).std() / data['Close'].rolling(20).std()
            
        # Overnight gaps (important for Forex)
        features['overnight_gap'] = \
            (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            
        return features