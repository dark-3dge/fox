# regime_detection.py
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

class ForexRegimeDetector:
    """HMM-based regime detection for Forex markets"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.regime_labels = None
        
    def extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection"""
        features = []
        
        # Volatility features
        returns = data['Close'].pct_change().dropna()
        features.append(returns.rolling(5).std().values)
        features.append(returns.rolling(20).std().values)
        
        # Trend features
        features.append(data['Close'].pct_change(5).values)
        features.append(data['Close'].pct_change(20).values)
        
        # Range features (important for Forex)
        daily_range = (data['High'] - data['Low']) / data['Close']
        features.append(daily_range.rolling(5).mean().values)
        
        # Volume profile (if available)
        if 'Volume' in data.columns:
            volume_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
            features.append(volume_ratio.values)
        
        # Stack features
        features = np.column_stack([f[~np.isnan(f)] for f in features])
        return self.scaler.fit_transform(features)
    
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regimes"""
        features = self.extract_regime_features(data)
        
        # Fit HMM
        self.hmm_model.fit(features)
        
        # Predict regimes
        self.regime_labels = self.hmm_model.predict(features)
        
        # Map regimes to meaningful labels
        regime_mapping = self.interpret_regimes(data)
        
        return pd.Series(regime_mapping, index=data.index[-len(regime_mapping):])
    
    def interpret_regimes(self, data: pd.DataFrame) -> List[str]:
        """Interpret HMM states as market regimes"""
        regimes = []
        
        # Calculate statistics for each regime
        returns = data['Close'].pct_change().dropna().iloc[-len(self.regime_labels):]
        volatilities = returns.rolling(5).std()
        
        for regime in range(self.n_regimes):
            regime_returns = returns[self.regime_labels == regime]
            regime_vol = volatilities[self.regime_labels == regime]
            
            if len(regime_returns) > 0:
                avg_return = regime_returns.mean()
                avg_vol = regime_vol.mean()
                
                # Classify regime based on return/vol characteristics
                if avg_vol > volatilities.quantile(0.75):
                    if avg_return < returns.quantile(0.25):
                        regimes.append('HIGH_VOL_DOWNTURN')
                    else:
                        regimes.append('HIGH_VOL_UPTREND')
                else:
                    if abs(avg_return) < returns.abs().quantile(0.25):
                        regimes.append('LOW_VOL_RANGING')
                    elif avg_return > 0:
                        regimes.append('TREND_UP')
                    else:
                        regimes.append('TREND_DOWN')
            else:
                regimes.append('UNDEFINED')
        
        return regimes