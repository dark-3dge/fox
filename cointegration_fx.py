# cointegration_fx.py
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pykalman import KalmanFilter

class ForexPairsTrading:
    """Cointegration-based pairs trading for Forex"""
    
    def __init__(self):
        self.pairs = [
            ('EURUSD', 'GBPUSD'),  # European currencies
            ('AUDUSD', 'NZDUSD'),  # Commodity currencies
            ('USDCHF', 'XAUUSD'),  # Safe havens
            ('EURUSD', 'USDCHF')   # Classic pair
        ]
        
    def johansen_test_fx(self, data1: pd.Series, 
                         data2: pd.Series, 
                         lag: int = 1) -> Dict:
        """Johansen test for cointegration with FX data"""
        # Create matrix of both series
        X = np.column_stack([data1.values, data2.values])
        
        # Johansen test
        result = coint_johansen(X, det_order=0, k_ar_diff=lag)
        
        # Trace test and max eigenvalue test
        trace_stat = result.lr1[0]  # Trace statistic
        max_eig_stat = result.lr2[0]  # Max eigenvalue statistic
        
        # Critical values (95%)
        trace_crit = result.cvt[:, 0]
        max_eig_crit = result.cvm[:, 0]
        
        cointegrated = trace_stat > trace_crit[0]
        
        return {
            'cointegrated': cointegrated,
            'trace_stat': trace_stat,
            'max_eig_stat': max_eig_stat,
            'eigenvectors': result.evec[:, 0],
            'hedge_ratio': result.evec[0, 0] / result.evec[1, 0]
        }
    
    def kalman_filter_hedge_ratio(self, x: np.ndarray, 
                                 y: np.ndarray) -> np.ndarray:
        """Dynamic hedge ratio estimation using Kalman Filter"""
        # State transition matrix (constant hedge ratio but allows drift)
        transition_matrix = [[1, 0], [0, 1]]
        
        # Observation matrix
        observation_matrix = [[1, 1]]
        
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=[0, 0],
            initial_state_covariance=np.eye(2),
            observation_covariance=1,
            transition_covariance=np.eye(2) * 0.01
        )
        
        # The state contains [alpha, beta] where y = alpha + beta*x
        states, _ = kf.filter(y.reshape(-1, 1))
        
        return states[:, 1]  # Return beta (hedge ratio)
    
    def generate_pairs_signals(self, price1: pd.Series, 
                              price2: pd.Series,
                              window: int = 20) -> pd.Series:
        """Generate trading signals for cointegrated pair"""
        # Calculate spread
        hedge_ratio = self.kalman_filter_hedge_ratio(
            price1.values, price2.values
        )
        
        # Use latest hedge ratio
        current_hr = hedge_ratio[-1]
        spread = price2 - current_hr * price1
        
        # Z-score normalization
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        zscore = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals = pd.Series(0, index=price1.index)
        
        # Long signal when spread is too low
        signals[zscore < -2] = 1
        # Short signal when spread is too high
        signals[zscore > 2] = -1
        # Exit when spread reverts
        signals[(zscore > -0.5) & (zscore < 0.5)] = 0
        
        return signals