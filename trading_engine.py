# trading_engine.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ForexGoldTradingSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all statistical and ML models"""
        # Time Series Models
        self.arma_model = ARMAModel()
        self.garch_model = GARCHModel()
        
        # Cointegration Engine
        self.johansen_test = JohansenCointegration()
        self.kalman_filter = KalmanFilterPairs()
        
        # Machine Learning Models
        self.random_forest = RandomForestForecaster()
        self.svm_model = SVMForecaster()
        self.hmm_model = HMMRegimeDetector()
        
        # Bayesian Models
        self.bayesian_reg = BayesianRegression()
        self.stoch_vol = StochasticVolatility()
        
        # Risk Management
        self.risk_manager = RiskManager()
        
    def process_market_data(self, data: pd.DataFrame) -> Dict:
        """Main pipeline for processing market data"""
        signals = {}
        
        # 1. Time Series Analysis
        ts_features = self.extract_time_series_features(data)
        
        # 2. Cointegration Analysis for FX pairs
        coint_signals = self.detect_cointegrated_pairs(data)
        
        # 3. Regime Detection
        regime = self.hmm_model.detect_regime(data)
        
        # 4. Machine Learning Predictions
        ml_predictions = self.ml_pipeline(data, regime)
        
        # 5. Bayesian Inference
        bayesian_signals = self.bayesian_pipeline(data)
        
        # 6. Generate Trading Signals
        signals = self.generate_signals(
            ts_features, coint_signals, regime,
            ml_predictions, bayesian_signals
        )
        
        # 7. Risk Management Overlay
        signals = self.risk_manager.apply_risk_rules(signals, regime)
        
        return signals