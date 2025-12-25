# risk_management.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class ForexRiskManager:
    """Advanced risk management for Forex trading"""
    
    def __init__(self, max_position_size: float = 0.1,
                 max_daily_loss: float = 0.02,
                 max_portfolio_var: float = 0.01):
        
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_var = max_portfolio_var
        self.position_sizing_model = self.initialize_position_sizing()
        
    def initialize_position_sizing(self):
        """Kelly criterion and volatility-based position sizing"""
        def kelly_position_size(win_rate: float, 
                               win_loss_ratio: float) -> float:
            """Kelly criterion for optimal position sizing"""
            if win_loss_ratio <= 0:
                return 0
            kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
            return max(0, min(kelly * 0.5, self.max_position_size))
        
        def volatility_position_size(current_vol: float,
                                    avg_vol: float,
                                    regime: str) -> float:
            """Volatility-adjusted position sizing"""
            vol_ratio = current_vol / avg_vol
            
            if regime == 'HIGH_VOL':
                return self.max_position_size * 0.5 / vol_ratio
            elif regime == 'LOW_VOL':
                return self.max_position_size * 0.8
            else:
                return self.max_position_size * 0.65
        
        return {
            'kelly': kelly_position_size,
            'vol_adjusted': volatility_position_size
        }
    
    def calculate_var(self, positions: Dict, 
                     covariance_matrix: pd.DataFrame,
                     confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk"""
        
        # Extract position values
        position_values = np.array(list(positions.values()))
        
        # Calculate portfolio variance
        portfolio_variance = position_values.T @ \
                            covariance_matrix.values @ \
                            position_values
        
        # Calculate VaR
        portfolio_std = np.sqrt(portfolio_variance)
        z_score = stats.norm.ppf(1 - confidence_level)
        var = abs(z_score * portfolio_std)
        
        return var
    
    def dynamic_stop_loss(self, entry_price: float,
                         current_price: float,
                         volatility: float,
                         regime: str) -> Tuple[float, float]:
        """Dynamic stop-loss and take-profit levels"""
        
        # Base stops on volatility
        atr_multiplier = {
            'HIGH_VOL': 1.0,
            'NORMAL': 1.5,
            'LOW_VOL': 2.0
        }.get(regime, 1.5)
        
        stop_distance = volatility * atr_multiplier
        
        # Direction-agnostic stops
        if current_price >= entry_price:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 2)  # 2:1 RR
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 2)
        
        return stop_loss, take_profit
    
    def check_risk_limits(self, portfolio: Dict, 
                         daily_pnl: float,
                         current_var: float) -> bool:
        """Check if risk limits are breached"""
        
        # Daily loss limit
        if daily_pnl < -self.max_daily_loss * self.initial_capital:
            return False
        
        # VaR limit
        if current_var > self.max_portfolio_var * self.initial_capital:
            return False
        
        # Concentration risk
        max_position = max(abs(p) for p in portfolio.values())
        if max_position > self.max_position_size * self.initial_capital:
            return False
        
        return True