# backtesting.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ForexBacktester:
    """Advanced backtesting for Forex strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def vectorized_backtest(self, signals: pd.Series, 
                           prices: pd.DataFrame,
                           transaction_cost: float = 0.0002,  # 2 pips
                           slippage: float = 0.0001) -> pd.DataFrame:
        """Vectorized backtest for speed"""
        
        # Initialize portfolio
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['signal'] = signals
        portfolio['price'] = prices['Close']
        portfolio['returns'] = prices['Close'].pct_change()
        
        # Calculate strategy returns
        portfolio['strategy_returns'] = portfolio['signal'].shift(1) * \
                                       portfolio['returns']
        
        # Transaction costs
        portfolio['position_changes'] = portfolio['signal'].diff().abs()
        portfolio['transaction_costs'] = portfolio['position_changes'] * \
                                        transaction_cost
        
        # Slippage
        portfolio['slippage'] = portfolio['position_changes'] * slippage
        
        # Net returns
        portfolio['net_returns'] = portfolio['strategy_returns'] - \
                                  portfolio['transaction_costs'] - \
                                  portfolio['slippage']
        
        # Cumulative returns
        portfolio['cumulative_strategy'] = \
            (1 + portfolio['net_returns']).cumprod()
        portfolio['cumulative_buy_hold'] = \
            (1 + portfolio['returns']).cumprod()
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(portfolio)
        
        return portfolio, metrics
    
    def calculate_strategy_decay(self, portfolio: pd.DataFrame,
                                window_days: int = 252) -> Dict:
        """Calculate strategy decay over time"""
        
        decay_analysis = {}
        
        # Split data into rolling windows
        for start in range(0, len(portfolio) - window_days, window_days//4):
            end = start + window_days
            window_data = portfolio.iloc[start:end]
            
            if len(window_data) < window_days//2:
                continue
            
            # Calculate Sharpe ratio for this window
            sharpe = self.calculate_sharpe_ratio(
                window_data['net_returns']
            )
            
            # Store results
            decay_analysis[window_data.index[-1]] = {
                'sharpe': sharpe,
                'returns': window_data['net_returns'].mean() * 252,
                'volatility': window_data['net_returns'].std() * np.sqrt(252),
                'max_drawdown': self.calculate_max_drawdown(
                    window_data['cumulative_strategy']
                )
            }
        
        return decay_analysis
    
    def calculate_performance_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        returns = portfolio['net_returns'].dropna()
        cumulative = portfolio['cumulative_strategy'].dropna()
        
        metrics = {
            'total_return': cumulative.iloc[-1] - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(cumulative),
            'calmar_ratio': self.calculate_calmar_ratio(returns, cumulative),
            'win_rate': (returns > 0).mean(),
            'profit_factor': abs(returns[returns > 0].sum() / 
                                returns[returns < 0].sum()),
            'average_win': returns[returns > 0].mean(),
            'average_loss': returns[returns < 0].mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
        }
        
        return metrics