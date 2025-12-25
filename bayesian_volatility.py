# bayesian_volatility.py
import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
from scipy import stats

class BayesianStochasticVolatility:
    """Bayesian stochastic volatility model for Forex"""
    
    def __init__(self):
        self.trace = None
        self.model = None
        
    def build_sv_model(self, returns: np.ndarray) -> pm.Model:
        """Build stochastic volatility model"""
        with pm.Model() as sv_model:
            
            # Prior for persistence of volatility
            phi = pm.Beta('phi', alpha=20, beta=1.5)
            
            # Prior for volatility of volatility
            sigma = pm.Exponential('sigma', 50)
            
            # Prior for long-run mean of volatility
            mu = pm.Normal('mu', mu=0, sigma=10)
            
            # Stochastic volatility process
            s = pm.GaussianRandomWalk('s', mu=mu, sigma=sigma, 
                                      shape=len(returns))
            
            # Observation model
            nu = pm.Exponential('nu', 1/10)
            returns_obs = pm.StudentT('returns_obs', 
                                      nu=nu,
                                      mu=0,
                                      lam=pm.math.exp(-2*s),
                                      observed=returns)
            
        return sv_model
    
    def fit_model(self, returns: pd.Series, 
                  samples: int = 2000, 
                  tune: int = 1000) -> None:
        """Fit the stochastic volatility model"""
        returns_scaled = returns * 100  # Scale for numerical stability
        
        self.model = self.build_sv_model(returns_scaled.values)
        
        with self.model:
            # Use NUTS sampler
            self.trace = pm.sample(
                samples,
                tune=tune,
                chains=2,
                cores=2,
                return_inferencedata=True,
                target_accept=0.9
            )
    
    def predict_volatility(self, steps_ahead: int = 10) -> Dict:
        """Forecast volatility"""
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        with self.model:
            # Forecast future volatility
            volatility_forecast = pm.sample_posterior_predictive(
                self.trace,
                var_names=['s'],
                samples=1000
            )
        
        # Calculate VaR and expected shortfall
        var_95 = np.percentile(volatility_forecast['s'], 5, axis=0)
        var_99 = np.percentile(volatility_forecast['s'], 1, axis=0)
        
        return {
            'volatility_forecast': volatility_forecast['s'].mean(axis=0),
            'volatility_ci': np.percentile(volatility_forecast['s'], 
                                          [5, 95], axis=0),
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': self.calculate_expected_shortfall(
                volatility_forecast['s']
            )
        }
    
    def calculate_expected_shortfall(self, samples: np.ndarray) -> float:
        """Calculate expected shortfall (CVaR)"""
        var_95 = np.percentile(samples, 5)
        es = samples[samples <= var_95].mean()
        return es