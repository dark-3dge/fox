# Create environment
conda create -n forex_trading python=3.9
conda activate forex_trading

# Install Python dependencies
pip install -r requirements.txt

# Install R dependencies
Rscript install_r_packages.R

# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
statsmodels>=0.12.0
arch>=4.15.0
pymc3>=3.11.0
pykalman>=0.9.5
hmmlearn>=0.2.7
xgboost>=1.4.0
lightgbm>=3.2.0
streamlit>=1.0.0
plotly>=5.3.0
dash>=2.0.0
fastapi>=0.70.0
uvicorn>=0.15.0
sqlalchemy>=1.4.0
ta-lib>=0.4.0
ccxt>=2.0.0
yfinance>=0.1.70
python-dotenv>=0.19.0