# main.py
import asyncio
import pandas as pd
from datetime import datetime
import schedule
import time

class ForexGoldTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self.load_config(config_path)
        self.trading_system = ForexGoldTradingSystem(self.config)
        self.data_fetcher = DataFetcher(self.config['data_sources'])
        self.risk_manager = ForexRiskManager()
        self.performance_tracker = PerformanceTracker()
        
        # Initialize Web UI
        self.dashboard = TradingDashboard()
        
    async def run_trading_cycle(self):
        """Main trading cycle"""
        while True:
            try:
                # 1. Fetch latest data
                data = await self.data_fetcher.fetch_latest_data()
                
                # 2. Generate signals
                signals = self.trading_system.process_market_data(data)
                
                # 3. Apply risk management
                approved_signals = self.risk_manager.validate_signals(
                    signals, 
                    self.performance_tracker.current_positions
                )
                
                # 4. Execute trades
                if approved_signals:
                    await self.execute_trades(approved_signals)
                
                # 5. Update dashboard
                self.dashboard.update(
                    signals=signals,
                    positions=self.performance_tracker.current_positions,
                    performance=self.performance_tracker.get_metrics()
                )
                
                # 6. Log results
                self.log_trading_cycle()
                
                # 7. Wait for next cycle
                await asyncio.sleep(self.config['trading_interval'])
                
            except Exception as e:
                self.log_error(e)
                await asyncio.sleep(60)  # Wait before retrying
    
    async def execute_trades(self, signals: Dict):
        """Execute trades through broker API"""
        for symbol, signal in signals.items():
            if signal['action'] != 'HOLD':
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    symbol, 
                    signal['confidence'],
                    self.performance_tracker.current_capital
                )
                
                # Execute trade
                trade_result = await self.broker.execute_order(
                    symbol=symbol,
                    action=signal['action'],
                    quantity=position_size,
                    order_type='MARKET' if signal['urgency'] == 'HIGH' else 'LIMIT'
                )
                
                # Update tracker
                self.performance_tracker.update_positions(trade_result)
    
    def run_backtest(self, start_date: str, end_date: str):
        """Run comprehensive backtest"""
        # Load historical data
        historical_data = self.data_fetcher.fetch_historical_data(
            start_date, end_date
        )
        
        # Run backtest
        backtester = ForexBacktester()
        results = {}
        
        for symbol, data in historical_data.items():
            # Generate signals for entire period
            signals = self.trading_system.backtest_generate_signals(data)
            
            # Run backtest
            portfolio, metrics = backtester.vectorized_backtest(
                signals, data
            )
            
            # Calculate strategy decay
            decay = backtester.calculate_strategy_decay(portfolio)
            
            results[symbol] = {
                'portfolio': portfolio,
                'metrics': metrics,
                'decay': decay
            }
        
        return results

if __name__ == "__main__":
    # Initialize and run the trading system
    bot = ForexGoldTradingBot()
    
    # Run in appropriate mode
    if bot.config['mode'] == 'backtest':
        results = bot.run_backtest(
            bot.config['backtest_start'],
            bot.config['backtest_end']
        )
        bot.dashboard.display_backtest_results(results)
        
    elif bot.config['mode'] == 'live':
        # Run Web UI in separate thread
        import threading
        ui_thread = threading.Thread(
            target=bot.dashboard.run,
            daemon=True
        )
        ui_thread.start()
        
        # Run trading bot
        asyncio.run(bot.run_trading_cycle())