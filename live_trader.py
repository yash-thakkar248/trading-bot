#!/usr/bin/env python3
"""
Multi-Strategy Trading Bot Main Entry Point
Supports backtest, single live check, and integrates with new strategy system
"""

import time
import signal
import sys
from datetime import datetime, timedelta
from data.collector import fetch_ohlcv
from strategy.base_strategy import strategy_manager
from execution.execution_engine import execute_trade
from visualization.charting import plot_candles, plot_strategy_comparison
from visualization.charting import plot_candles

class LiveTradingBot:
    def __init__(self):
        self.running = True
        self.last_candle_time = None
        self.iteration_count = 0
        self.last_chart_update = None
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] Shutdown signal received...")
        self.running = False
        
    def log_with_timestamp(self, message):
        """Print message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
        
    def wait_for_new_candle(self):
        """Wait for a new 1-minute candle to form"""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        
        if self.last_candle_time is None:
            self.last_candle_time = current_minute
            return True
            
        # Check if we're in a new minute
        if current_minute > self.last_candle_time:
            self.last_candle_time = current_minute
            return True
            
        return False
        
    def fetch_latest_data(self):
        """Fetch the latest market data"""
        try:
            df = fetch_ohlcv()
            if df is None or len(df) == 0:
                self.log_with_timestamp("ERROR: No data received")
                return None
                
            latest_candle = df.iloc[-1]
            self.log_with_timestamp(
                f"Latest: ${latest_candle['close']:.2f} "
                f"(O:${latest_candle['open']:.2f} H:${latest_candle['high']:.2f} "
                f"L:${latest_candle['low']:.2f} V:{latest_candle['volume']:.1f})"
            )
            return df
            
        except Exception as e:
            self.log_with_timestamp(f"ERROR fetching data: {e}")
            return None
            
    def process_trading_signals(self, df):
        """Process the latest data and execute any trading signals from all strategies"""
        try:
            # Generate signals from all active strategies
            signals = strategy_manager.generate_signals(df)
            
            executed_trades = []
            
            if signals:
                for signal in signals:
                    # Execute the trade
                    trade = execute_trade(signal)
                    
                    if trade:
                        executed_trades.append(trade)
                        if signal['action'] == 'BUY':
                            self.log_with_timestamp(
                                f"[{signal['strategy']}] BUY EXECUTED: ${signal['trade_amount']:,.2f} "
                                f"({signal['shares']:.6f} shares) at ${signal['price']:.2f}"
                            )
                        else:
                            self.log_with_timestamp(
                                f"[{signal['strategy']}] SELL EXECUTED: ${signal['trade_amount']:,.2f} "
                                f"at ${signal['price']:.2f} | P&L: ${signal.get('pnl', 0):+,.2f}"
                            )
                    else:
                        self.log_with_timestamp(f"WARNING: [{signal['strategy']}] Signal generated but trade execution failed")
                
                return executed_trades
            else:
                # No signals - show brief status
                summary = strategy_manager.get_summary()
                active_positions = [name for name, data in summary['strategy_summaries'].items() 
                                  if data['current_position']]
                
                position_status = f"Positions: {', '.join(active_positions) if active_positions else 'None'}"
                balance_status = f"Balance: ${summary['current_balance']:,.2f}"
                self.log_with_timestamp(f"No signals | {position_status} | {balance_status}")
                
        except Exception as e:
            self.log_with_timestamp(f"ERROR processing signals: {e}")
            
        return []
        
    def update_charts(self, df, force=False):
        """Update charts periodically or on force"""
        now = datetime.now()
        
        # Update charts every 5 minutes or on force
        if (force or 
            self.last_chart_update is None or 
            (now - self.last_chart_update).seconds >= 300):
            
            try:
                summary = strategy_manager.get_summary()
                all_trades = summary['all_trades']
                
                if len(all_trades) > 0:
                    chart_filename = f"live_multi_strategy_{now.strftime('%Y%m%d_%H%M')}.png"
                    plot_candles(df, all_trades, chart_filename)
                    self.log_with_timestamp(f"Chart updated: charts/{chart_filename}")
                else:
                    self.log_with_timestamp("Chart skipped: No trades to display")
                    
                self.last_chart_update = now
                
            except Exception as e:
                self.log_with_timestamp(f"ERROR updating charts: {e}")
                
    def print_summary(self):
        """Print current trading summary"""
        summary = strategy_manager.get_summary()
        
        print("\n" + "="*70)
        print("LIVE MULTI-STRATEGY SUMMARY")
        print("="*70)
        print(f"Running Time: {self.iteration_count} iterations")
        print(f"Active Strategies: {', '.join(summary['active_strategies'])}")
        print(f"Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"Current Balance: ${summary['current_balance']:,.2f}")
        print(f"Total P&L: ${summary['total_pnl']:+,.2f} ({summary.get('total_return_percent', 0):+.2f}%)")
        print(f"Total Trades: {summary['total_trades']}")
        
        print(f"\nPER-STRATEGY BREAKDOWN:")
        for strategy_name, strategy_data in summary['strategy_summaries'].items():
            print(f"  {strategy_name}:")
            print(f"    Position: {strategy_data['current_position'] or 'None'}")
            print(f"    Trades: {strategy_data['trades']}")
            print(f"    P&L: ${strategy_data['pnl']:+,.2f}")
            
        print("="*70 + "\n")
        
    def run_continuous(self):
        """Main live trading loop"""
        self.log_with_timestamp("Starting Continuous Multi-Strategy Live Trading...")
        
        # Show configuration
        strategy_manager.get_config_summary()
        
        # Reset all strategies for fresh start
        strategy_manager.reset()
        
        self.log_with_timestamp("Bot initialized. Entering live trading loop...")
        self.log_with_timestamp("Press Ctrl+C to stop the bot safely")
        
        try:
            while self.running:
                if self.wait_for_new_candle() or self.iteration_count == 0:
                    self.iteration_count += 1
                    
                    self.log_with_timestamp(f"--- Iteration {self.iteration_count} ---")
                    
                    # Fetch latest market data
                    df = self.fetch_latest_data()
                    if df is None:
                        self.log_with_timestamp("Skipping iteration due to data fetch failure")
                        time.sleep(10)
                        continue
                    
                    # Process trading signals from all strategies
                    trades = self.process_trading_signals(df)
                    
                    # Update charts if there were trades
                    if trades:
                        self.update_charts(df, force=True)
                    elif self.iteration_count % 10 == 0:
                        self.update_charts(df)
                    
                    # Print summary every 30 iterations
                    if self.iteration_count % 30 == 0:
                        self.print_summary()
                
                # Sleep for 5 seconds before checking again
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.log_with_timestamp("Keyboard interrupt received")
        except Exception as e:
            self.log_with_timestamp(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
            
    def shutdown(self):
        """Clean shutdown procedure"""
        self.log_with_timestamp("Shutting down Multi-Strategy Live Trading Bot...")
        
        # Get final data for last chart update
        try:
            df = self.fetch_latest_data()
            if df is not None:
                self.update_charts(df, force=True)
        except:
            pass
        
        # Print final summary
        self.print_summary()
        
        self.log_with_timestamp("Bot shutdown complete. Final charts and summary saved.")

def run_backtest():
    """Run historical backtest with multi-strategy system"""
    print("Starting Multi-Strategy Trading Bot Backtest...")
    
    # Show configuration
    strategy_manager.get_config_summary()
    
    # Reset all strategies
    strategy_manager.reset()
    
    print("Fetching market data...")
    df = fetch_ohlcv()
    print(f"Fetched {len(df)} candles")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    print(f"\nRunning backtest with {len(strategy_manager.active_strategies)} active strategies...")
    all_trades = []
    
    # Process each candle
    for i in range(1, len(df)):
        current_df = df.iloc[:i+1].copy()
        
        # Generate signals from all active strategies
        signals = strategy_manager.generate_signals(current_df)
        
        # Execute all signals
        for signal in signals:
            trade = execute_trade(signal)
            if trade:
                all_trades.append(trade)
                if signal['action'] == 'BUY':
                    print(f"[{signal['strategy']}] BUY ${signal['trade_amount']:,.2f} ({signal['shares']:.6f} shares) at ${signal['price']:.2f}")
                else:
                    print(f"[{signal['strategy']}] SELL ${signal['trade_amount']:,.2f} at ${signal['price']:.2f} | P&L: ${signal.get('pnl', 0):+,.2f}")
    
    # Get comprehensive summary
    summary = strategy_manager.get_summary()
    
    # Print detailed results
    print("\n" + "="*80)
    print("MULTI-STRATEGY BACKTEST RESULTS")
    print("="*80)
    print(f"Active Strategies:       {', '.join(summary['active_strategies'])}")
    print(f"Initial Balance:         ${summary['initial_balance']:,.2f}")
    print(f"Final Balance:           ${summary['current_balance']:,.2f}")
    print(f"Total Return:            ${summary['total_pnl']:+,.2f} ({summary.get('total_return_percent', 0):+.2f}%)")
    print()
    print(f"Total Signals:           {len(all_trades)}")
    print(f"Buy Signals:             {summary['buy_signals']}")
    print(f"Sell Signals:            {summary['sell_signals']}")
    print(f"Completed Trades:        {summary['total_trades']}")
    
    if summary['total_trades'] > 0:
        avg_pnl = summary['total_pnl'] / summary['total_trades']
        print(f"Avg P&L per Trade:       ${avg_pnl:+,.2f}")
        
        # Calculate overall win rate
        winning_trades = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
        win_rate = (winning_trades / summary['total_trades']) * 100
        print(f"Overall Win Rate:        {win_rate:.1f}% ({winning_trades}/{summary['total_trades']})")
    
    # Strategy-specific breakdown
    print(f"\nSTRATEGY BREAKDOWN:")
    for strategy_name, strategy_data in summary['strategy_summaries'].items():
        print(f"  {strategy_name.upper()}:")
        print(f"    Current Position: {strategy_data['current_position'] or 'None'}")
        print(f"    Completed Trades: {strategy_data['trades']}")
        print(f"    Total P&L: ${strategy_data['pnl']:+,.2f}")
        if strategy_data['trades'] > 0:
            avg_strategy_pnl = strategy_data['pnl'] / strategy_data['trades']
            print(f"    Avg P&L/Trade: ${avg_strategy_pnl:+,.2f}")
        print(f"    Strategy Type: {strategy_data['params'].get('type', 'Unknown')}")
    
    print("="*80)
    
    # Generate enhanced charts
    print(f"\nGenerating multi-strategy charts...")
    try:
        plot_candles(df, all_trades, "multi_strategy_backtest.png")
        plot_strategy_comparison(all_trades, "strategy_comparison.png")
        print("Charts saved:")
        print("  - charts/multi_strategy_backtest.png (main trading chart)")
        print("  - charts/strategy_comparison.png (strategy performance comparison)")
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()
    
    return df, all_trades, summary

def run_single_live_check():
    """Run single live mode check with multi-strategy system"""
    print("Running Single Multi-Strategy Live Check...")
    
    # Show configuration
    strategy_manager.get_config_summary()
    
    # Get current state
    summary = strategy_manager.get_summary()
    if len(summary['all_trades']) == 0:
        print("No previous trades - initializing fresh strategies")
        strategy_manager.reset()
    else:
        print(f"Continuing with existing state - {len(summary['all_trades'])} previous trades")
    
    print("Fetching latest market data...")
    df = fetch_ohlcv()
    print(f"Latest price: ${df.iloc[-1]['close']:.2f}")
    
    # Use recent data for strategies
    recent_df = df.tail(50)
    
    # Generate signals from all active strategies
    signals = strategy_manager.generate_signals(recent_df)
    
    executed_trades = []
    if signals:
        print(f"Generated {len(signals)} signals from active strategies:")
        
        for signal in signals:
            trade = execute_trade(signal)
            if trade:
                executed_trades.append(trade)
                if signal['action'] == 'BUY':
                    print(f"[{signal['strategy']}] NEW BUY SIGNAL:")
                    print(f"   Amount: ${signal['trade_amount']:,.2f}")
                    print(f"   Shares: {signal['shares']:.6f}")
                    print(f"   Price: ${signal['price']:.2f}")
                    print(f"   Reason: {signal['reason']}")
                else:
                    print(f"[{signal['strategy']}] NEW SELL SIGNAL:")
                    print(f"   Amount: ${signal['trade_amount']:,.2f}")
                    print(f"   P&L: ${signal.get('pnl', 0):+,.2f}")
                    print(f"   Price: ${signal['price']:.2f}")
                    print(f"   Reason: {signal['reason']}")
    else:
        print("No trading signals generated from any active strategy")
    
    # Generate chart with all trades
    all_trades = strategy_manager.get_summary()['all_trades']
    try:
        plot_candles(df, all_trades, "live_single_multi_strategy.png")
        print("Live chart saved as charts/live_single_multi_strategy.png")
    except Exception as e:
        print(f"Error generating chart: {e}")
    
    # Show current portfolio status
    updated_summary = strategy_manager.get_summary()
    print(f"\nMulti-Strategy Portfolio Status:")
    print(f"   Balance: ${updated_summary['current_balance']:,.2f}")
    print(f"   Total P&L: ${updated_summary['total_pnl']:+,.2f}")
    print(f"   Active Strategies: {', '.join(updated_summary['active_strategies'])}")
    print(f"   Total Trades: {len(updated_summary['all_trades'])}")
    
    # Show individual strategy positions
    for strategy_name, strategy_data in updated_summary['strategy_summaries'].items():
        position = strategy_data['current_position'] or 'None'
        print(f"   {strategy_name}: {position} (P&L: ${strategy_data['pnl']:+,.2f})")

def run_continuous_live_trading():
    """Run continuous live trading with multi-strategy system"""
    print("CONTINUOUS MULTI-STRATEGY LIVE TRADING")
    print("=======================================")
    print("This will run continuously with multiple strategies making real trading decisions.")
    print("Each active strategy can trade independently every minute.")
    print()
    
    # Show active strategies
    print(f"Active Strategies: {', '.join(strategy_manager.active_strategies.keys())}")
    print()
    
    confirm = input("Are you sure you want to start continuous multi-strategy live trading? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Continuous live trading cancelled.")
        return
    
    bot = LiveTradingBot()
    bot.run_continuous()

if __name__ == "__main__":
    import os
    
    # Create necessary directories
    os.makedirs("charts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("MULTI-STRATEGY TRADING BOT")
    print("===========================")
    print("Choose trading mode:")
    print("1. Backtest (full historical analysis with multiple strategies)")
    print("2. Single Live Check (one-time check across all active strategies)")
    print("3. Continuous Live Trading (infinite loop with multiple strategies)")
    
    try:
        choice = input("\nEnter 1, 2, or 3 (or press Enter for backtest): ").strip()
        
        if choice == "2":
            run_single_live_check()
        elif choice == "3":
            run_continuous_live_trading()
        else:
            run_backtest()
            
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()