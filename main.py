from data.collector import fetch_ohlcv
from strategy.base_strategy import simple_strategy, get_strategy_summary, reset_strategy, get_config_summary
from execution.execution_engine import execute_trade
from visualization.charting import plot_candles

def run_bot():
    print("Starting Trading Bot...")
    
    # Show current configuration
    get_config_summary()
    
    # Reset strategy state to load fresh config
    reset_strategy()
    
    print("Fetching market data...")
    df = fetch_ohlcv()
    print(f"Fetched {len(df)} candles")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Run backtest on all historical data
    print("\nRunning strategy backtest...")
    all_trades = []
    
    # Process each candle to find all trading opportunities
    for i in range(1, len(df)):
        # Get data up to current candle
        current_df = df.iloc[:i+1].copy()
        
        # Generate signal
        signal = simple_strategy(current_df)
        
        # Execute trade if signal exists
        if signal:
            trade = execute_trade(signal)
            if trade:
                all_trades.append(trade)
                if signal['action'] == 'BUY':
                    print(f"BUY ${signal['trade_amount']:,.2f} ({signal['shares']:.6f} shares) at ${signal['price']:.2f}")
                else:
                    print(f"SELL ${signal['trade_amount']:,.2f} ({signal['shares']:.6f} shares) at ${signal['price']:.2f} | P&L: ${signal['pnl']:+,.2f} ({signal['pnl_percent']:+.2f}%)")
    
    # Get strategy summary
    summary = get_strategy_summary()
    
    # Print detailed results
    print("\n" + "="*70)
    print("TRADING RESULTS")
    print("="*70)
    print(f"Initial Balance:        ${summary['initial_balance']:,.2f}")
    print(f"Final Balance:          ${summary['current_balance']:,.2f}")
    print(f"Total Return:           ${summary['total_pnl']:+,.2f} ({summary['total_return_percent']:+.2f}%)")
    print(f"Position Size:          {summary['position_size_percent']}% per trade")
    print()
    print(f"Total Signals:          {len(all_trades)}")
    print(f"Buy Signals:            {summary['buy_signals']}")
    print(f"Sell Signals:           {summary['sell_signals']}")
    print(f"Completed Trades:       {summary['total_trades']}")
    print(f"Total Trading Costs:    ${summary.get('total_trading_costs', 0):,.2f}")
    
    if summary['total_trades'] > 0:
        avg_pnl = summary['total_pnl'] / summary['total_trades']
        print(f"Avg P&L per Trade:      ${avg_pnl:+,.2f}")
        
        # Calculate win rate
        winning_trades = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
        if summary['total_trades'] > 0:
            win_rate = (winning_trades / summary['total_trades']) * 100
            print(f"Win Rate:               {win_rate:.1f}% ({winning_trades}/{summary['total_trades']})")
    
    # Current position info
    print()
    print(f"Current Position:       {summary['current_position'] or 'None'}")
    if summary['current_position']:
        current_value = summary['shares_held'] * df.iloc[-1]['close']
        unrealized_pnl = current_value - summary['invested_amount']
        print(f"Shares Held:            {summary['shares_held']:.6f}")
        print(f"Invested Amount:        ${summary['invested_amount']:,.2f}")
        print(f"Current Value:          ${current_value:,.2f}")
        print(f"Unrealized P&L:         ${unrealized_pnl:+,.2f}")
    
    # Check for issues
    if summary['buy_signals'] != summary['sell_signals']:
        print(f"\nWARNING: Open position! BUY: {summary['buy_signals']}, SELL: {summary['sell_signals']}")
    else:
        print("\nAll positions closed properly")
    
    print("="*70)
    
    # Generate charts
    print(f"\nGenerating charts...")
    try:
        plot_candles(df, all_trades, "backtest_results.png")
        print("Chart saved as charts/backtest_results.png")
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()
    
    return df, all_trades, summary

def run_live_mode():
    """
    FIXED: Run in live mode - now works properly by maintaining strategy state
    """
    print("Running in LIVE mode...")
    
    # Show configuration but DON'T reset strategy (maintain state between calls)
    get_config_summary()
    
    # Only reset if no previous state exists
    summary = get_strategy_summary()
    if len(summary['all_trades']) == 0:
        print("No previous trades found - initializing fresh strategy state")
        reset_strategy()
    else:
        print(f"Continuing with existing strategy state - {len(summary['all_trades'])} previous trades")
    
    print("Fetching latest market data...")
    df = fetch_ohlcv()
    print(f"Latest price: ${df.iloc[-1]['close']:.2f}")
    
    # FIXED: Use enough historical data for strategy (not just last candle)
    # Strategy needs at least 2 candles to compare prices
    recent_df = df.tail(50)  # Use last 50 candles for context
    
    # Run strategy on recent data
    signal = simple_strategy(recent_df)
    
    # Execute trade (mock)
    trade = execute_trade(signal)
    
    # Show results
    recent_trades = []
    if trade:
        recent_trades.append(trade)
        if signal['action'] == 'BUY':
            print(f"NEW BUY SIGNAL:")
            print(f"   Amount: ${signal['trade_amount']:,.2f}")
            print(f"   Shares: {signal['shares']:.6f}")
            print(f"   Price: ${signal['price']:.2f}")
            print(f"   Reason: {signal['reason']}")
        else:
            print(f"NEW SELL SIGNAL:")
            print(f"   Amount: ${signal['trade_amount']:,.2f}")
            print(f"   P&L: ${signal['pnl']:+,.2f} ({signal['pnl_percent']:+.2f}%)")
            print(f"   Price: ${signal['price']:.2f}")
            print(f"   Reason: {signal['reason']}")
    else:
        print("No trading signal generated")
    
    # Show all trades for live chart (recent + any new)
    all_live_trades = get_strategy_summary()['all_trades']
    
    # Generate chart with all trades
    try:
        plot_candles(df, all_live_trades, "live_results.png")
        print("Live chart saved as charts/live_results.png")
    except Exception as e:
        print(f"Error generating live chart: {e}")
    
    # Show current portfolio status
    updated_summary = get_strategy_summary()
    print(f"\nPortfolio Status:")
    print(f"   Balance: ${updated_summary['current_balance']:,.2f}")
    print(f"   Position: {updated_summary['current_position'] or 'None'}")
    print(f"   Total P&L: ${updated_summary['total_pnl']:+,.2f}")
    print(f"   Total Trades: {len(updated_summary['all_trades'])}")
    
    return df, recent_trades, updated_summary

if __name__ == "__main__":
    import os
    
    # Create necessary directories
    os.makedirs("charts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("Trading Bot")
    print("Choose mode:")
    print("1. Backtest (full historical analysis)")
    print("2. Live mode (latest signal + maintain state)")
    
    try:
        choice = input("Enter 1 or 2 (or press Enter for backtest): ").strip()
        
        if choice == "2":
            run_live_mode()
        else:
            run_bot()
            
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()