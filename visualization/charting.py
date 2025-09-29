import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from datetime import datetime, timedelta
import json

class TradeReportGenerator:
    """Generate detailed trade reports with enhanced charts"""
    
    def __init__(self):
        self.report_dir = "trade_reports"
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(f"{self.report_dir}/charts", exist_ok=True)
        
    def generate_comprehensive_trade_report(self, df, all_trades, filename_prefix="trade_report"):
        """Generate a comprehensive report for all trades with individual analysis"""
        print(f"Generating comprehensive trade reports...")
        
        # Group trades by strategy and pair them (buy-sell)
        trade_pairs = self._pair_trades_by_strategy(all_trades)
        
        if not trade_pairs:
            print("No completed trade pairs found for reporting")
            return 0
        
        # Generate summary report
        summary_report = self._generate_summary_report(trade_pairs, all_trades)
        
        # Generate individual trade reports
        individual_reports = []
        for pair_id, (pair_key, (strategy_name, buy_trade, sell_trade)) in enumerate(trade_pairs.items(), 1):
            individual_report = self._generate_individual_trade_report(
                df, buy_trade, sell_trade, strategy_name, pair_id
            )
            individual_reports.append(individual_report)
        
        # Create master HTML report
        self._create_master_html_report(summary_report, individual_reports, filename_prefix)
        
        print(f"Trade reports generated in '{self.report_dir}' directory")
        return len(trade_pairs)
    
    def _pair_trades_by_strategy(self, all_trades):
        """Pair buy and sell trades by strategy"""
        trade_pairs = {}
        strategy_trades = defaultdict(list)
        
        # Group trades by strategy
        for trade in all_trades:
            strategy_name = trade.get('strategy', 'Unknown')
            strategy_trades[strategy_name].append(trade)
        
        pair_id = 1
        for strategy_name, trades in strategy_trades.items():
            # Sort trades by time to ensure proper pairing
            trades.sort(key=lambda x: pd.to_datetime(x['time']))
            
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            # Pair trades chronologically
            for i, buy_trade in enumerate(buy_trades):
                if i < len(sell_trades):
                    sell_trade = sell_trades[i]
                    trade_pairs[f"{strategy_name}_{pair_id}"] = (strategy_name, buy_trade, sell_trade)
                    pair_id += 1
        
        return trade_pairs
    
    def _generate_individual_trade_report(self, df, buy_trade, sell_trade, strategy_name, pair_id):
        """Generate detailed report for a single trade pair"""
        
        # Calculate trade metrics
        buy_price = buy_trade['price']
        sell_price = sell_trade['price']
        buy_time = pd.to_datetime(buy_trade['time'])
        sell_time = pd.to_datetime(sell_trade['time'])
        
        pnl = sell_trade.get('pnl', sell_price - buy_price)
        pnl_percent = sell_trade.get('pnl_percent', ((sell_price - buy_price) / buy_price) * 100)
        trade_amount = buy_trade.get('trade_amount', 0)
        shares = buy_trade.get('shares', 0)
        
        # Extract trade window data
        trade_df = self._extract_trade_window_data(df, buy_time, sell_time)
        
        # Generate enhanced chart for this trade
        chart_filename = f"trade_{pair_id}_{strategy_name}_chart.png"
        self._create_individual_trade_chart(trade_df, buy_trade, sell_trade, chart_filename)
        
        # Analyze trade performance
        analysis = self._analyze_trade_performance(trade_df, buy_trade, sell_trade)
        
        # Create detailed report data
        report_data = {
            'trade_id': pair_id,
            'strategy': strategy_name,
            'buy_details': {
                'price': buy_price,
                'time': buy_time.strftime('%Y-%m-%d %H:%M:%S'),
                'reason': buy_trade.get('reason', 'N/A'),
                'amount': trade_amount,
                'shares': shares
            },
            'sell_details': {
                'price': sell_price,
                'time': sell_time.strftime('%Y-%m-%d %H:%M:%S'),
                'reason': sell_trade.get('reason', 'N/A')
            },
            'performance': {
                'pnl_dollar': pnl,
                'pnl_percent': pnl_percent,
                'duration': str(sell_time - buy_time),
                'duration_hours': (sell_time - buy_time).total_seconds() / 3600
            },
            'analysis': analysis,
            'chart_filename': chart_filename
        }
        
        return report_data
    
    def _extract_trade_window_data(self, df, buy_time, sell_time, buffer_hours=24):
        """Extract data around the trade period with buffer"""
        start_time = buy_time - timedelta(hours=buffer_hours)
        end_time = sell_time + timedelta(hours=buffer_hours)
        
        # Ensure timestamp column exists
        if 'timestamp' in df.columns:
            df_time = pd.to_datetime(df['timestamp'])
        else:
            df_time = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        
        # Filter data within time window
        mask = (df_time >= start_time) & (df_time <= end_time)
        trade_window = df[mask].copy()
        
        if 'timestamp' not in trade_window.columns and not isinstance(trade_window.index, pd.DatetimeIndex):
            trade_window['timestamp'] = df_time[mask].values
        
        return trade_window
    
    def _create_individual_trade_chart(self, trade_df, buy_trade, sell_trade, filename):
        """Create enhanced chart for individual trade"""
        if len(trade_df) < 2:
            return
            
        # Prepare data for mplfinance
        if 'timestamp' in trade_df.columns:
            plot_df = trade_df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']].copy()
        else:
            plot_df = trade_df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Create figure for detailed analysis
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), 
                                gridspec_kw={'height_ratios': [3, 1, 1]}, 
                                sharex=True)
        
        # Main candlestick plot
        ax_main = axes[0]
        ax_volume = axes[1] 
        ax_indicators = axes[2]
        
        # Plot candlesticks manually
        for idx, (timestamp, row) in enumerate(plot_df.iterrows()):
            color = 'green' if row['close'] > row['open'] else 'red'
            
            # Body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            
            ax_main.add_patch(plt.Rectangle((idx-0.3, body_bottom), 0.6, body_height, 
                                          facecolor=color, alpha=0.7))
            
            # Wicks
            ax_main.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=1)
        
        # Plot volume
        for idx, (timestamp, row) in enumerate(plot_df.iterrows()):
            color = 'green' if row['close'] > row['open'] else 'red'
            ax_volume.bar(idx, row['volume'], color=color, alpha=0.6)
        
        # Add moving averages
        plot_df['SMA_20'] = plot_df['close'].rolling(20).mean()
        plot_df['EMA_12'] = plot_df['close'].ewm(span=12).mean()
        
        x_range = range(len(plot_df))
        ax_main.plot(x_range, plot_df['SMA_20'], color='blue', linewidth=2, alpha=0.7, label='SMA(20)')
        ax_main.plot(x_range, plot_df['EMA_12'], color='orange', linewidth=2, alpha=0.7, label='EMA(12)')
        
        # Add RSI to indicators subplot
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        rsi = calculate_rsi(plot_df['close'])
        ax_indicators.plot(x_range, rsi, color='purple', linewidth=2, label='RSI(14)')
        ax_indicators.axhline(70, color='red', linestyle='--', alpha=0.7)
        ax_indicators.axhline(30, color='red', linestyle='--', alpha=0.7)
        ax_indicators.set_ylim(0, 100)
        
        # Find buy and sell points in the data
        buy_time = pd.to_datetime(buy_trade['time'])
        sell_time = pd.to_datetime(sell_trade['time'])
        
        buy_idx = None
        sell_idx = None
        
        for idx, timestamp in enumerate(plot_df.index):
            if abs((timestamp - buy_time).total_seconds()) < 3600:  # Within 1 hour
                buy_idx = idx
            if abs((timestamp - sell_time).total_seconds()) < 3600:  # Within 1 hour
                sell_idx = idx
        
        # Add trade markers
        if buy_idx is not None:
            ax_main.scatter(buy_idx, buy_trade['price'], color='lime', s=200, 
                          marker='^', zorder=5, edgecolors='black', linewidth=2)
            ax_main.annotate(f'BUY\n${buy_trade["price"]:.2f}\n{buy_time.strftime("%m/%d %H:%M")}',
                           xy=(buy_idx, buy_trade['price']),
                           xytext=(20, 30), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=10)
        
        if sell_idx is not None:
            pnl = sell_trade.get('pnl', 0)
            pnl_percent = sell_trade.get('pnl_percent', 0)
            ax_main.scatter(sell_idx, sell_trade['price'], color='red', s=200,
                          marker='v', zorder=5, edgecolors='black', linewidth=2)
            ax_main.annotate(f'SELL\n${sell_trade["price"]:.2f}\n{sell_time.strftime("%m/%d %H:%M")}\nP&L: ${pnl:+.2f} ({pnl_percent:+.1f}%)',
                           xy=(sell_idx, sell_trade['price']),
                           xytext=(-80, -50), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor='lightcoral' if pnl < 0 else 'lightgreen', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=10)
        
        # Add trade duration line
        if buy_idx is not None and sell_idx is not None:
            ax_main.plot([buy_idx, sell_idx], 
                       [buy_trade['price'], sell_trade['price']], 
                       'b--', linewidth=3, alpha=0.7, label='Trade Path')
        
        # Format axes
        ax_main.set_title(f"Trade #{buy_trade.get('strategy', 'Unknown')} Analysis\n"
                         f"Duration: {sell_time - buy_time} | "
                         f"P&L: ${sell_trade.get('pnl', 0):+.2f} ({sell_trade.get('pnl_percent', 0):+.1f}%)",
                         fontsize=14, fontweight='bold')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        ax_volume.set_title('Volume')
        ax_volume.grid(True, alpha=0.3)
        
        ax_indicators.set_title('RSI')
        ax_indicators.legend()
        ax_indicators.grid(True, alpha=0.3)
        
        # Set x-axis labels
        x_labels = [timestamp.strftime('%m/%d %H:%M') for timestamp in plot_df.index[::max(1, len(plot_df)//10)]]
        x_positions = list(range(0, len(plot_df), max(1, len(plot_df)//10)))
        ax_indicators.set_xticks(x_positions)
        ax_indicators.set_xticklabels(x_labels, rotation=45)
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(self.report_dir, "charts", filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _analyze_trade_performance(self, trade_df, buy_trade, sell_trade):
        """Analyze trade performance and market conditions"""
        if len(trade_df) < 2:
            return {"error": "Insufficient data for analysis"}
        
        buy_time = pd.to_datetime(buy_trade['time'])
        sell_time = pd.to_datetime(sell_trade['time'])
        buy_price = buy_trade['price']
        sell_price = sell_trade['price']
        
        # Find trade period in data
        if 'timestamp' in trade_df.columns:
            time_col = pd.to_datetime(trade_df['timestamp'])
        else:
            time_col = trade_df.index
        
        trade_mask = (time_col >= buy_time) & (time_col <= sell_time)
        trade_period_data = trade_df[trade_mask]
        
        if len(trade_period_data) > 0:
            max_price = trade_period_data['high'].max()
            min_price = trade_period_data['low'].min()
            
            # Calculate maximum favorable/adverse excursion
            max_favorable = ((max_price - buy_price) / buy_price) * 100
            max_adverse = ((min_price - buy_price) / buy_price) * 100
            
            # Volume analysis
            avg_volume_during = trade_period_data['volume'].mean()
        else:
            max_favorable = max_adverse = avg_volume_during = 0
            max_price = min_price = buy_price
        
        # Pre-trade volume analysis
        pre_trade_data = trade_df[time_col < buy_time]
        avg_volume_before = pre_trade_data['volume'].mean() if len(pre_trade_data) > 0 else avg_volume_during
        
        # Volatility analysis
        if len(trade_df) > 1:
            trade_df_copy = trade_df.copy()
            trade_df_copy['price_change'] = trade_df_copy['close'].pct_change() * 100
            avg_volatility = trade_df_copy['price_change'].std()
        else:
            avg_volatility = 0
        
        analysis = {
            'price_action': {
                'max_favorable_excursion': max_favorable,
                'max_adverse_excursion': max_adverse,
                'entry_quality': self._assess_entry_quality(max_favorable, max_adverse),
                'exit_quality': self._assess_exit_quality(sell_price, max_price, min_price)
            },
            'volume_analysis': {
                'avg_volume_before': avg_volume_before,
                'avg_volume_during': avg_volume_during,
                'volume_increase': ((avg_volume_during / avg_volume_before - 1) * 100) if avg_volume_before > 0 else 0
            },
            'market_conditions': {
                'volatility': avg_volatility,
                'volatility_category': self._categorize_volatility(avg_volatility)
            },
            'trade_quality': {
                'duration_category': self._categorize_duration(sell_time - buy_time),
                'size_category': self._categorize_trade_size(buy_trade.get('trade_amount', 0))
            }
        }
        
        return analysis
    
    def _assess_entry_quality(self, max_favorable, max_adverse):
        """Assess quality of trade entry"""
        if max_favorable > 5 and abs(max_adverse) < 2:
            return "Excellent - Quick profit, minimal drawdown"
        elif max_favorable > 2 and abs(max_adverse) < 5:
            return "Good - Positive movement with acceptable risk"
        elif abs(max_adverse) > 10:
            return "Poor - Significant adverse movement"
        else:
            return "Average - Mixed signals"
    
    def _assess_exit_quality(self, exit_price, max_price, min_price):
        """Assess quality of trade exit"""
        if max_price > min_price and max_price > 0 and min_price > 0:
            exit_ratio = (exit_price - min_price) / (max_price - min_price)
            if exit_ratio > 0.8:
                return "Excellent - Near peak exit"
            elif exit_ratio > 0.6:
                return "Good - Upper range exit"
            elif exit_ratio > 0.4:
                return "Average - Mid-range exit"
            else:
                return "Poor - Lower range exit"
        return "Unable to assess"
    
    def _categorize_volatility(self, volatility):
        """Categorize market volatility"""
        if volatility < 1:
            return "Low Volatility"
        elif volatility < 3:
            return "Normal Volatility"
        elif volatility < 5:
            return "High Volatility"
        else:
            return "Extreme Volatility"
    
    def _categorize_duration(self, duration):
        """Categorize trade duration"""
        hours = duration.total_seconds() / 3600
        if hours < 1:
            return "Scalp Trade (< 1 hour)"
        elif hours < 24:
            return "Intraday Trade"
        elif hours < 168:
            return "Short-term Trade (< 1 week)"
        else:
            return "Long-term Trade (> 1 week)"
    
    def _categorize_trade_size(self, amount):
        """Categorize trade size"""
        if amount < 100:
            return "Small Position"
        elif amount < 1000:
            return "Medium Position"
        elif amount < 10000:
            return "Large Position"
        else:
            return "Very Large Position"
    
    def _generate_summary_report(self, trade_pairs, all_trades):
        """Generate summary statistics for all trades"""
        total_trades = len(trade_pairs)
        if total_trades == 0:
            return {"error": "No completed trades found"}
        
        # Calculate overall statistics
        total_pnl = sum(sell_trade.get('pnl', 0) for pair_key, (strategy_name, buy_trade, sell_trade) in trade_pairs.items())
        winning_trades = sum(1 for pair_key, (strategy_name, buy_trade, sell_trade) in trade_pairs.items() if sell_trade.get('pnl', 0) > 0)
        win_rate = (winning_trades / total_trades) * 100
        
        # Strategy breakdown
        strategy_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
        for pair_key, (strategy_name, buy_trade, sell_trade) in trade_pairs.items():
            strategy_stats[strategy_name]['trades'] += 1
            strategy_stats[strategy_name]['pnl'] += sell_trade.get('pnl', 0)
            if sell_trade.get('pnl', 0) > 0:
                strategy_stats[strategy_name]['wins'] += 1
        
        # Calculate average trade duration
        durations = []
        for pair_key, (strategy_name, buy_trade, sell_trade) in trade_pairs.items():
            buy_time = pd.to_datetime(buy_trade['time'])
            sell_time = pd.to_datetime(sell_trade['time'])
            durations.append((sell_time - buy_time).total_seconds() / 3600)  # hours
        
        avg_duration_hours = np.mean(durations) if durations else 0
        
        summary = {
            'overall_stats': {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / total_trades,
                'win_rate': win_rate,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'avg_duration_hours': avg_duration_hours
            },
            'strategy_breakdown': dict(strategy_stats),
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def _create_master_html_report(self, summary, individual_reports, filename_prefix):
        """Create comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .summary {{ background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .trade-report {{ background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .trade-header {{ background-color: #3498db; color: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
                .metric-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; font-size: 14px; }}
                .profitable {{ color: #27ae60; }}
                .unprofitable {{ color: #e74c3c; }}
                .trade-details {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .detail-section {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                .analysis-section {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
                .reason-box {{ background-color: #e8f4f8; padding: 10px; border-left: 4px solid #3498db; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Multi-Strategy Trading Performance Report</h1>
                <p>Generated on: {summary['generation_time']}</p>
            </div>
        """
        
        # Add summary section
        if 'overall_stats' in summary:
            stats = summary['overall_stats']
            html_content += f"""
            <div class="summary">
                <h2>Overall Performance Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">{stats['total_trades']}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value {'profitable' if stats['total_pnl'] >= 0 else 'unprofitable'}">${stats['total_pnl']:+,.2f}</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{stats['win_rate']:.1f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${stats['avg_pnl_per_trade']:+,.2f}</div>
                        <div class="metric-label">Avg P&L/Trade</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{stats['avg_duration_hours']:.1f}h</div>
                        <div class="metric-label">Avg Duration</div>
                    </div>
                </div>
                
                <h3>Strategy Performance Breakdown</h3>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Trades</th>
                        <th>Total P&L</th>
                        <th>Win Rate</th>
                        <th>Avg P&L/Trade</th>
                    </tr>
            """
            
            for strategy, data in summary['strategy_breakdown'].items():
                win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
                avg_pnl = data['pnl'] / data['trades'] if data['trades'] > 0 else 0
                pnl_class = 'profitable' if data['pnl'] >= 0 else 'unprofitable'
                
                html_content += f"""
                    <tr>
                        <td><strong>{strategy}</strong></td>
                        <td>{data['trades']}</td>
                        <td class="{pnl_class}">${data['pnl']:+,.2f}</td>
                        <td>{win_rate:.1f}%</td>
                        <td class="{pnl_class}">${avg_pnl:+,.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # Add individual trade reports
        html_content += "<h2>Individual Trade Analysis</h2>"
        
        for report in individual_reports:
            pnl = report['performance']['pnl_dollar']
            pnl_class = 'profitable' if pnl >= 0 else 'unprofitable'
            
            html_content += f"""
            <div class="trade-report">
                <div class="trade-header">
                    <h3>Trade #{report['trade_id']}: {report['strategy']} Strategy</h3>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value {pnl_class}">${pnl:+,.2f}</div>
                        <div class="metric-label">Profit/Loss</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value {pnl_class}">{report['performance']['pnl_percent']:+.2f}%</div>
                        <div class="metric-label">Return %</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{report['performance']['duration']}</div>
                        <div class="metric-label">Duration</div>
                    </div>
                </div>
                
                <div class="trade-details">
                    <div class="detail-section">
                        <h4>BUY Details</h4>
                        <p><strong>Price:</strong> ${report['buy_details']['price']:.2f}</p>
                        <p><strong>Time:</strong> {report['buy_details']['time']}</p>
                        <p><strong>Amount:</strong> ${report['buy_details']['amount']:,.2f}</p>
                        <p><strong>Shares:</strong> {report['buy_details']['shares']:.6f}</p>
                        <div class="reason-box">
                            <strong>Buy Reason:</strong><br>
                            {report['buy_details']['reason']}
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h4>SELL Details</h4>
                        <p><strong>Price:</strong> ${report['sell_details']['price']:.2f}</p>
                        <p><strong>Time:</strong> {report['sell_details']['time']}</p>
                        <div class="reason-box">
                            <strong>Sell Reason:</strong><br>
                            {report['sell_details']['reason']}
                        </div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h4>Trade Chart Analysis</h4>
                    <img src="charts/{report['chart_filename']}" alt="Trade Chart" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                </div>
            """
            
            # Add analysis section if available
            if 'analysis' in report and 'price_action' in report['analysis']:
                analysis = report['analysis']
                html_content += f"""
                <div class="analysis-section">
                    <h4>Trade Analysis</h4>
                    <div class="trade-details">
                        <div class="detail-section">
                            <h5>Price Action Analysis</h5>
                            <p><strong>Max Favorable:</strong> +{analysis['price_action']['max_favorable_excursion']:.2f}%</p>
                            <p><strong>Max Adverse:</strong> {analysis['price_action']['max_adverse_excursion']:.2f}%</p>
                            <p><strong>Entry Quality:</strong> {analysis['price_action']['entry_quality']}</p>
                            <p><strong>Exit Quality:</strong> {analysis['price_action']['exit_quality']}</p>
                        </div>
                        
                        <div class="detail-section">
                            <h5>Market Conditions</h5>
                            <p><strong>Volatility:</strong> {analysis['market_conditions']['volatility']:.2f}% ({analysis['market_conditions']['volatility_category']})</p>
                            <p><strong>Duration Type:</strong> {analysis['trade_quality']['duration_category']}</p>
                            <p><strong>Position Size:</strong> {analysis['trade_quality']['size_category']}</p>
                            <p><strong>Volume Change:</strong> {analysis['volume_analysis']['volume_increase']:+.1f}%</p>
                        </div>
                    </div>
                </div>
                """
            
            html_content += "</div>"
        
        # Close HTML
        html_content += """
            </body>
            </html>
        """
        
        # Save HTML report
        report_filename = f"{self.report_dir}/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Master HTML report saved: {report_filename}")
        return report_filename

def plot_candles(df, trades=None, filename="chart.png"):
    """
    Enhanced multi-strategy candlestick chart with comprehensive trade reporting
    This function now generates detailed trade reports in addition to the main chart
    """
    os.makedirs("charts", exist_ok=True)
    chart_path = f"charts/{filename}"
    
    # Generate detailed trade reports if trades exist
    if trades and len(trades) > 0:
        print(f"Plotting {len(trades)} trades and generating detailed reports...")
        
        # Create the trade report generator
        report_generator = TradeReportGenerator()
        completed_trades = report_generator.generate_comprehensive_trade_report(df, trades, "trading_analysis")
        
        if completed_trades > 0:
            print(f"Generated comprehensive reports for {completed_trades} completed trades")
        else:
            print("No completed trade pairs found for detailed reporting")
    
    # Continue with existing chart generation logic
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_plot = df.set_index("timestamp")
    else:
        df_plot = df.copy()

    if trades and len(trades) > 0:
        # Group trades by strategy for visualization
        strategy_trades = defaultdict(list)
        for trade in trades:
            strategy_name = trade.get('strategy', 'Unknown')
            strategy_trades[strategy_name].append(trade)
        
        # Define strategy colors
        strategy_colors = {
            'Simple': {'buy': '#00FF00', 'sell': '#FF4444', 'line': '#00AA00'},
            'TrendFollowing': {'buy': '#0066FF', 'sell': '#FF6600', 'line': '#0044BB'}, 
            'MeanReversion': {'buy': '#FF00FF', 'sell': '#FFFF00', 'line': '#AA00AA'},
            'Breakout': {'buy': '#00FFFF', 'sell': '#FF8800', 'line': '#00AAAA'},
            'ASTA_TripleScreen': {'buy': '#32CD32', 'sell': '#DC143C', 'line': '#228B22'},
            'asta_triple_screen': {'buy': '#32CD32', 'sell': '#DC143C', 'line': '#228B22'}
        }
        
        # Create main visualization
        fig = plt.figure(figsize=(18, 12))
        ax1 = plt.subplot(2, 1, 1)
        ax_volume = plt.subplot(2, 1, 2)
        
        # Plot candlesticks and volume
        mpf.plot(df_plot, type="candle", ax=ax1, volume=ax_volume, 
                 style="yahoo", show_nontrading=False)
        
        # Add trade markers and connections
        total_pnl = 0
        trade_count = 0
        
        for strategy_name, strategy_trade_list in strategy_trades.items():
            colors = strategy_colors.get(strategy_name, 
                     {'buy': '#666666', 'sell': '#999999', 'line': '#777777'})
            
            buy_trades = [t for t in strategy_trade_list if t["action"] == "BUY"]
            sell_trades = [t for t in strategy_trade_list if t["action"] == "SELL"]
            
            # Plot markers
            if buy_trades:
                buy_times = [pd.to_datetime(t["time"]) for t in buy_trades]
                buy_prices = [t["price"] for t in buy_trades]
                ax1.scatter(buy_times, buy_prices, color=colors['buy'], marker="^", 
                           s=120, alpha=0.9, label=f"{strategy_name} BUY ({len(buy_trades)})",
                           edgecolors='black', linewidth=1)
            
            if sell_trades:
                sell_times = [pd.to_datetime(t["time"]) for t in sell_trades]
                sell_prices = [t["price"] for t in sell_trades]
                ax1.scatter(sell_times, sell_prices, color=colors['sell'], marker="v", 
                           s=120, alpha=0.9, label=f"{strategy_name} SELL ({len(sell_trades)})",
                           edgecolors='black', linewidth=1)
            
            # Connect trade pairs and show P&L
            for i, sell_trade in enumerate(sell_trades):
                if i < len(buy_trades):
                    buy_trade = buy_trades[i]
                    pnl = sell_trade.get("pnl", 0)
                    total_pnl += pnl
                    trade_count += 1
                    
                    buy_time = pd.to_datetime(buy_trade["time"])
                    sell_time = pd.to_datetime(sell_trade["time"])
                    
                    # Connection line
                    ax1.plot([buy_time, sell_time], [buy_trade["price"], sell_trade["price"]],
                            color=colors['line'], linestyle="--", linewidth=2, alpha=0.7)
                    
                    # P&L annotation
                    mid_time = buy_time + (sell_time - buy_time) / 2
                    mid_price = (buy_trade["price"] + sell_trade["price"]) / 2
                    
                    ax1.annotate(f"${pnl:+.0f}", (mid_time, mid_price),
                               xytext=(5, 5), textcoords="offset points",
                               bbox=dict(boxstyle="round,pad=0.3", 
                                        facecolor=colors['buy'] if pnl >= 0 else colors['sell'],
                                        alpha=0.8), fontsize=9, ha="center")
        
        # Chart formatting
        ax1.set_title(f"Multi-Strategy Trading Analysis\n"
                     f"Total P&L: ${total_pnl:+,.2f} | Completed Trades: {trade_count} | "
                     f"Strategies: {len(strategy_trades)}\n"
                     f"Detailed reports generated in 'trade_reports' directory", 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax_volume.set_title("Volume", fontsize=12)
        ax_volume.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Main trading chart saved: {chart_path}")
        print(f"Check 'trade_reports' directory for individual trade analysis")
        
    else:
        print("No trades to plot - creating basic candlestick chart...")
        mpf.plot(df_plot, type="candle", volume=True, style="yahoo", 
                savefig=chart_path, show_nontrading=False, figsize=(12, 8))
        print(f"Basic chart saved: {chart_path}")

def plot_strategy_comparison(trades, filename="strategy_comparison.png"):
    """Create detailed strategy comparison charts - enhanced version"""
    if not trades:
        print("No trades for strategy comparison")
        return
    
    os.makedirs("charts", exist_ok=True)
    
    # Group trades by strategy
    strategy_trades = defaultdict(list)
    for trade in trades:
        strategy_name = trade.get('strategy', 'Unknown')
        if trade['action'] == 'SELL' and 'pnl' in trade:
            strategy_trades[strategy_name].append(trade['pnl'])
    
    if not strategy_trades:
        print("No completed trades with P&L data")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    strategy_names = list(strategy_trades.keys())
    pnl_data = [strategy_trades[name] for name in strategy_names]
    
    # Enhanced visualizations with more detailed analysis
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    
    # 1. P&L Distribution (Box Plot)
    bp = ax1.boxplot(pnl_data, labels=strategy_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_title("P&L Distribution by Strategy", fontsize=14, fontweight='bold')
    ax1.set_ylabel("P&L ($)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
    
    # 2. Win Rate Analysis
    win_rates = []
    trade_counts = []
    for name in strategy_names:
        trades_pnl = strategy_trades[name]
        wins = sum(1 for pnl in trades_pnl if pnl > 0)
        win_rate = (wins / len(trades_pnl)) * 100 if trades_pnl else 0
        win_rates.append(win_rate)
        trade_counts.append(len(trades_pnl))
    
    bars = ax2.bar(strategy_names, win_rates, color=colors[:len(strategy_names)], alpha=0.8)
    ax2.set_title("Win Rate by Strategy", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    for bar, rate, count in zip(bars, win_rates, trade_counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{rate:.1f}%\n({count} trades)', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # 3. Average P&L per Trade
    avg_pnls = [sum(trades_pnl)/len(trades_pnl) if trades_pnl else 0 for trades_pnl in pnl_data]
    bars = ax3.bar(strategy_names, avg_pnls, color=colors[:len(strategy_names)], alpha=0.8)
    ax3.set_title("Average P&L per Trade", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Avg P&L ($)")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.7)
    
    for bar, avg in zip(bars, avg_pnls):
        ax3.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + (10 if avg >= 0 else -15),
                f'${avg:+.2f}', ha='center', 
                va='bottom' if avg >= 0 else 'top', 
                fontsize=11, fontweight='bold')
    
    # 4. Total P&L by Strategy
    total_pnls = [sum(trades_pnl) for trades_pnl in pnl_data]
    bars = ax4.bar(strategy_names, total_pnls, color=colors[:len(strategy_names)], alpha=0.8)
    ax4.set_title("Total P&L by Strategy", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Total P&L ($)")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='red', linestyle='--', alpha=0.7)
    
    for bar, total in zip(bars, total_pnls):
        ax4.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + (15 if total >= 0 else -25),
                f'${total:+.2f}', ha='center', 
                va='bottom' if total >= 0 else 'top', 
                fontsize=11, fontweight='bold')
    
    plt.suptitle("Enhanced Strategy Performance Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"charts/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced strategy comparison chart saved: charts/{filename}")