import os
import torch
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
import time

# Minimal GPT definition to load the checkpoint
# (In a real setup, we'd import this from train.py)
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"

# Note: The actual backtest logic depends on how we want the model 
# to "produce a strategy". For this prototype, we'll assume the model
# is asked to predict price direction (Up/Down) for a given sequence of prices.

def get_market_data(tickers=["SPY", "QQQ", "BTC-USD"], period="2y"):
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, period=period)['Close']
    return data

def calculate_metrics(returns):
    cagr = (returns + 1).prod() ** (252 / len(returns)) - 1
    vol = returns.std() * (252 ** 0.5)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_drawdown
    }

def run_backtest():
    if not os.path.exists('model_latest.pt'):
        print("No model found. Skipping backtest.")
        return
    
    checkpoint = torch.load('model_latest.pt', map_location='cpu')
    print(f"Loaded model from step {checkpoint['step']} with val_bpb {checkpoint['val_bpb']:.4f}")
    
    # Placeholder for actual signal generation logic
    # In the future, we will feed the model price sequences 
    # and extract its 'proposal' or 'signal'.
    
    data = get_market_data()
    
    # MOCK STRATEGY: 
    # We will replace this with the model's actual inference.
    # For now, it's just a placeholder to show the results collection.
    results = []
    for ticker in data.columns:
        rets = data[ticker].pct_change().dropna()
        # Mock signal: random but reproducible for the example
        np.random.seed(42)
        signals = np.random.choice([-1, 0, 1], size=len(rets)) 
        strat_rets = signals * rets
        metrics = calculate_metrics(strat_rets)
        metrics['Ticker'] = ticker
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['Sharpe'].mean()
    
    # Save backtest results
    results_file = 'backtest_results.tsv'
    header = not os.path.exists(results_file)
    
    with open(results_file, 'a') as f:
        if header:
            f.write("timestamp	step	val_bpb	avg_sharpe	metrics_json
")
        
        timestamp = int(time.time())
        metrics_json = df_results.to_json(orient='records')
        f.write(f"{timestamp}	{checkpoint['step']}	{checkpoint['val_bpb']:.6f}	{avg_sharpe:.4f}	{metrics_json}
")
    
    print(f"Backtest complete. Avg Sharpe: {avg_sharpe:.4f}")
    print(f"Results appended to {results_file}")

if __name__ == "__main__":
    try:
        run_backtest()
    except Exception as e:
        print(f"Backtest failed: {e}")
