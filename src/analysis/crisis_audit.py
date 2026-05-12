import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.regime_detector import RegimeDetector
from src.analysis.backtest import Backtester

def run_historical_deep_dive():
    # 1. Load Data
    loader = DataLoader()
    df_raw = loader.get_all_data()
    df = FeatureBuilder(df_raw).build()
    
    # 2. Walk-forward Prediction (No lookahead)
    detector = RegimeDetector()
    df['regime'] = detector.walk_forward_predict(df, train_window=252*2, step=21)
    
    # 3. Backtest
    backtester = Backtester(df)
    results = backtester.simulate_regime_strategy(carry_rate=0.20)
    
    # 4. Crisis Analysis Windows
    crises = {
        "2018 Lira Crisis": ("2018-01-01", "2018-12-31"),
        "2021 Currency Collapse": ("2021-06-01", "2022-03-31"),
        "2023 Post-Election": ("2023-01-01", "2023-12-31")
    }
    
    print("=== HISTORICAL CRISIS PREDICTION AUDIT ===\n")
    
    for name, (start, end) in crises.items():
        window = results.loc[start:end]
        if window.empty: continue
        
        # Check first date of 'Panic' (Regime 2)
        panic_dates = window[window['regime'] == 2].index
        
        # Calculate return during window
        win_return = (window['equity_curve'].iloc[-1] / window['equity_curve'].iloc[0]) - 1
        bh_return = (window['fx'].iloc[-1] / window['fx'].iloc[0]) - 1
        
        print(f"CRISIS: {name}")
        print(f"Window: {start} to {end}")
        if not panic_dates.empty:
            print(f"First 'Panic' Signal Detected: {panic_dates[0].strftime('%Y-%m-%d')}")
        else:
            print("No 'Panic' Signal detected in this window.")
            
        print(f"Strategy Return: {win_return:.2%}")
        print(f"Buy & Hold USD Return: {bh_return:.2%}")
        print("-" * 30)

if __name__ == "__main__":
    run_historical_deep_dive()
