import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.regime_detector import RegimeDetector

def carry_risk_report(target_yield=0.40):
    loader = DataLoader()
    df = FeatureBuilder(loader.get_all_data()).build()
    
    detector = RegimeDetector()
    df['regime'] = detector.walk_forward_predict(df)
    
    # Analyze 'Regime 0' (Stability) durations
    df['is_stable'] = (df['regime'] == 0).astype(int)
    df['block'] = (df['is_stable'] != df['is_stable'].shift()).cumsum()
    stable_blocks = df[df['is_stable'] == 1].groupby('block').size()
    
    print("=== CARRY TRADE RISK ANALYSIS (40% Yield) ===")
    print(f"Average Stability Duration: {stable_blocks.mean():.1f} days")
    print(f"Median Stability Duration: {stable_blocks.median():.1f} days")
    print(f"Shortest Stability Window: {stable_blocks.min():.1f} days")
    
    # Calculate Monthly Breakeven
    monthly_cushion = target_yield / 12
    print(f"\nYour Monthly Interest Cushion: {monthly_cushion:.2%}")
    
    # Latest State
    latest = df.iloc[-1]
    current_regime = latest['regime']
    
    print(f"\nCURRENT STATUS (Latest Data):")
    print(f"Regime: {int(current_regime)} ({'STABLE' if current_regime==0 else 'CAUTION' if current_regime==1 else 'PANIC'})")
    
    if current_regime == 0:
        print("Recommendation: Strategy suggests HOLDING bonds, but watch Intervention Score.")
    else:
        print("Recommendation: Strategy suggests EXITING bonds. Devaluation risk exceeds carry.")

if __name__ == "__main__":
    carry_risk_report()
