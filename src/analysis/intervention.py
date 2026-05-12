import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class InterventionInference:
    def __init__(self, df):
        self.df = df.copy()

    def calculate_intervention_score(self):
        """
        Calculates a probability/intensity score of FX intervention.
        Higher score indicates higher likelihood of intervention.
        """
        # 1. Volatility Suppression (Low vol relative to EM peers)
        # Low vol ratio means high suppression
        if "vol_suppression_ratio" in self.df.columns:
            self.df["vol_score"] = 1 - self.df["vol_suppression_ratio"].rolling(21).rank(pct=True)
        else:
            self.df["vol_score"] = 0

        # 2. Reserve Burn (Negative reserve change when FX is stable)
        if "reserve_burn_proxy" in self.df.columns:
            # Normalize reserve burn proxy
            self.df["reserve_score"] = self.df["reserve_burn_proxy"].rolling(63).rank(pct=True)
        else:
            self.df["reserve_score"] = 0

        # 3. Trend Deviation (High deviation without volatility)
        # If we are far from long-run trend but volatility is low
        if "trend_deviation" in self.df.columns:
            self.df["trend_score"] = np.abs(self.df["trend_deviation"]).rolling(63).rank(pct=True)
        else:
            self.df["trend_score"] = 0

        # Composite Score
        self.df["intervention_intensity"] = (
            0.4 * self.df["vol_score"] + 
            0.4 * self.df["reserve_score"] + 
            0.2 * self.df["trend_score"]
        ).fillna(0)

        return self.df

    def identify_intervention_events(self, threshold=0.8):
        if "intervention_intensity" not in self.df.columns:
            self.calculate_intervention_score()
        
        return self.df[self.df["intervention_intensity"] > threshold]

if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.features.builder import FeatureBuilder
    loader = DataLoader()
    raw_data = loader.get_all_data()
    features = FeatureBuilder(raw_data).build()
    
    inference = InterventionInference(features)
    results = inference.calculate_intervention_score()
    print(results[["intervention_intensity"]].tail())
