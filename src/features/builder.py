import pandas as pd
import numpy as np
from src.utils.config import load_config

class FeatureBuilder:
    def __init__(self, df):
        self.df = df.copy()
        self.config = load_config()

    def add_fx_features(self):
        # Log returns
        self.df["fx_returns"] = np.log(self.df["fx"] / self.df["fx"].shift(1))
        
        # Rolling Volatility (annualized)
        window = self.config["modeling"]["intervention"]["rolling_vol_window"]
        self.df["fx_vol"] = self.df["fx_returns"].rolling(window=window).std() * np.sqrt(252)
        
        # Volatility Suppression Proxy: Ratio of TRY vol to EM FX vol
        if "em_fx" in self.df.columns:
            self.df["em_fx_returns"] = np.log(self.df["em_fx"] / self.df["em_fx"].shift(1))
            self.df["em_fx_vol"] = self.df["em_fx_returns"].rolling(window=window).std() * np.sqrt(252)
            self.df["vol_suppression_ratio"] = self.df["fx_vol"] / self.df["em_fx_vol"]
        
        return self

    def add_macro_features(self):
        # Real Interest Rate proxy (if we have policy rate and cpi)
        # Note: CPI is usually monthly, policy rate daily. 
        # Here we assume they are aligned/ffilled in the loader.
        if "policy_rate" in self.df.columns and "cpi" in self.df.columns:
            # Simplistic real rate: policy_rate - cpi_inflation_yoy
            self.df["cpi_yoy"] = self.df["cpi"].pct_change(252) * 100
            self.df["real_rate"] = self.df["policy_rate"] - self.df["cpi_yoy"]

        # Reserve dynamics
        if "reserves" in self.df.columns:
            res_window = self.config["modeling"]["intervention"]["reserves_window"]
            self.df["reserve_change"] = self.df["reserves"].diff(res_window)
            self.df["reserve_burn_proxy"] = np.where(
                (self.df["reserve_change"] < 0) & (self.df["fx_returns"] <= 0.001), 
                -self.df["reserve_change"], 0
            )
            
        return self

    def add_trend_features(self):
        # Distance from rolling mean (Mean reversion proxy)
        self.df["ma_50"] = self.df["fx"].rolling(50).mean()
        self.df["ma_200"] = self.df["fx"].rolling(200).mean()
        self.df["dist_ma_200"] = (self.df["fx"] - self.df["ma_200"]) / self.df["ma_200"]
        
        # Drift - Linear trend estimation residuals
        # This helps identify periods where the currency deviates from its long-run path
        x = np.arange(len(self.df))
        y = np.log(self.df["fx"].values)
        # Mask NaNs
        mask = ~np.isnan(y)
        if mask.any():
            coeffs = np.polyfit(x[mask], y[mask], 1)
            self.df["log_trend"] = np.exp(coeffs[0] * x + coeffs[1])
            self.df["trend_deviation"] = (self.df["fx"] - self.df["log_trend"]) / self.df["log_trend"]

        return self

    def add_advanced_quant_features(self):
        # Distance to All-Time High (Breakout proxy)
        self.df["ath"] = self.df["fx"].cummax()
        self.df["dist_ath"] = (self.df["fx"] - self.df["ath"]) / self.df["ath"]
        
        # Volatility-Adjusted Returns (63-day rolling)
        window = 63
        rolling_ret = self.df["fx_returns"].rolling(window=window).mean()
        rolling_vol = self.df["fx_returns"].rolling(window=window).std()
        self.df["vol_adj_ret"] = (rolling_ret / rolling_vol).fillna(0)
        
        # Acceleration (Rate of change of returns)
        self.df["fx_accel"] = self.df["fx_returns"].diff()
        
        return self

    def build(self):
        self.add_fx_features()
        self.add_macro_features()
        self.add_trend_features()
        self.add_advanced_quant_features()
        return self.df.dropna()

if __name__ == "__main__":
    from src.data.loader import DataLoader
    loader = DataLoader()
    raw_data = loader.get_all_data()
    builder = FeatureBuilder(raw_data)
    features = builder.build()
    print(features.tail())
