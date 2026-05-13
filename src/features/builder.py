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

    def add_macro_features(self, inflation_multiplier=1.2):
        # 1. REAL FX RETURNS (Inflation-adjusted)
        # USDTRY return minus (TR Inflation - US Inflation proxy)
        if "cpi" in self.df.columns:
            # Monthly CPI to Daily log-inflation
            self.df["tr_inflation_daily"] = np.log(self.df["cpi"] / self.df["cpi"].shift(1)).ffill() / 21
            self.df["real_fx_returns"] = self.df["fx_returns"] - self.df["tr_inflation_daily"]
        
        # 2. CORE RESERVES (Net Foreign Assets minus Swaps)
        # Note: In EVDS, Net Foreign Assets is TP.AB.A04. 
        # Total Swaps data is often TP.AB.A06 (approx) or external. 
        # Here we use a proxy if swap series is missing, otherwise direct.
        if "net_reserves" in self.df.columns:
            # Institutional Reality: Net Reserves are the only 'true' ammunition
            self.df["core_reserves"] = self.df["net_reserves"]
            self.df["reserve_adequacy_ratio"] = self.df["core_reserves"] / self.df["reserves"]

        if "policy_rate" in self.df.columns and "cpi" in self.df.columns:
            self.df["cpi_yoy"] = self.df["cpi"].pct_change(252) * 100
            self.df["adjusted_cpi_yoy"] = self.df["cpi_yoy"] * inflation_multiplier
            self.df["real_rate"] = self.df["policy_rate"] - self.df["adjusted_cpi_yoy"]
            self.df["credibility_gap"] = self.df["adjusted_cpi_yoy"] - self.df["policy_rate"]

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

    def add_causal_identification_features(self):
        """
        Implements peer-comparison (Synthetic Control) and latent reserve analysis.
        """
        # 1. EM Peer Alpha (Idiosyncratic TRY move)
        peer_cols = [c for c in self.df.columns if "peer_" in c]
        if peer_cols:
            peer_returns = np.log(self.df[peer_cols] / self.df[peer_cols].shift(1)).fillna(0)
            # Replace infinities from potential zero-price errors in some peer data
            peer_returns = peer_returns.replace([np.inf, -np.inf], 0)
            self.df["em_basket_returns"] = peer_returns.mean(axis=1)
            # TRY Alpha: Excess return of USDTRY over EM peer average
            self.df["try_idiosyncratic_move"] = (self.df["fx_returns"] - self.df["em_basket_returns"]).fillna(0)
        
        # 2. Latent Reserve Flow Identity
        if "current_account" in self.df.columns and "reserves" in self.df.columns:
            self.df["ca_proxy"] = self.df["current_account"].ffill().fillna(0) / 21
            self.df["latent_intervention"] = (self.df["reserves"].diff() - self.df["ca_proxy"]).fillna(0)
            self.df["shadow_fx_pressure"] = self.df["latent_intervention"].rolling(21).sum().fillna(0)

        return self

    def add_robust_scaling_features(self):
        # Student-t / Heavy Tail Robust features
        # Handle zero vol to avoid inf
        vol_safe = self.df["fx_vol"].shift(1).replace(0, 0.0001).fillna(0.0001)
        self.df["robust_returns"] = (self.df["fx_returns"] / vol_safe).fillna(0).replace([np.inf, -np.inf], 0)
        return self

    def build(self):
        self.add_fx_features()
        self.add_macro_features()
        self.add_trend_features()
        self.add_advanced_quant_features()
        self.add_causal_identification_features()
        self.add_robust_scaling_features()
        return self.df.dropna()

if __name__ == "__main__":
    from src.data.loader import DataLoader
    loader = DataLoader()
    raw_data = loader.get_all_data()
    builder = FeatureBuilder(raw_data)
    features = builder.build()
    print(features.tail())
