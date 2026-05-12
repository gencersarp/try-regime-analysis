import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import matplotlib.pyplot as plt
from src.utils.config import load_config

class RegimeDetector:
    def __init__(self, n_components=3):
        self.config = load_config()
        self.n_components = n_components or self.config["modeling"]["hmm"]["n_components"]
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, 
            covariance_type=self.config["modeling"]["hmm"]["covariance_type"],
            n_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()

    def fit_predict(self, df, features=["fx_returns", "fx_vol"]):
        X = df[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled)
        regimes = self.model.predict(X_scaled)
        
        # Sort regimes by mean return
        means = self.model.means_[:, 0]
        idx = np.argsort(means)
        rank_map = {old: new for new, old in enumerate(idx)}
        regimes = np.array([rank_map[r] for r in regimes])
        
        return regimes

    def walk_forward_predict(self, df, features=["fx_returns", "fx_vol"], train_window=252, step=21):
        """
        Fits and predicts in a walk-forward manner to avoid lookahead bias.
        """
        n = len(df)
        regimes = np.zeros(n)
        regimes[:train_window] = -1 
        
        for i in range(train_window, n, step):
            end_train = i
            start_train = max(0, i - 1260) 
            train_data = df.iloc[start_train:end_train][features].values
            
            X_train_scaled = self.scaler.fit_transform(train_data)
            self.model.fit(X_train_scaled)
            
            predict_end = min(i + step, n)
            test_data = df.iloc[i:predict_end][features].values
            X_test_scaled = self.scaler.transform(test_data)
            
            pred = self.model.predict(X_test_scaled)
            
            means = self.model.means_[:, 0]
            idx = np.argsort(means)
            rank_map = {old: new for new, old in enumerate(idx)}
            regimes[i:predict_end] = np.array([rank_map[r] for r in pred])
            
        return regimes

    def detect_structural_breaks(self, df, target="fx_returns"):
        y = df[target].values
        X = np.ones((len(y), 1))
        stat, pval, _ = breaks_cusumolsresid(y, X)
        return stat, pval

    def get_regime_characteristics(self, df, regimes):
        df_temp = df.copy()
        df_temp["regime"] = regimes
        chars = df_temp[df_temp["regime"] != -1].groupby("regime").agg({
            "fx_returns": ["mean", "std"],
            "fx_vol": "mean",
            "vol_suppression_ratio": "mean" if "vol_suppression_ratio" in df.columns else "count"
        })
        return chars
