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
            covariance_type="diag", # More robust to singular matrices
            n_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()

    def fit_predict(self, df, features=["fx_returns", "fx_vol"]):
        X = df[features].values
        # Filter out NaNs/Infs
        X = X[np.isfinite(X).all(axis=1)]
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled)
        regimes = self.model.predict(X_scaled)
        
        # Sort regimes by mean return
        means = self.model.means_[:, 0]
        idx = np.argsort(means)
        rank_map = {old: new for new, old in enumerate(idx)}
        regimes = np.array([rank_map[r] for r in regimes])
        
        return regimes

    def walk_forward_predict(self, df, features=["robust_returns", "fx_vol", "try_idiosyncratic_move"], train_window=252, step=21):
        """
        Fits and predicts in a walk-forward manner using robust features.
        """
        n = len(df)
        regimes = np.zeros(n)
        regimes[:train_window] = -1 
        
        for i in range(train_window, n, step):
            end_train = i
            start_train = max(0, i - 1260) 
            train_data = df.iloc[start_train:end_train][features].values
            
            # Critical: Remove NaNs and Infs
            mask = np.isfinite(train_data).all(axis=1)
            train_data = train_data[mask]
            
            if len(train_data) < 100:
                continue

            self.model.fit(train_data)
            
            predict_end = min(i + step, n)
            test_data = df.iloc[i:predict_end][features].values
            
            # Predict only on finite data
            test_mask = np.isfinite(test_data).all(axis=1)
            if test_mask.any():
                pred = self.model.predict(test_data[test_mask])
                
                means = self.model.means_[:, 0]
                idx = np.argsort(means)
                rank_map = {old: new for new, old in enumerate(idx)}
                
                full_pred = np.zeros(len(test_data))
                full_pred[test_mask] = np.array([rank_map[r] for r in pred])
                regimes[i:predict_end] = full_pred
            
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
