import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class RegimeHazardModel:
    def __init__(self, df, regime_col="regime", target_regime=2):
        """
        Estimates the hazard (probability) of transitioning into a target regime (e.g., Panic).
        """
        self.df = df.copy()
        self.regime_col = regime_col
        self.target_regime = target_regime
        self.scaler = StandardScaler()
        self.model = LogisticRegression(class_weight='balanced')

    def prepare_features(self, features=["shadow_fx_pressure", "try_idiosyncratic_move", "fx_vol", "credibility_gap"]):
        # Shift target by 5 days (predicting transition in the next week)
        self.df["target_transition"] = (self.df[self.regime_col].shift(-5) == self.target_regime).astype(int)
        
        # Only train on periods where we are NOT currently in the target regime
        train_mask = self.df[self.regime_col] != self.target_regime
        X = self.df.loc[train_mask, features].dropna()
        y = self.df.loc[X.index, "target_transition"]
        
        return X, y

    def fit(self, features=["shadow_fx_pressure", "try_idiosyncratic_move", "fx_vol", "credibility_gap"]):
        X, y = self.prepare_features(features)
        if len(np.unique(y)) < 2:
            print("Insufficient transition events for hazard modeling.")
            return False
            
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return True

    def predict_hazard(self, current_df, features=["shadow_fx_pressure", "try_idiosyncratic_move", "fx_vol", "credibility_gap"]):
        X = current_df[features].fillna(method='ffill').fillna(0)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return probs
