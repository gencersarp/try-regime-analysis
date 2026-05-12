import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

class QuantStats:
    @staticmethod
    def calculate_half_life(series):
        """
        Calculates the half-life of mean reversion for an OU process.
        """
        series_lag = series.shift(1)
        series_diff = series.diff()
        
        # Drop NaNs
        valid = ~(series_lag.isna() | series_diff.isna())
        if not valid.any():
            return np.inf
            
        model = OLS(series_diff[valid], sm.add_constant(series_lag[valid]))
        res = model.fit()
        lambda_val = res.params.iloc[1]
        
        if lambda_val >= 0:
            return np.inf # Not mean reverting
            
        half_life = -np.log(2) / lambda_val
        return half_life

    @staticmethod
    def test_stationarity(series):
        """ADF Test for stationarity."""
        res = adfuller(series.dropna())
        return {
            "ADF Statistic": res[0],
            "p-value": res[1],
            "Stationary": res[1] < 0.05
        }

    @staticmethod
    def calculate_rolling_beta(y, x, window=63):
        """Rolling beta of y with respect to x."""
        # Align series
        df = pd.concat([y, x], axis=1).dropna()
        df.columns = ["y", "x"]
        
        def get_beta(sub_df):
            if len(sub_df) < window: return np.nan
            model = OLS(sub_df["y"], sm.add_constant(sub_df["x"])).fit()
            return model.params.iloc[1]
            
        return df["y"].rolling(window=window).apply(lambda s: get_beta(df.loc[s.index]), raw=False)
