import pytest
import pandas as pd
import numpy as np
from src.features.builder import FeatureBuilder
from src.models.regime_detector import RegimeDetector

def test_feature_builder():
    df = pd.DataFrame({
        "fx": [10, 10.1, 10.2, 10.3, 10.4],
        "em_fx": [1.1, 1.2, 1.3, 1.4, 1.5],
        "reserves": [100, 95, 90, 85, 80],
        "policy_rate": [15, 15, 15, 15, 15],
        "cpi": [100, 101, 102, 103, 104]
    }, index=pd.date_range("2020-01-01", periods=5))
    
    builder = FeatureBuilder(df)
    features = builder.build()
    
    assert "fx_returns" in features.columns
    assert "fx_vol" in features.columns
    assert "reserve_change" in features.columns

def test_regime_detector():
    # Generate synthetic data for 2 regimes
    np.random.seed(42)
    r1 = np.random.normal(0, 0.01, 100)
    r2 = np.random.normal(0.05, 0.02, 100)
    v1 = np.random.normal(0.1, 0.01, 100)
    v2 = np.random.normal(0.5, 0.05, 100)
    
    df = pd.DataFrame({
        "fx_returns": np.concatenate([r1, r2]),
        "fx_vol": np.concatenate([v1, v2])
    })
    
    detector = RegimeDetector(n_components=2)
    regimes = detector.fit_predict(df)
    
    assert len(np.unique(regimes)) == 2
    # Regime 1 should have higher mean return
    chars = detector.get_regime_characteristics(df, regimes)
    assert chars.loc[1, ("fx_returns", "mean")] > chars.loc[0, ("fx_returns", "mean")]
