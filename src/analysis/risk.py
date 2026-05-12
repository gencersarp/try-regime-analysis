import numpy as np
import scipy.stats as stats

class RiskManager:
    def __init__(self, returns):
        self.returns = returns.dropna()

    def calculate_var(self, confidence_level=0.95):
        """Historical Value at Risk."""
        return np.percentile(self.returns, (1 - confidence_level) * 100)

    def calculate_expected_shortfall(self, confidence_level=0.95):
        """Conditional VaR."""
        var = self.calculate_var(confidence_level)
        return self.returns[self.returns <= var].mean()

    def calculate_regime_risk(self, df, regime_col="regime"):
        """Calculates risk metrics per regime."""
        risk_metrics = {}
        for r in df[regime_col].unique():
            r_returns = df[df[regime_col] == r]["net_returns"]
            if len(r_returns) > 20:
                risk_metrics[f"Regime {r}"] = {
                    "Daily VaR (95%)": f"{self.calculate_var_for_series(r_returns):.2%}",
                    "Daily ES (95%)": f"{self.calculate_es_for_series(r_returns):.2%}",
                    "Volatility": f"{r_returns.std() * np.sqrt(252):.2%}"
                }
        return risk_metrics

    @staticmethod
    def calculate_var_for_series(series, confidence_level=0.95):
        return np.percentile(series, (1 - confidence_level) * 100)

    @staticmethod
    def calculate_es_for_series(series, confidence_level=0.95):
        var = np.percentile(series, (1 - confidence_level) * 100)
        return series[series <= var].mean()
