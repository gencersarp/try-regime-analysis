import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, df, initial_capital=100000, transaction_costs=0.0005):
        """
        initial_capital: Base currency amount.
        transaction_costs: 5 bps per trade (common for TRY spreads).
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.tc = transaction_costs

    def simulate_regime_strategy(self, signal_col="regime"):
        """
        DYNAMIC CARRY & VOL-SENSITIVE COSTS
        - Uses the 'policy_rate' time-series for carry calculation.
        - Spreads widen during high-volatility (Panic) regimes.
        """
        # Define positions
        self.df["position"] = 0
        self.df.loc[self.df[signal_col] == 2, "position"] = 1
        self.df.loc[self.df[signal_col] == 0, "position"] = -1
        self.df["position"] = self.df["position"].shift(1).fillna(0)
        
        # Calculate Returns
        self.df["strategy_returns"] = self.df["position"] * self.df["fx_returns"]
        
        # DYNAMIC CARRY: Use actual historical policy rates
        # daily_carry = (Annual Rate / 100) / 252
        if "policy_rate" in self.df.columns:
            self.df["daily_carry"] = (self.df["policy_rate"] / 100) / 252
            self.df["carry_returns"] = np.where(self.df["position"] == -1, self.df["daily_carry"], 0)
            self.df["carry_returns"] = np.where(self.df["position"] == 1, -self.df["daily_carry"], self.df["carry_returns"])
        else:
            self.df["carry_returns"] = 0
        
        # VOL-SENSITIVE TRANSACTION COSTS
        # Base TC is 5bps, but it scales with FX Volatility
        # During panics, vol can be 5x normal, so TC will be 25bps+
        self.df["dynamic_tc"] = self.tc * (1 + (self.df["fx_vol"] / self.df["fx_vol"].median()).fillna(1))
        self.df["trades"] = self.df["position"].diff().abs().fillna(0)
        self.df["costs"] = self.df["trades"] * self.df["dynamic_tc"]
        
        # Net Returns
        self.df["net_returns"] = self.df["strategy_returns"] + self.df["carry_returns"] - self.df["costs"]
        self.df["cum_returns"] = (1 + self.df["net_returns"]).cumprod()
        self.df["equity_curve"] = self.initial_capital * self.df["cum_returns"]
        
        return self.df

    def get_metrics(self):
        returns = self.df["net_returns"]
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        cum_ret = self.df["cum_returns"].iloc[-1] - 1
        
        # Drawdown
        equity = self.df["equity_curve"]
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "Total Return": f"{cum_ret:.2%}",
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Win Rate": f"{(returns > 0).mean():.2%}"
        }
