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

    def simulate_regime_strategy(self, signal_col="regime", carry_rate=0.0):
        """
        Simple strategy: 
        - Regime 0 (Stability): Carry trade (Short USDTRY / Long TRY)
        - Regime 1 (Moderate Dep): Neutral/Hedged
        - Regime 2 (Panic): Long USDTRY
        """
        # Define positions: 1 for Long USDTRY, -1 for Short USDTRY, 0 for Cash
        self.df["position"] = 0
        self.df.loc[self.df[signal_col] == 2, "position"] = 1
        self.df.loc[self.df[signal_col] == 0, "position"] = -1
        
        # Lag signals to prevent lookahead bias (execute at T+1)
        self.df["position"] = self.df["position"].shift(1).fillna(0)
        
        # Calculate returns
        self.df["strategy_returns"] = self.df["position"] * self.df["fx_returns"]
        
        # Add Carry (Daily rate)
        # In Regime 0 (Stability), we earn TRY interest (Simplified)
        # In Regime 2 (Panic), we pay TRY interest to be Long USD
        daily_carry = carry_rate / 252
        self.df["carry_returns"] = np.where(self.df["position"] == -1, daily_carry, 0)
        self.df["carry_returns"] = np.where(self.df["position"] == 1, -daily_carry, self.df["carry_returns"])
        
        # Transaction Costs
        self.df["trades"] = self.df["position"].diff().abs().fillna(0)
        self.df["costs"] = self.df["trades"] * self.tc
        
        # Net Returns
        self.df["net_returns"] = self.df["strategy_returns"] + self.df["carry_returns"] - self.df["costs"]
        
        # Performance
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
