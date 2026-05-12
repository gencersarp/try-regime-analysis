import pandas as pd
from datetime import datetime

class ResearchReporter:
    def __init__(self, df, regime_chars, output_path="reports/research_summary.md"):
        self.df = df
        self.regime_chars = regime_chars
        self.output_path = output_path

    def generate_report(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report = f"""# TRY Regime Analysis Research Report
Generated on: {now}

## Executive Summary
This report analyzes the Turkish Lira (TRY) FX dynamics, focusing on regime detection and intervention inference.

## Regime Characteristics
The Hidden Markov Model identified {len(self.regime_chars)} distinct regimes:

| Regime | Mean Return | Volatility | Avg Intervention Score |
|--------|-------------|------------|------------------------|
"""
        for i, row in self.regime_chars.iterrows():
            mean_ret = row[("fx_returns", "mean")] * 100 # percentage
            vol = row[("fx_vol", "mean")]
            int_score = self.df[self.df["regime"] == i]["intervention_intensity"].mean()
            report += f"| {i} | {mean_ret:.4f}% | {vol:.4f} | {int_score:.4f} |\n"

        report += """
## Intervention Analysis
- **Intervention Intensity Threshold (0.8):**
"""
        high_int_days = len(self.df[self.df["intervention_intensity"] > 0.8])
        pct_high_int = (high_int_days / len(self.df)) * 100
        
        report += f"- Identified {high_int_days} days of high intervention probability ({pct_high_int:.2f}% of data).\n"
        
        # Latest State
        latest = self.df.iloc[-1]
        report += f"""
## Current Market State (Latest Data)
- **Date:** {self.df.index[-1].strftime("%Y-%m-%d")}
- **Current Regime:** {latest['regime']}
- **Intervention Score:** {latest['intervention_intensity']:.4f}
- **Volatility:** {latest['fx_vol']:.4f}
- **Trend Deviation:** {latest['trend_deviation']:.4f}

## Figures
- [Regime Timeline](./figures/regime_timeline.png)
- [Intervention Analysis](./figures/intervention_analysis.png)
- [Volatility Suppression](./figures/vol_suppression.png)
"""
        with open(self.output_path, "w") as f:
            f.write(report)
        
        return self.output_path
