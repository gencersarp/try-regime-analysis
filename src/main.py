import typer
import pandas as pd
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.regime_detector import RegimeDetector
from src.analysis.intervention import InterventionInference
from src.visualization.plots import plot_regime_timeline, plot_intervention_analysis, plot_vol_suppression
import matplotlib.pyplot as plt
import os

from src.analysis.reporter import ResearchReporter

app = typer.Typer()

@app.command()
def fetch_data(cache: bool = True):
    """Fetch raw data from all sources."""
    loader = DataLoader()
    data = loader.get_all_data(cache=cache)
    typer.echo(f"Data fetched. Shape: {data.shape}")

from src.analysis.backtest import Backtester
from src.analysis.risk import RiskManager

@app.command()
def analyze(output_dir: str = "reports", carry_rate: float = 0.20):
    """Run full regime, intervention, and backtest analysis."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(f"{output_dir}/figures"):
        os.makedirs(f"{output_dir}/figures")

    # 1. Load and Build Features
    loader = DataLoader()
    raw_data = loader.get_all_data()
    builder = FeatureBuilder(raw_data)
    df = builder.build()
    
    # 2. Regime Detection (Walk-forward)
    detector = RegimeDetector()
    typer.echo("Running walk-forward regime detection with advanced features...")
    # Using Returns, Volatility, Vol-Adjusted Returns, and Acceleration
    features_to_use = ["fx_returns", "fx_vol", "vol_adj_ret", "fx_accel"]
    regimes = detector.walk_forward_predict(df, features=features_to_use)
    df["regime"] = regimes
    df_clean = df[df["regime"] != -1].copy()
    
    # 3. Intervention Inference
    inference = InterventionInference(df_clean)
    df_clean = inference.calculate_intervention_score()
    
    # 4. Backtesting
    typer.echo("Running strategy backtest...")
    backtester = Backtester(df_clean)
    results = backtester.simulate_regime_strategy(carry_rate=carry_rate)
    metrics = backtester.get_metrics()
    
    # 5. Risk Management
    risk = RiskManager(results["net_returns"])
    risk_profile = risk.calculate_regime_risk(results)
    
    # 6. Generate Reports
    typer.echo("Generating plots and reports...")
    
    fig1 = plot_regime_timeline(results, results["regime"])
    fig1.savefig(f"{output_dir}/figures/regime_timeline.png")
    
    fig2 = plot_intervention_analysis(results)
    fig2.savefig(f"{output_dir}/figures/intervention_analysis.png")
    
    # Save results
    results.to_csv(f"{output_dir}/processed_results.csv")
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        import json
        json.dump({"backtest": metrics, "risk": risk_profile}, f, indent=4)
    
    # Generate Research Report
    chars = detector.get_regime_characteristics(results, results["regime"])
    reporter = ResearchReporter(results, chars, f"{output_dir}/research_summary.md")
    report_path = reporter.generate_report()
    
    typer.echo(f"Analysis complete. Results saved to {output_dir}")
    typer.echo(f"Backtest Stats: {metrics}")

if __name__ == "__main__":
    app()
