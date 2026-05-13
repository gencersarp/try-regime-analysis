import typer
import pandas as pd
from typing import Optional
from datetime import datetime
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.regime_detector import RegimeDetector
from src.models.hazard import RegimeHazardModel
from src.analysis.intervention import InterventionInference
from src.analysis.backtest import Backtester
from src.analysis.risk import RiskManager
from src.analysis.reporter import ResearchReporter
from src.visualization.plots import plot_regime_timeline, plot_intervention_analysis, plot_vol_suppression
import matplotlib.pyplot as plt
import os

from src.dashboard.engine import DashboardEngine

app = typer.Typer()

@app.command()
def fetch(as_of: Optional[str] = None):
    """Fetch raw data point-in-time."""
    loader = DataLoader()
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else datetime.now()
    data = loader.get_point_in_time_data(as_of_dt)
    typer.echo(f"Point-in-time data fetched for {as_of_dt.date()}. Shape: {data.shape}")

@app.command()
def dashboard(as_of: str = typer.Option(None, help="Snapshot date YYYY-MM-DD")):
    """Print the Carry Trade Risk Dashboard."""
    loader = DataLoader()
    engine = DashboardEngine()
    
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else datetime.now()
    
    # 1. Fetch raw data
    raw_data = loader.get_all_data()
    
    # 2. Build features (including macro and idiosyncratic moves)
    builder = FeatureBuilder(raw_data)
    df = builder.build()
    
    # 3. Run Dashboard
    results = engine.run(df, as_of_dt)
    engine.print_terminal_report(results)

@app.command()
def fetch_data(cache: bool = True):
    """Fetch raw data from all sources."""
    loader = DataLoader()
    data = loader.get_all_data(cache=cache)
    typer.echo(f"Data fetched. Shape: {data.shape}")

@app.command()
def analyze(output_dir: str = "reports", carry_rate: float = 0.45):
    """Run institutional-grade macro regime analysis."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(f"{output_dir}/figures"):
        os.makedirs(f"{output_dir}/figures")

    # 1. Load and Build Features
    typer.echo("Fetching live market data and EM peer basket...")
    loader = DataLoader()
    raw_data = loader.get_all_data(cache=False)
    builder = FeatureBuilder(raw_data)
    df = builder.build()
    
    # 2. Regime Detection (Robust Walk-forward)
    detector = RegimeDetector()
    typer.echo("Running robust walk-forward regime detection with Student-t normalization...")
    # Identification: Use REAL (inflation-adj) idiosyncratic moves
    # This addresses the 'inflation differential' gap
    features_to_use = ["robust_returns", "fx_vol", "try_idiosyncratic_move"]
    regimes = detector.walk_forward_predict(df, features=features_to_use)
    df["regime"] = regimes
    df_clean = df[df["regime"] != -1].copy()
    
    # 3. Regime Transition Hazard
    typer.echo("Estimating regime transition hazard (Probability of Breakdown)...")
    hazard = RegimeHazardModel(df_clean)
    # Include Core Reserves and Real Rates in Hazard identification
    hazard_features = ["try_idiosyncratic_move", "fx_vol", "robust_returns"]
    if hazard.fit(features=hazard_features):
        df_clean["breakdown_hazard"] = hazard.predict_hazard(df_clean, features=hazard_features)
    else:
        df_clean["breakdown_hazard"] = 0.0

    # 4. Intervention Inference (Reserve Flow Identity)
    inference = InterventionInference(df_clean)
    df_clean = inference.calculate_intervention_score()
    
    # 5. Backtesting (Dynamic Carry & Vol-Adjusted Spreads)
    typer.echo("Running strategy backtest (Dynamic Carry & Vol-Sensitive Spreads)...")
    backtester = Backtester(df_clean)
    # The backtester now uses 'policy_rate' automatically
    results = backtester.simulate_regime_strategy()
    metrics = backtester.get_metrics()
    
    # 6. Generate Reports
    typer.echo("Generating institutional reports...")
    
    fig1 = plot_regime_timeline(results, results["regime"], title="TRY Robust Regimes & Breakdown Hazard")
    fig1.savefig(f"{output_dir}/figures/regime_timeline.png")
    
    # Custom Hazard Plot
    plt.figure(figsize=(15, 5))
    plt.plot(results.index, results["breakdown_hazard"], color="orange", label="Breakdown Hazard (Prob)")
    plt.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    plt.title("Regime Transition Probability (Probability of Entering Panic State)")
    plt.legend()
    plt.savefig(f"{output_dir}/figures/breakdown_hazard.png")
    
    # Save results
    results.to_csv(f"{output_dir}/institutional_results.csv")
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        import json
        json.dump({"backtest": metrics}, f, indent=4)
    
    # Generate Research Report
    chars = detector.get_regime_characteristics(results, results["regime"])
    reporter = ResearchReporter(results, chars, f"{output_dir}/research_summary.md")
    report_path = reporter.generate_report()
    
    typer.echo(f"Analysis complete. Results saved to {output_dir}")
    typer.echo(f"Institutional Metrics: {metrics}")

if __name__ == "__main__":
    app()
