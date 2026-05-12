import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_regime_timeline(df, regimes, title="TRY FX Regimes"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # FX Price
    ax1.plot(df.index, df["fx"], color="black", alpha=0.7)
    ax1.set_title(title)
    ax1.set_ylabel("USDTRY")
    
    # Color background by regime
    # Handle both series and arrays
    regime_values = regimes.values if hasattr(regimes, "values") else regimes
    unique_regimes = np.unique(regime_values)
    
    # Filter out -1 (Training period) if present
    unique_regimes = unique_regimes[unique_regimes >= 0]
    
    colors = sns.color_palette("viridis", len(unique_regimes))
    for i, r in enumerate(unique_regimes):
        mask = regime_values == r
        ax1.fill_between(df.index, df["fx"].min(), df["fx"].max(), where=mask, color=colors[i], alpha=0.3, label=f"Regime {int(r)}")
    
    # Returns/Volatility
    ax2.plot(df.index, df["fx_returns"], color="gray", alpha=0.5, label="Returns")
    ax2.set_ylabel("Log Returns")
    
    plt.legend()
    plt.tight_layout()
    return fig

def plot_intervention_analysis(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Intervention Score
    ax1.plot(df.index, df["intervention_intensity"], color="red", label="Intervention Intensity")
    ax1.axhline(0.8, color="black", linestyle="--", alpha=0.5)
    ax1.set_title("Inferred Intervention Intensity")
    ax1.set_ylabel("Score")
    
    # Reserves vs Price
    ax2.plot(df.index, df["fx"], color="blue", label="USDTRY")
    ax2_twin = ax2.twinx()
    if "reserves" in df.columns:
        ax2_twin.plot(df.index, df["reserves"], color="green", alpha=0.5, label="Gross Reserves")
    ax2_twin.set_ylabel("Reserves")
    
    plt.legend()
    plt.tight_layout()
    return fig

def plot_vol_suppression(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["fx_vol"], label="TRY Volatility")
    if "em_fx_vol" in df.columns:
        plt.plot(df.index, df["em_fx_vol"], label="EM FX Volatility", alpha=0.7)
    plt.title("Volatility Suppression Analysis")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()
