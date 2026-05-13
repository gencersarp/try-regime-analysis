import pandas as pd
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder

def check_live_market_conditions():
    loader = DataLoader()
    # Fetch fresh data (bypass cache)
    df_raw = loader.get_all_data(cache=False)
    
    # Get latest values
    print(f"DEBUG: Available columns in df_raw: {df_raw.columns.tolist()}")
    
    latest_fx = df_raw['fx'].iloc[-1]
    
    latest_policy_rate = 0
    if 'policy_rate' in df_raw.columns:
        latest_policy_rate = df_raw['policy_rate'].iloc[-1]
    else:
        # Fallback to manual if API mapping failed for a specific run
        print("Warning: 'policy_rate' not found in columns. Using estimated 50.0%.")
        latest_policy_rate = 50.0
    
    # Calculate YoY Inflation from TUIK data
    latest_cpi_yoy = 0
    if 'cpi' in df_raw.columns:
        cpi_series = df_raw['cpi'].dropna()
        if len(cpi_series) > 12:
            latest_cpi_yoy = (cpi_series.iloc[-1] / cpi_series.iloc[-12] - 1) * 100
    
    if latest_cpi_yoy == 0:
        print("Warning: 'cpi' data incomplete. Using official target proxy 65.0%.")
        latest_cpi_yoy = 65.0
    
    print("=== LIVE MARKET DATA VERIFICATION ===")
    print(f"Current USDTRY Rate: {latest_fx:.4f}")
    print(f"CBRT Policy Rate (Official): {latest_policy_rate:.2f}%")
    print(f"TUIK Annual Inflation (YoY): {latest_cpi_yoy:.2f}%")
    
    # Real Rate Calculation
    official_real_rate = latest_policy_rate - latest_cpi_yoy
    print(f"Official Real Rate: {official_real_rate:.2f}%")
    
    # Stress Test (ENAG Proxy)
    stress_inflation = latest_cpi_yoy * 1.5
    stress_real_rate = latest_policy_rate - stress_inflation
    print(f"Stressed Real Rate (1.5x Inflation): {stress_real_rate:.2f}%")
    
    # Carry Logic
    print("\n=== BOND TRADE ANALYSIS ===")
    user_bond_yield = 40.0
    monthly_yield = user_bond_yield / 12
    print(f"Your Bond Yield: {user_bond_yield:.2f}%")
    print(f"Monthly Carry Cushion: {monthly_yield:.2f}%")
    
    # Check 30-day USDTRY depreciation
    fx_30d_ago = df_raw['fx'].iloc[-30]
    depreciation_30d = (latest_fx / fx_30d_ago - 1) * 100
    
    print(f"USDTRY Depreciation (Last 30 days): {depreciation_30d:.2f}%")
    
    if depreciation_30d > monthly_yield:
        print("\nCRITICAL: The Lira is devaluing faster than your bond interest is paying.")
        print("You are currently LOSING money in USD terms.")
    else:
        print("\nSTATUS: Your bond interest is currently covering the depreciation.")

if __name__ == "__main__":
    check_live_market_conditions()
