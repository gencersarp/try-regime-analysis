from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import yaml
from src.dashboard.gauges import Gauge, REERGauge, NetReservesGauge, SwapRateGauge, CDSGauge, VolRegimeGauge
from src.data.models import GaugeBand

class DashboardEngine:
    def __init__(self, config_path: str = "configs/main_config.yaml", events_path: str = "configs/event_calendar.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(events_path, "r") as f:
            self.events = yaml.safe_load(f)
            
        self.gauges: List[Gauge] = [
            REERGauge("REER Overvaluation", "BIS broad REER z-score", False),
            NetReservesGauge("Net Reserves ex-swaps", "CBRT Net Foreign Assets minus Swaps", True),
            SwapRateGauge("Offshore Swap Spread", "1W offshore swap rate spread", False),
            CDSGauge("5Y CDS", "Turkey 5-year CDS level", False),
            VolRegimeGauge("Realized Volatility", "21-day annualized USDTRY vol z-score", False)
        ]

    def _is_within_event_window(self, as_of: datetime, window_days: int = 5) -> bool:
        for event in self.events.get("events", []):
            event_date = pd.to_datetime(event["date"])
            diff = (event_date - as_of).days
            if 0 <= diff <= window_days:
                return True
        return False

    def run(self, df: pd.DataFrame, as_of: datetime) -> Dict[str, Any]:
        results = []
        counts = {GaugeBand.GREEN: 0, GaugeBand.AMBER: 0, GaugeBand.RED: 0}
        
        for gauge in self.gauges:
            val = gauge.compute(df, as_of)
            band = gauge.band() if not isinstance(gauge, VolRegimeGauge) else gauge.band_fixed(val)
            results.append({
                "name": gauge.name,
                "value": val,
                "band": band,
                "explanation": gauge.explain(),
                "timestamp": gauge.last_timestamp
            })
            counts[band] += 1

        # Composite Rule
        within_event = self._is_within_event_window(as_of)
        if counts[GaugeBand.RED] > 0:
            composite = GaugeBand.RED
        elif counts[GaugeBand.AMBER] >= 2:
            composite = GaugeBand.AMBER
        elif within_event:
            composite = GaugeBand.AMBER
        else:
            composite = GaugeBand.GREEN

        # Sizing Recommendation
        sizing = "100% of max carry allocation"
        if composite == GaugeBand.AMBER:
            sizing = "50%, stops tightened to 1.5x ATR"
        elif composite == GaugeBand.RED:
            sizing = "0%, do not re-enter until composite GREEN for 10 days"

        return {
            "as_of": as_of,
            "gauges": results,
            "composite": composite,
            "event_warning": within_event,
            "recommendation": sizing
        }

    def print_terminal_report(self, results: Dict[str, Any]):
        print(f"\n--- TRY CARRY TRADE RISK DASHBOARD (As of {results['as_of'].strftime('%Y-%m-%d')}) ---")
        print(f"COMPOSITE STATUS: {results['composite'].upper()}")
        print(f"Event Window Warning: {'YES' if results['event_warning'] else 'NO'}")
        print(f"Position Sizing: {results['recommendation']}")
        print("-" * 60)
        for g in results["gauges"]:
            color = "\033[92m" if g["band"] == GaugeBand.GREEN else "\033[93m" if g["band"] == GaugeBand.AMBER else "\033[91m"
            reset = "\033[0m"
            print(f"{g['name']}: {color}{g['band'].upper()}{reset} (Value: {g['value']:.2f})")
            print(f"  └ {g['explanation']}")
        print("-" * 60)
        print("Disclaimer: Final sizing is the user's decision. This is one input.")
