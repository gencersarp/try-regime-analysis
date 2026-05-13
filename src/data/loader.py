import os
import pandas as pd
import numpy as np
import yfinance as yf
from evds import evdsAPI
import pandas_datareader.data as web
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pyarrow.parquet as pq
import pyarrow as pa
from src.utils.config import load_config, setup_env
from src.data.models import SeriesMetadata, DataSource, DataFrequency, DataPoint, SeriesSnapshot

class DataLoader:
    def __init__(self):
        self.config = load_config()
        self.env = setup_env()
        self.evds = None
        if self.env.get("EVDS_API_KEY"):
            self.evds = evdsAPI(self.env["EVDS_API_KEY"])
        
        self.raw_dir = "data/raw"
        os.makedirs(self.raw_dir, exist_ok=True)

    def _get_cache_path(self, source: DataSource, symbol: str) -> str:
        return f"{self.raw_dir}/{source.value}/{symbol.replace('.', '_')}.parquet"

    def _apply_redenom_fix(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Handles the 2005 TRY redenomination (1,000,000 to 1).
        If date < 2005-01-01 and value > 100,000, divide by 1,000,000.
        """
        cutoff = pd.Timestamp("2005-01-01")
        mask = (df.index < cutoff) & (df[col] > 10000)
        df.loc[mask, col] = df.loc[mask, col] / 1000000
        return df

    def fetch_yfinance_series(self, name: str, symbol: str, as_of: Optional[datetime] = None) -> pd.Series:
        cache_path = self._get_cache_path(DataSource.YAHOO, symbol)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        start = self.config["data"]["start_date"]
        
        print(f"Fetching {name} ({symbol}) from Yahoo...")
        df = yf.download(symbol, start=start, progress=False)
        
        if df.empty:
            return pd.Series(name=name)

        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.get_level_values(0):
                res = df['Adj Close']
            else:
                res = df['Close']
            series = res.iloc[:, 0]
        else:
            series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        
        series.name = name
        
        # Redenom fix for FX
        if name == "fx":
            df_temp = pd.DataFrame(series)
            df_temp = self._apply_redenom_fix(df_temp, "fx")
            series = df_temp["fx"]

        # Cache to parquet
        series.to_frame().to_parquet(cache_path)
        
        if as_of:
            series = series[series.index <= as_of]
            
        return series

    def fetch_evds_series(self, as_of: Optional[datetime] = None) -> pd.DataFrame:
        if not self.evds:
            print("EVDS API Key missing. Skipping.")
            return pd.DataFrame()
            
        series_map = self.config["data"]["evds_series"]
        start = self.config["data"]["start_date"]
        start_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%d-%m-%Y")
        
        codes = list(series_map.values())
        print(f"Fetching EVDS codes: {codes}...")
        
        try:
            df = self.evds.get_data(codes, startdate=start_fmt)
            if df is None or df.empty:
                return pd.DataFrame()
            
            df["Tarih"] = pd.to_datetime(df["Tarih"], format="mixed")
            df.set_index("Tarih", inplace=True)
            
            # Map codes to names
            inv_map = {v: k for k, v in series_map.items()}
            new_cols = []
            for col in df.columns:
                mapped = False
                col_lookup = col.upper().replace("_", ".")
                for code, name in inv_map.items():
                    if code.upper() == col_lookup or code.upper().replace(".", "_") == col.upper():
                        new_cols.append(name)
                        mapped = True
                        break
                if not mapped:
                    new_cols.append(col)
            df.columns = new_cols
            
            # Apply point-in-time lags
            lags = self.config["data"]["publication_lags"]
            for col in df.columns:
                if col in lags:
                    # Shift forward by lag to represent when it was available
                    # Actually, for point-in-time, we want: if I am at date T,
                    # I can only see data with timestamp <= T - lag.
                    # In a backtest, when looking at index T, we should only see val from T - lag.
                    df[f"{col}_available_at"] = df.index + pd.Timedelta(days=lags[col])
            
            if as_of:
                # Filter rows where available_at <= as_of
                # This is tricky for a combined DF. 
                # Better to keep the raw and handle lag in feature building or a specific PIT getter.
                pass
                
            return df
        except Exception as e:
            print(f"Error fetching EVDS: {e}")
            return pd.DataFrame()

    def fetch_fred_series(self, as_of: Optional[datetime] = None) -> pd.DataFrame:
        if not self.env.get("FRED_API_KEY"):
            return pd.DataFrame()
            
        series_map = self.config["data"]["fred_series"]
        start = self.config["data"]["start_date"]
        
        print(f"Fetching FRED codes: {list(series_map.values())}...")
        try:
            df = web.DataReader(list(series_map.values()), "fred", start, api_key=self.env["FRED_API_KEY"])
            inv_map = {v: k for k, v in series_map.items()}
            df.rename(columns=inv_map, inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching FRED: {e}")
            return pd.DataFrame()

    def get_point_in_time_data(self, as_of: datetime) -> pd.DataFrame:
        """
        Returns a snapshot of all data as it would have appeared at 'as_of'.
        """
        # Load from cache or fetch
        # This implementation should strictly respect lags.
        yf_df = self.fetch_yfinance_series("fx", self.config["data"]["symbols"]["fx"], as_of=as_of).to_frame()
        # ... fetch others ...
        
        # For now, let's just use the existing combined logic but apply lags
        full_df = self.get_all_data(cache=True)
        pit_df = full_df[full_df.index <= as_of].copy()
        
        lags = self.config["data"]["publication_lags"]
        for col, lag in lags.items():
            if col in pit_df.columns and lag > 0:
                # Values at index T should be the values from T - lag
                pit_df[col] = pit_df[col].shift(lag)
        
        return pit_df

    def fetch_bis_reer(self) -> pd.Series:
        """
        Fetches BIS Broad REER for Turkey.
        Monthly data, interpolated to daily.
        """
        cache_path = self._get_cache_path(DataSource.BIS, "TR_REER")
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path).iloc[:, 0]

        print("Fetching BIS REER data...")
        # Note: Downloading the full BIS ZIP is heavy. 
        # For industrial-grade, we'd use a specific endpoint or pre-downloaded CSV.
        # Here we use a placeholder or the CBRT REER if BIS is inaccessible.
        reer = self.fetch_evds_series()["cpi"].copy() # Placeholder: same index
        reer.name = "reer"
        # Dummy values that look like REER for structure verification
        reer[:] = 100.0 
        return reer

    def fetch_offshore_swap(self) -> pd.Series:
        """Placeholder for offshore swap rates."""
        idx = self.fetch_yfinance_series("fx", "USDTRY=X").index
        series = pd.Series(0.40, index=idx, name="swap_rate")
        return series

    def fetch_cds(self) -> pd.Series:
        """Placeholder for 5Y CDS."""
        idx = self.fetch_yfinance_series("fx", "USDTRY=X").index
        series = pd.Series(300.0, index=idx, name="cds")
        return series

    def get_all_data(self, cache: bool = True) -> pd.DataFrame:
        # Existing combined logic updated to use Parquet and Pydantic-like flow internally
        yf_fx = self.fetch_yfinance_series("fx", self.config["data"]["symbols"]["fx"])
        fred_df = self.fetch_fred_series()
        evds_df = self.fetch_evds_series()
        
        reer = self.fetch_bis_reer()
        swaps = self.fetch_offshore_swap()
        cds = self.fetch_cds()
        
        peers = self.config["data"].get("peers", [])
        peer_series = []
        for i, p in enumerate(peers):
            peer_series.append(self.fetch_yfinance_series(f"peer_{i}", p))
            
        combined = pd.concat([yf_fx, fred_df, evds_df, reer, swaps, cds] + peer_series, axis=1)
        combined.sort_index(inplace=True)
        combined = combined.ffill()
        return combined

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.get_all_data()
    print(data.tail())
