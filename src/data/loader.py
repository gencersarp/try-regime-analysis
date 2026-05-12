import pandas as pd
import yfinance as yf
from evds import evdsAPI
import pandas_datareader.data as web
from datetime import datetime
from src.utils.config import load_config, setup_env
import os

class DataLoader:
    def __init__(self):
        self.config = load_config()
        self.env = setup_env()
        self.evds = None
        if self.env["EVDS_API_KEY"]:
            self.evds = evdsAPI(self.env["EVDS_API_KEY"])

    def fetch_yfinance_data(self):
        symbols = self.config["data"]["symbols"]
        start = self.config["data"]["start_date"]
        data = {}
        for name, sym in symbols.items():
            print(f"Fetching {name} ({sym})...")
            df = yf.download(sym, start=start)
            if df.empty:
                print(f"Warning: No data for {sym}")
                continue
            
            # Handle yfinance MultiIndex columns if necessary
            if isinstance(df.columns, pd.MultiIndex):
                # Check for 'Adj Close' first, then 'Close'
                if 'Adj Close' in df.columns.get_level_values(0):
                    col = df['Adj Close']
                elif 'Close' in df.columns.get_level_values(0):
                    col = df['Close']
                else:
                    print(f"Warning: Neither Adj Close nor Close found for {sym}")
                    continue
                # If MultiIndex has ticker names at level 1, select the ticker
                if len(col.shape) > 1 and sym in col.columns:
                    data[name] = col[sym]
                else:
                    data[name] = col.iloc[:, 0]
            else:
                if "Adj Close" in df.columns:
                    data[name] = df["Adj Close"]
                elif "Close" in df.columns:
                    data[name] = df["Close"]
                else:
                    continue
        return pd.DataFrame(data)

    def fetch_evds_data(self):
        if not self.evds:
            print("EVDS API Key missing. Skipping EVDS data.")
            return pd.DataFrame()
        
        series_config = self.config["data"]["evds_series"]
        start = self.config["data"]["start_date"]
        start_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%d-%m-%Y")
        
        # EVDS get_data expects a list of codes
        codes = list(series_config.values())
        print(f"Fetching EVDS codes: {codes}...")
        try:
            df = self.evds.get_data(codes, startdate=start_fmt)
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Use 'mixed' format as suggested by the error message to handle various date formats
            df["Tarih"] = pd.to_datetime(df["Tarih"], format="mixed")
            df.set_index("Tarih", inplace=True)
            
            # Rename columns from EVDS codes back to readable names
            inv_map = {v: k for k, v in series_config.items()}
            # Note: EVDS returns codes with underscores (TP_AB_A02) instead of dots (TP.AB.A02)
            new_cols = []
            for col in df.columns:
                # Replace dots with underscores in our map to match EVDS output
                mapped = False
                for code, name in inv_map.items():
                    clean_code = code.replace(".", "_")
                    if clean_code.upper() in col.upper():
                        new_cols.append(name)
                        mapped = True
                        break
                if not mapped:
                    new_cols.append(col)
            df.columns = new_cols
            return df
        except Exception as e:
            print(f"Error fetching EVDS: {e}")
            return pd.DataFrame()

    def fetch_fred_data(self):
        if not self.env["FRED_API_KEY"]:
            print("FRED API Key missing. Skipping FRED.")
            return pd.DataFrame()
            
        series = self.config["data"]["fred_series"]
        start = self.config["data"]["start_date"]
        print("Fetching FRED data...")
        try:
            # Setting the API key for pandas_datareader if needed
            # os.environ["FRED_API_KEY"] = self.env["FRED_API_KEY"]
            df = web.DataReader(list(series.values()), "fred", start, api_key=self.env["FRED_API_KEY"])
            inv_map = {v: k for k, v in series.items()}
            df.rename(columns=inv_map, inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching FRED: {e}")
            return pd.DataFrame()

    def get_all_data(self, cache=True):
        cache_path = "data/cache/raw_combined.csv"
        if cache and os.path.exists(cache_path):
            print("Loading data from cache...")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty:
                return df

        yf_df = self.fetch_yfinance_data()
        fred_df = self.fetch_fred_data()
        evds_df = self.fetch_evds_data()

        # Concatenate and sort
        combined = pd.concat([yf_df, fred_df, evds_df], axis=1)
        combined.sort_index(inplace=True)
        
        # Filter dates after start_date
        start_date = pd.to_datetime(self.config["data"]["start_date"])
        combined = combined[combined.index >= start_date]

        # Fill missing values for daily analysis
        # Only drop if the target 'fx' is missing
        if "fx" in combined.columns:
            combined = combined.ffill()
            combined = combined.dropna(subset=["fx"])
        
        if cache and not combined.empty:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            combined.to_csv(cache_path)
        
        return combined

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.get_all_data()
    print(data.head())
