import os
import yaml
from dotenv import load_dotenv

def load_config(config_path="configs/main_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_env():
    load_dotenv()
    return {
        "EVDS_API_KEY": os.getenv("EVDS_API_KEY"),
        "FRED_API_KEY": os.getenv("FRED_API_KEY")
    }
