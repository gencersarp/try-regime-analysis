from datetime import date, datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from enum import Enum

class DataSource(str, Enum):
    CBRT = "evds"
    FRED = "fred"
    YAHOO = "yfinance"
    BIS = "bis"

class GaugeBand(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

class GaugeMetadata(BaseModel):
    name: str
    description: str
    is_leading: bool
    thresholds: Dict[GaugeBand, float]
    source_citation: str

class DashboardConfig(BaseModel):
    gauges: List[GaugeMetadata]
    composite_rules: Dict[str, str]

class DataFrequency(str, Enum):
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"

class SeriesMetadata(BaseModel):
    name: str
    symbol: str
    source: DataSource
    frequency: DataFrequency
    publication_lag_days: int = Field(..., description="Usual lag between observation and publication")
    description: Optional[str] = None

class DataPoint(BaseModel):
    timestamp: datetime
    value: float
    available_at: datetime = Field(..., description="Timestamp when this data point became publicly available")

class SeriesSnapshot(BaseModel):
    metadata: SeriesMetadata
    data: List[DataPoint]
    fetched_at: datetime = Field(default_factory=datetime.now)

class ConfigData(BaseModel):
    start_date: date
    end_date: Optional[date] = None
    symbols: Dict[str, str]
    peers: List[str]
    evds_series: Dict[str, str]
    fred_series: Dict[str, str]
    publication_lags: Dict[str, int]
