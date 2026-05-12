# TRY Regime Analysis: Intervention, Stability, and Structural Breaks

## Research Motivation
This project investigates the dynamics of the Turkish Lira (TRY) FX regime. It focuses on identifying whether the currency is artificially stabilized, detecting intervention regimes, and predicting structural breaks or devaluations.

### Key Research Questions
- Is TRY following a controlled crawl or a free float?
- Can central bank intervention be inferred from volatility suppression and reserve dynamics?
- What macro signals precede sharp policy shifts or devaluations?
- Are there hidden mean-reverting regimes in a long-run depreciation trend?

## Project Structure
- `/src`: Core source code for data ingestion, feature engineering, and modeling.
- `/notebooks`: Research and exploration notebooks.
- `/reports`: Generated research reports and visualizations.
- `/configs`: Configuration files for reproducibility.

## How to Run

### 1. Prerequisites
- Python 3.9+
- A CBRT EVDS API Key (Get it free at [evds2.tcmb.gov.tr](https://evds2.tcmb.gov.tr/))
- (Optional) A FRED API Key (Get it free at [fred.stlouisfed.org](https://fred.stlouisfed.org/))

### 2. Installation
```bash
git clone <your-repo-url>
cd try-regime-analysis
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
EVDS_API_KEY=your_key_here
FRED_API_KEY=your_fred_key_here
```

### 4. Running the Pipeline
- **Fetch Data**:
  ```bash
  python src/main.py fetch-data
  ```
- **Run Full Analysis**:
  ```bash
  python src/main.py analyze --carry-rate 0.45 --output-dir reports
  ```
  *(Note: --carry-rate should be updated to current TRY overnight rates, e.g., 0.45 for 45%)*

- **Interactive Research**:
  Open `research_to_production.ipynb` in Jupyter Lab or VS Code to run the interactive cells.

## Data Source Costs & API Status
| Source | Cost | Key Required? | Notes |
|--------|------|---------------|-------|
| **CBRT EVDS** | Free | Yes | Official Turkish Central Bank data. Requires registration. |
| **FRED** | Free | Yes | St. Louis Fed data. High rate limits. |
| **yfinance** | Free | No | Scrapes Yahoo Finance. No key, but subject to IP rate limits. |
| **Investing.com** | Free | No | Used via internal scrapers or CSVs if needed. |

## Recommended Strategic Additions
1. **Swap Market Data**: Integrate 1-week and 1-month offshore TRY swap rates. High swap rates often precede devaluations as the CBRT squeezes liquidity.
2. **News Sentiment**: Add a scraper for Bloomberg or Reuters headlines related to "CBRT", "TCMB", or "Lira".
3. **Alternative Reserve Proxies**: Track state bank FX positions if data is available, as these are often used for "backdoor" interventions.
4. **Kalman Filter**: Use a Kalman Filter for dynamic drift estimation instead of simple rolling means for more responsive trend detection.

## Real Money Deployment Checklist
Before deploying capital using this system, ensure:
1.  **Data Integrity**: Verify that your EVDS and FRED API keys are active and fetching the most recent daily data.
2.  **Walk-Forward Consistency**: Ensure the `train_window` and `step` parameters in the regime detector are tuned to capture recent policy shifts without overfitting.
3.  **Carry Rates**: Update the `carry_rate` parameter to reflect current TRY overnight lending/borrowing rates (often volatile).
4.  **Transaction Costs**: Confirm that your broker's spreads and commissions are within the 5 bps (0.0005) assumption.
5.  **Liquidity Risk**: Be aware that in "Panic" regimes (Regime 2), USDTRY liquidity can evaporate, and spreads can widen significantly beyond historical averages.
6.  **Tail Risk**: The HMM assumes Gaussian returns; however, TRY is prone to "Black Swan" events. Always use hard stop-losses and respect the VaR/ES limits.
