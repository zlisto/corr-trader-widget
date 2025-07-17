# Correlation Trading Simulator

This project is a Streamlit-based web application for simulating and visualizing correlation-based trading strategies on cryptocurrency pairs. It allows users to select trading pairs, set statistical thresholds, and analyze the performance of a strategy that generates signals from one asset and trades another correlated asset.

## Features
- Interactive Streamlit UI for parameter selection
- Visualization of cumulative returns and trading signals
- Performance metrics: Sharpe ratio, drawdown, trade stats
- Uses historical hourly open and volume data from Kraken

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data
Place the following CSV files in the `data/` directory:
- `kraken_hourly_open_2025-04-10_to_2025-07-12.csv`
- `kraken_hourly_volume_2025-04-10_to_2025-07-12.csv`

These files should contain hourly open prices and volumes for all trading pairs.

## Usage
Run the Streamlit app with:

```bash
streamlit run widget_corr_trader.py
```

- Select the start date, signal pair, traded pair, t-stat window, entry threshold, and trade fee.
- Click **Simulate & Plot** to view results and performance metrics.

## Files
- `widget_corr_trader.py`: Streamlit app UI and logic
- `corr_trade_simulator.py`: Core trading simulation logic
- `data/`: Directory for CSV data files

## License
MIT License 