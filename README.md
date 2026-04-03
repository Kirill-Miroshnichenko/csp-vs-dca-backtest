# Cash-Secured Put vs DCA — ETH Options Backtest

A Python backtest comparing two ETH accumulation strategies using Deribit options data:

- **Cash-Secured Put (CSP)** — systematically sell put options to collect premiums and accumulate ETH on assignment
- **DCA** — periodic ETH purchases at a fixed USD amount

## Results Summary

Backtest period: January 2024 – February 2026 | Capital: $11,200

| Strategy | Parameters | Portfolio | ETH accumulated | Avg buy price |
|---|---|---|---|---|
| CSP | moneyness=0.95, DTE=14±3, lot=0.2 | $12,105 (+8.1%) | 4.0 ETH | $2,960 |
| CSP | moneyness=0.95, DTE=7±3, lot=0.2 | $10,965 (-2.1%) | 4.0 ETH | $3,010 |
| CSP | moneyness=0.85, DTE=7±3, lot=0.2 | $10,930 (-2.4%) | 1.2 ETH | $2,333 |
| DCA | $100/week | $8,017 (-28.4%) | 3.92 ETH | $2,860 |

## Data Format

The script expects Parquet files with minute-bar options data from Deribit with the following columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime64[ns, UTC] | Bar timestamp |
| `instrument_name` | str | e.g. `ETH-28JUN24-2200-P` |
| `expiration_timestamp` | datetime64[ns, UTC] | Expiry at 08:00 UTC |
| `strike` | float | Strike price in USD |
| `option_type` | str | `"put"` or `"call"` |
| `mark_price` | float | In ETH (multiply by underlying_price for USD) |
| `underlying_price` | float | ETH/USD spot price |
| `days_to_expiration` | float | Days until expiry |

### Where to get data

- [Deribit API](https://docs.deribit.com/) — historical options data
- [Tardis.dev](https://tardis.dev/) — normalized Deribit tick data (paid)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/csp-vs-dca-backtest.git
cd csp-vs-dca-backtest
pip install -r requirements.txt
```

## Usage

1. Place your Parquet files in the `data/` folder
2. Update `DATA_FILES` in `backtest.py` with your filenames
3. Adjust strategy parameters in the `STRATEGY PARAMETERS` section
4. Run:

```bash
python backtest.py
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `INITIAL_CAPITAL_USD` | 11200 | Starting capital in USD |
| `TARGET_DTE` | 7 | Target days to expiration |
| `DTE_TOLERANCE` | 3 | Expiry search window ±days |
| `STRIKE_MONEYNESS` | 0.95 | Strike as fraction of spot (0.95 = 5% OTM) |
| `LOT_SIZE` | 0.2 | Position size in ETH (Deribit min = 0.1) |
| `DCA_INTERVAL` | weekly | DCA frequency: `"weekly"` or `"monthly"` |
| `DCA_AMOUNT_USD` | 100 | USD amount per DCA purchase |

## Memory Note

Full dataset (~300M rows) requires filtering to fit in 16GB RAM. The script automatically filters to the relevant DTE window on load, reducing memory usage by ~90%.

## License

MIT
