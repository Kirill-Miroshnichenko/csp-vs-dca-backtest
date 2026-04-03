# Cash-Secured Put vs DCA — ETH Options Backtest

A Python backtest comparing two ETH accumulation strategies using real Deribit options data (minute bars, Jan 2024 – Feb 2026).

- **Cash-Secured Put (CSP)** — systematically sell put options to collect premiums and accumulate ETH on assignment
- **DCA** — periodic ETH purchases at a fixed USD amount ($100/week)

---

## Results Summary

**Backtest period:** January 2024 – February 2026  
**Starting capital:** $11,200  
**ETH price:** $2,278 → $1,990 (−12.6%, bearish/sideways market)

### Grid sweep — all parameter combinations (lot = 0.2 ETH)

| Moneyness | DTE | Tolerance | Cycles | Assigned | CSP Portfolio | CSP Return | CSP ETH | CSP Premiums | DCA Portfolio | DCA Return |
|-----------|-----|-----------|--------|----------|---------------|------------|---------|--------------|---------------|------------|
| 0.85 | 7  | ±3 | 111 | 5.4%  | $10,999 | −1.8%  | 1.20 | $210  | $8,017 | −28.4% |
| 0.85 | 14 | ±3 | 55  | 12.7% | $10,655 | −4.9%  | 1.40 | $263  | $8,017 | −28.4% |
| 0.85 | 30 | ±7 | 25  | 12.0% | $10,935 | −2.4%  | 0.60 | $290  | $8,017 | −28.4% |
| 0.90 | 7  | ±3 | 111 | 14.4% | $8,865  | −20.8% | 3.20 | $456  | $8,017 | −28.4% |
| 0.90 | 14 | ±3 | 55  | 23.6% | $9,482  | −15.3% | 2.60 | $487  | $8,017 | −28.4% |
| 0.90 | 30 | ±7 | 25  | 20.0% | $10,387 | −7.3%  | 1.00 | $461  | $8,017 | −28.4% |
| 0.95 | 7  | ±3 | 97  | 20.6% | $7,985  | −28.7% | 4.00 | $867  | $8,017 | −28.4% |
| 0.95 | 14 | ±3 | 54  | 37.0% | $8,280  | −26.1% | 4.00 | $962  | $8,017 | −28.4% |
| 0.95 | 30 | ±7 | 25  | 52.0% | $9,280  | −17.1% | 2.60 | $724  | $8,017 | −28.4% |

**Key takeaway:** 8 out of 9 CSP configurations outperformed DCA on portfolio value in a bearish market. The conservative 0.85 moneyness / 7-day DTE configuration lost only −1.8% vs DCA's −28.4%.

---

## Data Format

The script expects Parquet files with minute-bar options data from Deribit:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime64[ns, UTC] | Bar timestamp |
| `instrument_name` | str | e.g. `ETH-28JUN24-2200-P` |
| `expiration_timestamp` | datetime64[ns, UTC] | Expiry at 08:00 UTC |
| `strike` | float | Strike price in USD |
| `option_type` | str | `"put"` or `"call"` |
| `mark_price` | float | In ETH (multiply by `underlying_price` for USD) |
| `underlying_price` | float | ETH/USD spot price |
| `days_to_expiration` | float | Days until expiry |

### Where to get data

- [Deribit API](https://docs.deribit.com/) — historical options data
- [Tardis.dev](https://tardis.dev/) — normalized Deribit tick data (paid)

---

## Installation

```bash
git clone https://github.com/Kirill-Miroshnichenko/csp-vs-dca-backtest.git
cd csp-vs-dca-backtest
pip install -r requirements.txt
```

---

## Usage

1. Place your Parquet files in the `data/` folder
2. Update `DATA_FILES` in `backtest.py` with your filenames:

```python
DATA_DIR = Path("data")

DATA_FILES = [
    DATA_DIR / "eth_options_part1.parquet",
    DATA_DIR / "eth_options_part2.parquet",
    DATA_DIR / "eth_options_part3.parquet",
    DATA_DIR / "eth_options_part4.parquet",
]
```

3. Adjust strategy parameters in the `STRATEGY PARAMETERS` section
4. Run:

```bash
python backtest.py
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_CAPITAL_USD` | 11200 | Starting capital in USD |
| `TARGET_DTE` | 7 | Target days to expiration |
| `DTE_TOLERANCE` | 3 | Expiry search window ±days |
| `STRIKE_MONEYNESS` | 0.85 | Strike as fraction of spot (0.85 = 15% OTM) |
| `LOT_SIZE` | 0.2 | Position size in ETH (Deribit min = 0.1) |
| `DCA_INTERVAL` | weekly | DCA frequency: `"weekly"` or `"monthly"` |
| `DCA_AMOUNT_USD` | 100 | USD amount per DCA purchase |

---

## Memory Note

Full dataset (~300M rows) requires filtering to fit in 16GB RAM. The script automatically filters to the relevant DTE window on load, reducing memory usage by ~90%.

---

## License

MIT
