"""
ETH Options Backtest: DCA vs Cash-Secured Put Selling
======================================================
Compares two ETH accumulation strategies over historical Deribit options data.

Strategy A — DCA:
    Buy ETH with a fixed USD amount every week at market price.

Strategy B — Put Selling:
    Every week, sell a cash-secured put at a chosen strike (ATM / 10% OTM / 20% OTM).
    - If assigned (price < strike at expiry): receive ETH at strike price, premium kept.
    - If not assigned (price > strike at expiry): keep premium, reinvest next cycle.

Usage:
    python backtest.py --data_dir ./data --weekly_budget 200 --strikes atm otm10 otm20

Requirements:
    pip install pandas numpy matplotlib seaborn tqdm
"""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    data_dir: str = "./data"
    weekly_budget_usd: float = 200.0          # USD invested per week in DCA
    expiry_days_target: int = 7               # target DTE when entering put (7 = weekly)
    expiry_days_tolerance: int = 3            # ± days tolerance for finding the right expiry
    strikes: list = field(default_factory=lambda: ["atm", "otm10", "otm20"])
    deribit_fee_rate: float = 0.0003          # 0.03% of underlying per option contract
    sell_hour_utc: int = 9                    # hour to enter new position after expiry (UTC)
    output_dir: str = "./results"
    plot: bool = True


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(config: BacktestConfig) -> pd.DataFrame:
    """Load all parquet/csv files from data_dir and return a clean DataFrame."""
    data_path = Path(config.data_dir)
    files = sorted(list(data_path.glob("*.parquet")) + list(data_path.glob("*.csv")))

    if not files:
        raise FileNotFoundError(f"No parquet or csv files found in {config.data_dir}")

    print(f"Loading {len(files)} file(s)...")
    dfs = []
    for f in files:
        print(f"  {f.name}")
        if f.suffix == ".parquet":
            dfs.append(pd.read_parquet(f))
        else:
            dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["expiration_timestamp"] = pd.to_datetime(df["expiration_timestamp"], utc=True)

    # Keep only puts with valid data
    df = df[df["option_type"] == "put"].copy()
    df = df[df["mark_price"] > 0].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df):,} put option rows")
    print(f"Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"Underlying price range: ${df['underlying_price'].min():,.0f} – ${df['underlying_price'].max():,.0f}")

    return df


# ---------------------------------------------------------------------------
# Strike Selection
# ---------------------------------------------------------------------------

def select_strike(
    available_strikes: pd.Series,
    current_price: float,
    strike_mode: str,
) -> Optional[float]:
    """
    Find the closest available strike to the target.
    strike_mode: 'atm' | 'otm10' | 'otm20'
    """
    targets = {
        "atm":   current_price * 1.00,
        "otm10": current_price * 0.90,
        "otm20": current_price * 0.80,
    }
    target = targets[strike_mode]
    if available_strikes.empty:
        return None
    idx = (available_strikes - target).abs().idxmin()
    return available_strikes[idx]


# ---------------------------------------------------------------------------
# DCA Backtest
# ---------------------------------------------------------------------------

def run_dca(weekly_ts: pd.DatetimeIndex, price_at: dict, budget: float) -> pd.DataFrame:
    """
    Simple DCA: buy ETH for `budget` USD each week.
    price_at: {timestamp -> underlying_price}
    """
    records = []
    eth_total = 0.0
    cash_spent = 0.0

    for ts in weekly_ts:
        price = price_at.get(ts)
        if price is None or price <= 0:
            continue
        eth_bought = budget / price
        eth_total += eth_bought
        cash_spent += budget
        records.append({
            "date": ts,
            "eth_bought": eth_bought,
            "price": price,
            "eth_total": eth_total,
            "cash_spent": cash_spent,
            "portfolio_usd": eth_total * price,
            "avg_entry_price": cash_spent / eth_total,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Put Selling Backtest
# ---------------------------------------------------------------------------

def run_put_selling(
    df: pd.DataFrame,
    config: BacktestConfig,
    strike_mode: str,
) -> pd.DataFrame:
    """
    Cash-secured put selling strategy.

    Each cycle:
    1. At entry timestamp, find a put expiring in ~7 days at target strike.
    2. Sell at mark_price → collect premium in USD.
    3. At expiry, check if assigned (underlying < strike).
       - Assigned: buy ETH at strike, deduct fee.
       - Not assigned: premium rolled into next cycle cash.
    4. Track total ETH, total cash deployed, premium income.
    """
    records = []
    eth_total = 0.0
    cash_balance = config.weekly_budget_usd  # starting collateral
    total_premiums = 0.0
    total_fees = 0.0
    assignments = 0
    total_cycles = 0

    # Get sorted unique timestamps (minute-level)
    all_ts = df["timestamp"].sort_values().unique()

    # Find weekly entry points (every 7 days from start)
    start_ts = pd.Timestamp(all_ts[0])
    end_ts = pd.Timestamp(all_ts[-1])

    entry_ts = start_ts
    while entry_ts <= end_ts:
        total_cycles += 1

        # --- 1. Find the closest timestamp in data ---
        nearest_idx = np.searchsorted(all_ts, entry_ts)
        if nearest_idx >= len(all_ts):
            break
        actual_entry_ts = pd.Timestamp(all_ts[nearest_idx])

        # --- 2. Get snapshot at entry ---
        snapshot = df[df["timestamp"] == actual_entry_ts].copy()
        if snapshot.empty:
            entry_ts += pd.Timedelta(days=7)
            continue

        underlying_price = snapshot["underlying_price"].iloc[0]

        # --- 3. Filter puts by target DTE ---
        target_dte = config.expiry_days_target
        tol = config.expiry_days_tolerance
        candidates = snapshot[
            (snapshot["days_to_expiration"] >= target_dte - tol) &
            (snapshot["days_to_expiration"] <= target_dte + tol)
        ].copy()

        if candidates.empty:
            # Fallback: find nearest available DTE
            candidates = snapshot.copy()

        # --- 4. Select strike ---
        available_strikes = candidates["strike"].drop_duplicates()
        chosen_strike = select_strike(available_strikes, underlying_price, strike_mode)

        if chosen_strike is None:
            entry_ts += pd.Timedelta(days=7)
            continue

        # Get the specific put row
        put_row = candidates[candidates["strike"] == chosen_strike].iloc[0]
        premium_eth = put_row["mark_price"]          # mark_price is in ETH
        premium_usd = premium_eth * underlying_price
        expiry_ts = put_row["expiration_timestamp"]
        dte = put_row["days_to_expiration"]

        # Fee to sell 1 put (0.03% of underlying, Deribit standard)
        fee_usd = config.deribit_fee_rate * underlying_price
        net_premium_usd = premium_usd - fee_usd
        total_fees += fee_usd

        # --- 5. Find price at expiry ---
        expiry_snap = df[df["timestamp"] >= expiry_ts]
        if expiry_snap.empty:
            # Data ends before expiry — use last available price
            expiry_price = df["underlying_price"].iloc[-1]
        else:
            expiry_price = expiry_snap.iloc[0]["underlying_price"]

        # --- 6. Assignment logic ---
        assigned = expiry_price < chosen_strike

        if assigned:
            # We buy ETH at strike price (collateral used)
            eth_received = cash_balance / chosen_strike  # 1 "unit" worth of ETH
            eth_total += eth_received
            assignments += 1
            cash_balance = net_premium_usd  # reset cash to just the premium earned
            assignment_note = "ASSIGNED"
        else:
            # Keep premium, cash stays intact
            cash_balance += net_premium_usd
            assignment_note = "expired worthless"

        total_premiums += net_premium_usd

        portfolio_usd = eth_total * expiry_price + cash_balance

        records.append({
            "date": actual_entry_ts,
            "expiry": expiry_ts,
            "dte": round(dte, 1),
            "underlying_entry": underlying_price,
            "underlying_expiry": expiry_price,
            "strike": chosen_strike,
            "strike_pct": round(chosen_strike / underlying_price * 100, 1),
            "premium_usd": round(premium_usd, 2),
            "fee_usd": round(fee_usd, 4),
            "net_premium_usd": round(net_premium_usd, 2),
            "assigned": assigned,
            "result": assignment_note,
            "eth_total": eth_total,
            "cash_balance": round(cash_balance, 2),
            "total_premiums": round(total_premiums, 2),
            "portfolio_usd": round(portfolio_usd, 2),
            "avg_entry_price": round(chosen_strike if assigned else 0, 2),
        })

        # Next entry: day after expiry
        entry_ts = expiry_ts + pd.Timedelta(days=1)

    df_out = pd.DataFrame(records)
    df_out["assignment_rate"] = assignments / max(total_cycles, 1)
    return df_out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, strategy_name: str) -> dict:
    """Compute summary performance metrics for a strategy."""
    if df.empty:
        return {}

    portfolio = df["portfolio_usd"].values
    returns = np.diff(portfolio) / portfolio[:-1]

    max_dd = 0.0
    peak = portfolio[0]
    for v in portfolio:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    sharpe = (returns.mean() / returns.std() * np.sqrt(52)) if returns.std() > 0 else 0

    metrics = {
        "Strategy": strategy_name,
        "Final ETH": round(df["eth_total"].iloc[-1], 4),
        "Final portfolio USD": f"${df['portfolio_usd'].iloc[-1]:,.0f}",
        "Total cash deployed USD": f"${df['cash_spent'].iloc[-1]:,.0f}" if "cash_spent" in df.columns else "N/A",
        "Total premiums earned USD": f"${df['total_premiums'].iloc[-1]:,.0f}" if "total_premiums" in df.columns else "N/A",
        "Max drawdown": f"{max_dd*100:.1f}%",
        "Sharpe ratio (weekly)": round(sharpe, 2),
        "Assignment rate": f"{df['assignment_rate'].iloc[-1]*100:.0f}%" if "assignment_rate" in df.columns else "N/A",
        "Avg entry price USD": f"${df['avg_entry_price'].mean():,.0f}" if "avg_entry_price" in df.columns else "N/A",
    }
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(dca_df: pd.DataFrame, put_results: dict, output_dir: str):
    """Generate comparison charts."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle("ETH Accumulation: DCA vs Put Selling (Deribit)", fontsize=15, fontweight="bold")

    colors = {
        "DCA": "#4A90D9",
        "Put ATM": "#E8593C",
        "Put 10% OTM": "#2ECC71",
        "Put 20% OTM": "#9B59B6",
    }

    # --- Chart 1: Portfolio value (USD) ---
    ax1 = axes[0]
    ax1.plot(dca_df["date"], dca_df["portfolio_usd"], label="DCA", color=colors["DCA"], linewidth=2)
    for label, df in put_results.items():
        if not df.empty:
            ax1.plot(df["date"], df["portfolio_usd"], label=label, color=colors.get(label, "gray"), linewidth=1.5)
    ax1.set_title("Portfolio Value (USD)", fontsize=12)
    ax1.set_ylabel("USD")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # --- Chart 2: ETH accumulated ---
    ax2 = axes[1]
    ax2.plot(dca_df["date"], dca_df["eth_total"], label="DCA", color=colors["DCA"], linewidth=2)
    for label, df in put_results.items():
        if not df.empty:
            ax2.plot(df["date"], df["eth_total"], label=label, color=colors.get(label, "gray"), linewidth=1.5)
    ax2.set_title("ETH Accumulated", fontsize=12)
    ax2.set_ylabel("ETH")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # --- Chart 3: Premiums earned (put selling only) ---
    ax3 = axes[2]
    for label, df in put_results.items():
        if not df.empty and "total_premiums" in df.columns:
            ax3.plot(df["date"], df["total_premiums"], label=label, color=colors.get(label, "gray"), linewidth=1.5)
    ax3.set_title("Cumulative Premiums Earned (Put Selling)", fontsize=12)
    ax3.set_ylabel("USD")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "backtest_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved → {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # CONFIGURE HERE — edit these values directly (works in Jupyter & CLI)
    # -----------------------------------------------------------------------
    config = BacktestConfig(
        data_dir          = "./data",
        weekly_budget_usd = 200.0,
        expiry_days_target = 7,
        expiry_days_tolerance = 3,
        strikes           = ["atm", "otm10", "otm20"],
        deribit_fee_rate  = 0.0003,
        output_dir        = "./results",
        plot              = True,
    )
    # -----------------------------------------------------------------------

    # --- Load data ---
    df = load_data(config)

    # --- Build weekly price index (for DCA) ---
    # Use the first observation of each Monday (or nearest available day)
    price_series = (
        df.groupby("timestamp")["underlying_price"]
        .first()
        .reset_index()
        .sort_values("timestamp")
    )
    price_series = price_series.set_index("timestamp")["underlying_price"]

    # Weekly timestamps every 7 days
    weekly_ts = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="7D",
        tz="UTC",
    )

    # Map each weekly_ts to nearest available price
    all_ts_sorted = price_series.index.sort_values()
    price_at = {}
    for wts in weekly_ts:
        idx = all_ts_sorted.searchsorted(wts)
        if idx < len(all_ts_sorted):
            price_at[wts] = price_series.iloc[idx]

    # --- Run DCA ---
    print("\n=== Running DCA backtest ===")
    dca_df = run_dca(weekly_ts, price_at, config.weekly_budget_usd)
    print(f"DCA cycles: {len(dca_df)}")

    # --- Run Put Selling ---
    put_results = {}
    strike_labels = {"atm": "Put ATM", "otm10": "Put 10% OTM", "otm20": "Put 20% OTM"}

    for strike_mode in config.strikes:
        label = strike_labels.get(strike_mode, strike_mode)
        print(f"\n=== Running Put Selling ({label}) ===")
        result_df = run_put_selling(df, config, strike_mode)
        put_results[label] = result_df
        print(f"Cycles: {len(result_df)}, Assignments: {result_df['assigned'].sum() if not result_df.empty else 0}")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_metrics = []

    if not dca_df.empty:
        dca_metrics = compute_metrics(dca_df, "DCA")
        all_metrics.append(dca_metrics)

    for label, res_df in put_results.items():
        if not res_df.empty:
            m = compute_metrics(res_df, label)
            all_metrics.append(m)

    if all_metrics:
        summary = pd.DataFrame(all_metrics).set_index("Strategy")
        print(summary.to_string())

        os.makedirs(config.output_dir, exist_ok=True)
        summary_path = os.path.join(config.output_dir, "summary.csv")
        summary.to_csv(summary_path)
        print(f"\nSummary saved → {summary_path}")

        # Save detailed logs
        if not dca_df.empty:
            dca_df.to_csv(os.path.join(config.output_dir, "dca_log.csv"), index=False)
        for label, res_df in put_results.items():
            if not res_df.empty:
                fname = f"puts_{label.replace(' ', '_').lower()}_log.csv"
                res_df.to_csv(os.path.join(config.output_dir, fname), index=False)

    # --- Plot ---
    if config.plot and not dca_df.empty:
        plot_results(dca_df, put_results, config.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
