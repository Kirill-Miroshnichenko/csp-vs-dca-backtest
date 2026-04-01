# =============================================================================
# Cash-Secured Put (CSP) vs DCA — ETH Options Backtest
# =============================================================================
#
# Strategy 1: DCA — periodic ETH purchases at fixed USD amount
# Strategy 2: Cash-Secured Put (CSP) — systematically sell puts to collect
#             premiums and accumulate ETH on assignment
#
# Data format (Parquet, minute bars from Deribit):
#   timestamp               — datetime64[ns, UTC]
#   instrument_name         — str, e.g. "ETH-28JUN24-2200-P"
#   expiration_timestamp    — datetime64[ns, UTC], expiry at 08:00 UTC
#   strike                  — float, strike price in USD
#   option_type             — str, "put" / "call"
#   mark_price              — float, in ETH (multiply by underlying_price for USD)
#   underlying_price        — float, ETH/USD spot price
#   days_to_expiration      — float, days until expiry
#
# Usage:
#   1. Place your Parquet files in DATA_DIR (or update DATA_FILES list)
#   2. Adjust strategy parameters in the CONFIG section
#   3. Run: python backtest.py
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# DATA PATHS — update these to point to your data files
# -----------------------------------------------------------------------------

DATA_DIR = Path("./data")           # folder containing Parquet files

# List your Parquet files here in chronological order
DATA_FILES = [
    DATA_DIR / "part_1.parquet",
    DATA_DIR / "part_2.parquet",
    DATA_DIR / "part_3.parquet",
    DATA_DIR / "part_4.parquet",
]

# To run on the included sample file (2 weeks of data for testing):
# DATA_FILES = [DATA_DIR / "sample.parquet"]

# -----------------------------------------------------------------------------
# STRATEGY PARAMETERS — edit here
# -----------------------------------------------------------------------------

# --- General ---
INITIAL_CAPITAL_USD = 11_200       # starting capital in USD

# --- CSP: expiry selection ---
TARGET_DTE    = 7                  # target days to expiration
DTE_TOLERANCE = 3                  # tolerance ± days: selects expiry in [TARGET_DTE ± DTE_TOLERANCE]

# --- CSP: strike selection ---
# STRIKE_MONEYNESS = 1.00 → ATM
# STRIKE_MONEYNESS = 0.95 → 5%  OTM
# STRIKE_MONEYNESS = 0.90 → 10% OTM
# STRIKE_MONEYNESS = 0.85 → 15% OTM
STRIKE_MONEYNESS = 0.95

# --- CSP: position size ---
# Deribit minimum lot size for ETH options = 0.1 ETH
LOT_SIZE = 0.2

# --- Fees ---
# Deribit taker fee: 0.03% of underlying, capped at 12.5% of premium
WHEEL_FEE_OPEN_RATE     = 0.0003   # fee on opening (selling) the put
WHEEL_FEE_DELIVERY_RATE = 0.00015  # delivery fee on assignment
DCA_FEE_RATE            = 0.001    # spot purchase fee (typical taker)

# --- DCA ---
DCA_INTERVAL   = "weekly"          # "weekly" or "monthly"
DCA_AMOUNT_USD = 100               # USD amount per purchase

# -----------------------------------------------------------------------------
# COLUMNS NEEDED — reduces memory usage on load
# -----------------------------------------------------------------------------

NEEDED_COLUMNS = [
    "timestamp",
    "instrument_name",
    "expiration_timestamp",
    "strike",
    "option_type",
    "mark_price",           # in ETH — multiply by underlying_price for USD
    "underlying_price",     # ETH/USD spot price
    "days_to_expiration",
]

# -----------------------------------------------------------------------------
# STEP 1: DATA LOADING
# -----------------------------------------------------------------------------

def load_puts_only(files: list[Path]) -> pd.DataFrame:
    """
    Loads put options filtered to the relevant DTE window.

    Filters applied per file to minimize memory usage:
    1. option_type == "put"
    2. mark_price > 0  (zero-premium rows are useless)
    3. days_to_expiration in [TARGET_DTE - DTE_TOLERANCE, TARGET_DTE + DTE_TOLERANCE]
       → drops ~90% of rows with irrelevant expiries
    4. Downcast float64 → float32 (~2x memory saving)

    Result: from ~300M rows to ~5-15M — fits in 16GB RAM.
    """
    dte_min = TARGET_DTE - DTE_TOLERANCE
    dte_max = TARGET_DTE + DTE_TOLERANCE

    chunks = []
    for f in files:
        print(f"  Loading {f.name}...")
        df = pd.read_parquet(f, columns=NEEDED_COLUMNS)
        before = len(df)

        df = df[df["option_type"] == "put"]
        df = df[
            (df["days_to_expiration"] >= dte_min) &
            (df["days_to_expiration"] <= dte_max)
        ]
        df = df[df["mark_price"] > 0].copy()

        float_cols = ["mark_price", "underlying_price", "strike", "days_to_expiration"]
        df[float_cols] = df[float_cols].astype("float32")
        df["mark_price_usd"] = df["mark_price"].astype("float64") * df["underlying_price"].astype("float64")

        after = len(df)
        ratio = after / before * 100 if before > 0 else 0
        mem   = df.memory_usage(deep=True).sum() / 1024**2
        print(f"    → {before:,} rows → {after:,} after filtering ({ratio:.1f}%) | {mem:.0f} MB")
        chunks.append(df)
        del df

    data = pd.concat(chunks, ignore_index=True)
    del chunks
    data = data.sort_values("timestamp").reset_index(drop=True)

    total_mem = data.memory_usage(deep=True).sum() / 1024**3
    print(f"\nTotal: {len(data):,} rows | {total_mem:.2f} GB in memory")
    print(f"Range: {data['timestamp'].min()} → {data['timestamp'].max()}")
    return data


def load_spot_series(files: list[Path]) -> pd.DataFrame:
    """
    Loads ETH spot price series independently of the DTE filter.
    Used for DCA so purchases are not affected by option data availability.
    """
    chunks = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "underlying_price", "option_type"])
        df = df[df["option_type"] == "put"][["timestamp", "underlying_price"]]
        chunks.append(df.drop_duplicates("timestamp"))
        del df

    spot = (
        pd.concat(chunks)
        .drop_duplicates("timestamp")
        .set_index("timestamp")
        .sort_index()
    )
    del chunks
    print(f"  Spot series: {len(spot):,} bars | {spot.index.min()} → {spot.index.max()}")
    return spot


def get_expiry_schedule(df: pd.DataFrame) -> pd.Series:
    """Returns all unique expiration timestamps found in the data."""
    expiries = (
        df["expiration_timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    print(f"\nUnique expiries in data: {len(expiries)}")
    print(f"First 5: {expiries.head().tolist()}")
    print(f"Last 5:  {expiries.tail().tolist()}")
    return expiries


# -----------------------------------------------------------------------------
# STEP 2: CONTRACT SELECTION
# -----------------------------------------------------------------------------

def select_contract(
    puts: pd.DataFrame,
    open_ts: pd.Timestamp,
    spot: float,
    target_dte: int    = TARGET_DTE,
    dte_tolerance: int = DTE_TOLERANCE,
    moneyness: float   = STRIKE_MONEYNESS,
) -> dict | None:
    """
    Selects one put contract at open_ts:

    1. Expiry selection: find all available expiries at open_ts,
       keep those within [target_dte ± dte_tolerance] days,
       pick the one closest to target_dte.

    2. Strike selection: target_strike = spot * moneyness,
       pick the available strike closest to target_strike.

    Returns a dict with contract details, or None if no suitable contract found.
    """
    snapshot = puts[puts["timestamp"] == open_ts].copy()
    if snapshot.empty:
        return None

    dte_min = target_dte - dte_tolerance
    dte_max = target_dte + dte_tolerance

    available_expiries = snapshot["expiration_timestamp"].unique()
    dte_map = {
        exp: (exp - open_ts).total_seconds() / 86400
        for exp in available_expiries
    }

    valid = {exp: dte for exp, dte in dte_map.items() if dte_min <= dte <= dte_max}
    if not valid:
        return None

    chosen_expiry = min(valid, key=lambda exp: abs(valid[exp] - target_dte))
    actual_dte    = valid[chosen_expiry]

    target_strike = spot * moneyness
    candidates    = snapshot[snapshot["expiration_timestamp"] == chosen_expiry].copy()
    if candidates.empty:
        return None

    candidates["strike_dist"] = (candidates["strike"] - target_strike).abs()
    best = candidates.loc[candidates["strike_dist"].idxmin()]

    return {
        "open_ts":          open_ts,
        "instrument_name":  best["instrument_name"],
        "expiration_ts":    chosen_expiry,
        "actual_dte":       round(actual_dte, 2),
        "strike":           best["strike"],
        "target_strike":    round(target_strike, 2),
        "spot_at_open":     spot,
        "underlying_price": spot,
        "premium_eth":      best["mark_price"],
        "premium_usd":      best["mark_price_usd"],
    }


# -----------------------------------------------------------------------------
# STEP 3: CSP BACKTEST LOOP
# -----------------------------------------------------------------------------

def build_csp_pnl(puts: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the Cash-Secured Put backtest cycle by cycle:
      open position → wait for expiry → open next position → ...

    Margin rule: do not open a new put if cash < strike * LOT_SIZE.
    If insufficient margin, wait 1 hour and retry.

    P&L mechanics:
    - cash += net_premium on open (premium received minus fee)
    - worthless expiry: no change
    - assigned: cash -= strike * LOT_SIZE + delivery_fee; eth += LOT_SIZE
    - portfolio_value = cash + eth * spot_at_expiry
    """
    all_timestamps = puts["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)

    cash = float(INITIAL_CAPITAL_USD)
    eth  = 0.0
    current_ts = all_timestamps.iloc[0]
    rows = []
    skipped_no_exp    = 0
    skipped_no_margin = 0

    while True:
        idx = all_timestamps.searchsorted(current_ts)
        if idx >= len(all_timestamps):
            break
        current_ts = all_timestamps.iloc[idx]

        spot_rows = puts[puts["timestamp"] == current_ts]["underlying_price"]
        if spot_rows.empty:
            current_ts = all_timestamps.iloc[min(idx + 1, len(all_timestamps) - 1)]
            continue
        spot = float(spot_rows.iloc[0])

        contract = select_contract(puts, current_ts, spot)
        if contract is None:
            skipped_no_exp += 1
            next_ts = current_ts + pd.Timedelta(hours=1)
            idx2 = all_timestamps.searchsorted(next_ts)
            if idx2 >= len(all_timestamps):
                break
            current_ts = all_timestamps.iloc[idx2]
            continue

        if cash < contract["strike"] * LOT_SIZE:
            skipped_no_margin += 1
            next_ts = current_ts + pd.Timedelta(hours=1)
            idx2 = all_timestamps.searchsorted(next_ts)
            if idx2 >= len(all_timestamps):
                break
            current_ts = all_timestamps.iloc[idx2]
            continue

        # Open position: receive premium net of fees
        gross_premium = contract["premium_usd"] * LOT_SIZE
        fee_open = min(
            contract["underlying_price"] * LOT_SIZE * WHEEL_FEE_OPEN_RATE,
            gross_premium * 0.125       # Deribit cap: max 12.5% of premium
        )
        net_premium = gross_premium - fee_open
        cash += net_premium

        # Wait for expiry
        exp_ts  = contract["expiration_ts"]
        exp_idx = all_timestamps.searchsorted(exp_ts)
        if exp_idx >= len(all_timestamps):
            break

        actual_exp_ts = all_timestamps.iloc[exp_idx]
        spot_exp_rows = puts[puts["timestamp"] == actual_exp_ts]["underlying_price"]
        if spot_exp_rows.empty:
            break
        spot_at_exp = float(spot_exp_rows.iloc[0])

        assigned = spot_at_exp < contract["strike"]
        if assigned:
            fee_delivery = contract["strike"] * LOT_SIZE * WHEEL_FEE_DELIVERY_RATE
            cash -= contract["strike"] * LOT_SIZE + fee_delivery
            eth  += LOT_SIZE

        portfolio = cash + eth * spot_at_exp

        rows.append({
            "ts":              actual_exp_ts,
            "open_ts":         contract["open_ts"],
            "instrument":      contract["instrument_name"],
            "actual_dte":      contract["actual_dte"],
            "strike":          contract["strike"],
            "spot_at_open":    contract["spot_at_open"],
            "premium_usd":     round(net_premium, 2),
            "fee_open":        round(fee_open, 4),
            "assigned":        assigned,
            "spot_at_exp":     spot_at_exp,
            "cash":            round(cash, 2),
            "eth":             eth,
            "portfolio_value": round(portfolio, 2),
        })

        next_idx = exp_idx + 1
        if next_idx >= len(all_timestamps):
            break
        current_ts = all_timestamps.iloc[next_idx]

    df = pd.DataFrame(rows)
    assigned_count = df["assigned"].sum()
    print(f"\nTotal cycles:          {len(df)}")
    print(f"Assigned (ETH):        {assigned_count} ({100*assigned_count/len(df):.1f}%)")
    print(f"Worthless:             {len(df)-assigned_count} ({100*(1-assigned_count/len(df)):.1f}%)")
    print(f"Skipped (no expiry):   {skipped_no_exp}")
    print(f"Skipped (no margin):   {skipped_no_margin}")
    return df


# -----------------------------------------------------------------------------
# STEP 4: DCA BACKTEST
# -----------------------------------------------------------------------------

def calc_dca_pnl(spot_series: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates DCA strategy: buy ETH at fixed USD intervals.
    Uses an independent spot series so purchases are not affected
    by option data availability or DTE filtering.
    """
    freq = "W-FRI" if DCA_INTERVAL == "weekly" else "MS"
    purchase_dates = spot_series.resample(freq).first().dropna()

    cash = float(INITIAL_CAPITAL_USD)
    eth  = 0.0
    rows = []

    for ts, row in purchase_dates.iterrows():
        spot       = row["underlying_price"]
        amount     = min(DCA_AMOUNT_USD, max(cash, 0))
        fee_dca    = amount * DCA_FEE_RATE
        eth_bought = (amount - fee_dca) / spot if amount > 0 else 0.0
        cash      -= amount
        eth       += eth_bought

        rows.append({
            "ts":              ts,
            "spot":            spot,
            "eth_bought":      round(eth_bought, 6),
            "cash":            round(cash, 2),
            "eth":             round(eth, 6),
            "portfolio_value": round(max(cash, 0) + eth * spot, 2),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# STEP 5: PRINT RESULTS
# -----------------------------------------------------------------------------

def print_csp_log(csp_pnl: pd.DataFrame) -> None:
    print(f"\n=== CSP CYCLE LOG ===")
    print(f"  Moneyness={STRIKE_MONEYNESS} | DTE={TARGET_DTE}±{DTE_TOLERANCE} | Lot={LOT_SIZE} ETH\n")
    print(f"  {'Opened':<18} {'Contract':<24} {'Strike':>7} {'Premium$':>9} {'Assigned':>8} {'Spot@Exp':>9} {'Cash':>10} {'ETH':>5} {'Portfolio':>12}")
    print("  " + "-" * 115)
    for _, r in csp_pnl.iterrows():
        print(
            f"  {r['open_ts'].strftime('%Y-%m-%d %H:%M'):<18} "
            f"{r['instrument']:<24} "
            f"{r['strike']:>7.0f} "
            f"{r['premium_usd']:>9.2f} "
            f"{'YES' if r['assigned'] else 'no':>8} "
            f"{r['spot_at_exp']:>9.0f} "
            f"{r['cash']:>10.2f} "
            f"{r['eth']:>5.1f} "
            f"{r['portfolio_value']:>12.2f}"
        )


def print_dca_log(dca_pnl: pd.DataFrame) -> None:
    print(f"\n=== DCA LOG ({DCA_INTERVAL}, ${DCA_AMOUNT_USD}/purchase) ===\n")
    print(f"  {'Date':<18} {'Spot':>8} {'ETH bought':>12} {'ETH total':>10} {'Cash':>10} {'Portfolio':>12}")
    print("  " + "-" * 75)
    for _, r in dca_pnl.iterrows():
        print(
            f"  {r['ts'].strftime('%Y-%m-%d %H:%M'):<18} "
            f"{r['spot']:>8.0f} "
            f"{r['eth_bought']:>12.6f} "
            f"{r['eth']:>10.6f} "
            f"{r['cash']:>10.2f} "
            f"{r['portfolio_value']:>12.2f}"
        )
    print(f"\n  Total purchases: {len(dca_pnl)}")


def print_summary(csp_pnl: pd.DataFrame, dca_pnl: pd.DataFrame) -> None:
    start_val = INITIAL_CAPITAL_USD

    w_end   = csp_pnl["portfolio_value"].iloc[-1]
    w_eth   = csp_pnl["eth"].iloc[-1]
    w_cash  = csp_pnl["cash"].iloc[-1]
    w_ret   = (w_end - start_val) / start_val * 100
    w_prm   = csp_pnl["premium_usd"].sum()

    assigned_rows = csp_pnl[csp_pnl["assigned"]]
    w_spent_eth   = (assigned_rows["strike"] * LOT_SIZE).sum() if len(assigned_rows) > 0 else 0
    w_avg_price   = w_spent_eth / w_eth if w_eth > 0 else 0

    d_end   = dca_pnl["portfolio_value"].iloc[-1]
    d_eth   = dca_pnl["eth"].iloc[-1]
    d_cash  = dca_pnl["cash"].iloc[-1]
    d_ret   = (d_end - start_val) / start_val * 100
    d_spent = INITIAL_CAPITAL_USD - d_cash
    d_avg   = d_spent / d_eth if d_eth > 0 else 0

    print("\n" + "=" * 57)
    print("  BACKTEST RESULTS")
    print("=" * 57)
    print(f"  Starting capital:      ${start_val:>10,.2f}")
    print("-" * 57)
    print(f"  CASH-SECURED PUT (moneyness={STRIKE_MONEYNESS}, DTE={TARGET_DTE}±{DTE_TOLERANCE})")
    print(f"    Portfolio value:     ${w_end:>10,.2f}  ({w_ret:+.1f}%)")
    print(f"    ETH accumulated:      {w_eth:>10.4f}")
    print(f"    Avg ETH buy price:   ${w_avg_price:>10,.2f}")
    print(f"    Cash remaining:      ${w_cash:>10,.2f}")
    print(f"    Total premiums (net):${w_prm:>10,.2f}")
    print("-" * 57)
    print(f"  DCA ({DCA_INTERVAL}, ${DCA_AMOUNT_USD}/purchase)")
    print(f"    Portfolio value:     ${d_end:>10,.2f}  ({d_ret:+.1f}%)")
    print(f"    ETH accumulated:      {d_eth:>10.4f}")
    print(f"    Avg ETH buy price:   ${d_avg:>10,.2f}")
    print(f"    Cash remaining:      ${d_cash:>10,.2f}")
    print(f"    Spent on ETH:        ${d_spent:>10,.2f}")
    print("=" * 57)


# -----------------------------------------------------------------------------
# STEP 6: CHARTS
# -----------------------------------------------------------------------------

def plot_results(csp_pnl: pd.DataFrame, dca_pnl: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    BG       = "#0F1117"
    BG_AX    = "#161B27"
    CLR_W    = "#38BDF8"   # CSP — blue
    CLR_D    = "#F97316"   # DCA — orange
    CLR_ASN  = "#A78BFA"   # Assigned — purple
    CLR_ETH  = "#34D399"   # ETH spot — green
    CLR_GRID = "#1E2533"
    CLR_BASE = "#4B5563"
    CLR_TXT  = "#E2E8F0"
    CLR_SUB  = "#64748B"

    w_ts     = csp_pnl["ts"].dt.tz_localize(None)
    d_ts     = dca_pnl["ts"].dt.tz_localize(None)
    assigned = csp_pnl[csp_pnl["assigned"]]

    w_ret = (csp_pnl["portfolio_value"].iloc[-1] - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD * 100
    d_ret = (dca_pnl["portfolio_value"].iloc[-1] - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD * 100

    fig = plt.figure(figsize=(14, 16), facecolor=BG)
    gs  = fig.add_gridspec(3, 1, hspace=0.45, left=0.08, right=0.94, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    fig.text(0.08, 0.965, "Cash-Secured Put vs DCA — ETH Backtest",
             color=CLR_TXT, fontsize=16, fontweight="bold", va="top")
    fig.text(0.08, 0.948,
             f"CSP: moneyness {STRIKE_MONEYNESS}x ATM · DTE {TARGET_DTE}±{DTE_TOLERANCE}d · lot {LOT_SIZE} ETH     "
             f"DCA: ${DCA_AMOUNT_USD}/{DCA_INTERVAL}     "
             f"Capital: ${INITIAL_CAPITAL_USD:,}",
             color=CLR_SUB, fontsize=9, va="top")

    def style_ax(ax, title, ylabel):
        ax.set_facecolor(BG_AX)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=CLR_SUB, labelsize=8.5)
        ax.grid(True, color=CLR_GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", color=CLR_SUB, fontsize=8.5)
        ax.text(0, 1.04, title, transform=ax.transAxes,
                color=CLR_TXT, fontsize=10, fontweight="bold", va="bottom")
        ax.set_ylabel(ylabel, color=CLR_SUB, fontsize=9)

    # ── 1. Portfolio Value ─────────────────────────────────────────────────────
    ax = axes[0]
    style_ax(ax, "Portfolio Value", "USD")

    eth_spot = dca_pnl[["ts", "spot"]].copy()
    eth_spot["ts"] = eth_spot["ts"].dt.tz_localize(None)
    eth_start = eth_spot["spot"].iloc[0]
    eth_spot["value"] = eth_spot["spot"] / eth_start * INITIAL_CAPITAL_USD
    eth_ret = (eth_spot["spot"].iloc[-1] - eth_start) / eth_start * 100

    ax.axhline(INITIAL_CAPITAL_USD, color=CLR_BASE, linewidth=0.9, linestyle="--", zorder=1)
    ax.fill_between(w_ts, INITIAL_CAPITAL_USD, csp_pnl["portfolio_value"],
                    where=csp_pnl["portfolio_value"] >= INITIAL_CAPITAL_USD,
                    color=CLR_W, alpha=0.08, zorder=1)
    ax.fill_between(w_ts, INITIAL_CAPITAL_USD, csp_pnl["portfolio_value"],
                    where=csp_pnl["portfolio_value"] < INITIAL_CAPITAL_USD,
                    color="#EF4444", alpha=0.08, zorder=1)

    ax.plot(w_ts, csp_pnl["portfolio_value"], color=CLR_W, linewidth=2.2, zorder=3, solid_capstyle="round")
    ax.plot(d_ts, dca_pnl["portfolio_value"], color=CLR_D, linewidth=2.2, zorder=3, solid_capstyle="round")
    ax.plot(eth_spot["ts"], eth_spot["value"], color=CLR_ETH, linewidth=1.5,
            zorder=2, linestyle="--", alpha=0.85, solid_capstyle="round")
    ax.scatter(assigned["ts"].dt.tz_localize(None), assigned["portfolio_value"],
               color=CLR_ASN, s=70, zorder=5, marker="D", linewidths=0)

    ax.annotate(f"${csp_pnl['portfolio_value'].iloc[-1]:,.0f}  ({w_ret:+.1f}%)",
                xy=(w_ts.iloc[-1], csp_pnl["portfolio_value"].iloc[-1]),
                xytext=(8, 4), textcoords="offset points",
                color=CLR_W, fontsize=8.5, fontweight="bold", va="center")
    ax.annotate(f"${dca_pnl['portfolio_value'].iloc[-1]:,.0f}  ({d_ret:+.1f}%)",
                xy=(d_ts.iloc[-1], dca_pnl["portfolio_value"].iloc[-1]),
                xytext=(8, -4), textcoords="offset points",
                color=CLR_D, fontsize=8.5, fontweight="bold", va="center")
    ax.annotate(f"ETH  ({eth_ret:+.1f}%)",
                xy=(eth_spot["ts"].iloc[-1], eth_spot["value"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=CLR_ETH, fontsize=8.5, fontweight="bold", va="center")

    ax.legend(handles=[
        Line2D([0], [0], color=CLR_W,    linewidth=2,   label="Cash-Secured Put"),
        Line2D([0], [0], color=CLR_D,    linewidth=2,   label="DCA"),
        Line2D([0], [0], color=CLR_ETH,  linewidth=1.5, linestyle="--", label="ETH spot (normalized)"),
        Line2D([0], [0], color=CLR_ASN,  marker="D", markersize=6, linewidth=0, label="Assigned"),
        Line2D([0], [0], color=CLR_BASE, linewidth=1,   linestyle="--", label="Capital"),
    ], fontsize=8.5, framealpha=0, labelcolor=CLR_SUB, loc="upper left")

    # ── 2. ETH Accumulated ────────────────────────────────────────────────────
    ax = axes[1]
    style_ax(ax, "ETH Accumulated", "ETH")

    ax.fill_between(w_ts, 0, csp_pnl["eth"], step="post", color=CLR_W, alpha=0.12, zorder=1)
    ax.fill_between(d_ts, 0, dca_pnl["eth"], step="post", color=CLR_D, alpha=0.10, zorder=1)
    ax.step(w_ts, csp_pnl["eth"], color=CLR_W, linewidth=2.2, where="post", zorder=3)
    ax.step(d_ts, dca_pnl["eth"], color=CLR_D, linewidth=2.2, where="post", zorder=3)
    ax.scatter(assigned["ts"].dt.tz_localize(None), assigned["eth"],
               color=CLR_ASN, s=70, zorder=5, marker="D", linewidths=0)

    ax.annotate(f"{csp_pnl['eth'].iloc[-1]:.3f} ETH",
                xy=(w_ts.iloc[-1], csp_pnl["eth"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=CLR_W, fontsize=8.5, fontweight="bold", va="center")
    ax.annotate(f"{dca_pnl['eth'].iloc[-1]:.4f} ETH",
                xy=(d_ts.iloc[-1], dca_pnl["eth"].iloc[-1]),
                xytext=(8, -10), textcoords="offset points",
                color=CLR_D, fontsize=8.5, fontweight="bold", va="center")

    # ── 3. Cash Remaining ─────────────────────────────────────────────────────
    ax = axes[2]
    style_ax(ax, "Cash Remaining", "USD")

    ax.fill_between(w_ts, 0, csp_pnl["cash"], color=CLR_W, alpha=0.10, zorder=1)
    ax.fill_between(d_ts, 0, dca_pnl["cash"], color=CLR_D, alpha=0.08, zorder=1)
    ax.plot(w_ts, csp_pnl["cash"], color=CLR_W, linewidth=2.2, zorder=3)
    ax.plot(d_ts, dca_pnl["cash"], color=CLR_D, linewidth=2.2, zorder=3)
    ax.axhline(0, color="#EF4444", linewidth=0.7, linestyle=":", zorder=2)

    ax.annotate(f"${csp_pnl['cash'].iloc[-1]:,.0f}",
                xy=(w_ts.iloc[-1], csp_pnl["cash"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=CLR_W, fontsize=8.5, fontweight="bold", va="center")
    ax.annotate(f"${dca_pnl['cash'].iloc[-1]:,.0f}",
                xy=(d_ts.iloc[-1], dca_pnl["cash"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=CLR_D, fontsize=8.5, fontweight="bold", va="center")

    out_path = DATA_DIR / "backtest_results.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"\nChart saved: {out_path}")
    plt.show()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== STEP 1: LOADING DATA ===\n")
    puts = load_puts_only(DATA_FILES)
    get_expiry_schedule(puts)

    print("\n  Loading spot series for DCA...")
    spot_series = load_spot_series(DATA_FILES)

    print("\n=== STEP 2: CASH-SECURED PUT P&L ===")
    csp_pnl = build_csp_pnl(puts)
    print_csp_log(csp_pnl)

    print("\n=== STEP 3: DCA P&L ===")
    dca_pnl = calc_dca_pnl(spot_series)
    print_dca_log(dca_pnl)

    print_summary(csp_pnl, dca_pnl)

    plot_results(csp_pnl, dca_pnl)

    print("\n✓ Done.")
