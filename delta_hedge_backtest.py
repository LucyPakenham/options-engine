"""
delta_hedge_backtest.py
Delta-hedging backtest: simulates writing (selling) one ATM call option and
dynamically hedging the resulting exposure using the Week 3 Greeks, over
real historical price data.

Reuses the trading-engine's day-by-day backtest loop structure and
transaction cost convention (TRANSACTION_COST = 0.001), and adapts its
risk-metrics style where it genuinely fits.

Week 9 of options-engine project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from bs_model import black_scholes
from greeks import delta as bs_delta

TICKER = "AAPL"
LOOKBACK_DAYS = 252  # historical window used to estimate volatility
HEDGE_WINDOW_DAYS = 60  # trading days the option position is held for
TRANSACTION_COST = 0.001  # 0.1% per trade - same convention as trading-engine
RISK_FREE_RATE = 0.03
TRADING_DAYS_PER_YEAR = 252


def fetch_price_history(ticker_symbol):
    """Same download/reshape pattern as trading-engine's Week4.py and week6.py."""
    raw = yf.download(ticker_symbol, period="2y")
    data = raw["Close"].squeeze()
    data = pd.DataFrame(data)
    data.columns = ["Close"]
    return data


def realized_volatility(prices, window=LOOKBACK_DAYS):
    """
    Annualized realized volatility from historical log returns - used as the
    constant sigma input for daily delta calculations. Black-Scholes assumes
    one fixed volatility (Week 1-2); real markets don't actually hold it
    constant (Week 8's smile/skew) - this backtest deliberately keeps that
    simplification, since the point here is testing the hedge mechanics, not
    re-litigating the vol surface.
    """
    log_returns = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
    daily_vol = log_returns.tail(window).std()
    return float(daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def run_delta_hedge_backtest(prices, sigma):
    """
    Simulate writing one ATM call and dynamically delta-hedging it over
    HEDGE_WINDOW_DAYS trading days. Returns a DataFrame with day-by-day
    state for both the hedged and unhedged (naked short) positions.
    """
    window = prices.tail(HEDGE_WINDOW_DAYS + 1).copy()
    S0 = window["Close"].iloc[0]
    K = S0  # ATM strike at initiation
    T_total = (len(window) - 1) / TRADING_DAYS_PER_YEAR

    # --- Initiate: sell 1 call, receive the premium ---
    premium = black_scholes(S0, K, T_total, RISK_FREE_RATE, sigma, "call")
    cash_hedged = premium
    shares_held = 0.0

    records = []

    for day_idx, (date, row) in enumerate(window.iterrows()):
        S = row["Close"]
        T_remaining = max((len(window) - 1 - day_idx) / TRADING_DAYS_PER_YEAR, 1e-6)

        option_value = black_scholes(S, K, T_remaining, RISK_FREE_RATE, sigma, "call")

        if T_remaining > 1e-4:
            current_delta = bs_delta(S, K, T_remaining, RISK_FREE_RATE, sigma, "call")
        else:
            current_delta = 1.0 if S > K else 0.0  # at expiry: fully hedged or flat

        # --- Rebalance the hedge to today's delta ---
        target_shares = current_delta
        trade_shares = target_shares - shares_held
        trade_value = abs(trade_shares) * S
        cash_hedged -= trade_shares * S
        cash_hedged -= (
            trade_value * TRANSACTION_COST
        )  # same convention as trading-engine
        shares_held = target_shares

        hedged_portfolio = cash_hedged + shares_held * S - option_value
        unhedged_portfolio = premium - option_value  # naked short call, no hedge at all

        records.append(
            {
                "date": date,
                "S": S,
                "T_remaining": T_remaining,
                "delta": current_delta,
                "option_value": option_value,
                "hedged_portfolio": hedged_portfolio,
                "unhedged_portfolio": unhedged_portfolio,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    print(f"Fetching price history for {TICKER}...")
    prices = fetch_price_history(TICKER)

    sigma = realized_volatility(prices)
    print(f"Estimated annualized realized volatility: {sigma:.2%}")

    results = run_delta_hedge_backtest(prices, sigma)

    hedged_changes = results["hedged_portfolio"].diff().dropna()
    unhedged_changes = results["unhedged_portfolio"].diff().dropna()

    print(f"\nFinal hedged P&L:   ${results['hedged_portfolio'].iloc[-1]:.2f}")
    print(f"Final unhedged P&L: ${results['unhedged_portfolio'].iloc[-1]:.2f}")
    print(f"\nHedged daily P&L std dev:   ${hedged_changes.std():.4f}")
    print(f"Unhedged daily P&L std dev: ${unhedged_changes.std():.4f}")
    print(
        f"Variance reduction from hedging: {(1 - hedged_changes.std() / unhedged_changes.std()) * 100:.1f}%"
    )

    # --- Plot: P&L curves + daily hedging error, mirroring trading-engine's week6.py layout ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(
        results["date"],
        results["hedged_portfolio"],
        label="Delta-Hedged",
        color="steelblue",
    )
    ax1.plot(
        results["date"],
        results["unhedged_portfolio"],
        label="Unhedged (Naked Short Call)",
        color="orange",
        linestyle="--",
    )
    ax1.set_title("Delta-Hedging Backtest: P&L Over Time")
    ax1.set_ylabel("P&L (USD)")
    ax1.legend()

    daily_hedging_error = results["hedged_portfolio"].diff()
    ax2.plot(results["date"], daily_hedging_error, color="steelblue")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_title("Daily Hedging Error (Hedged Portfolio)")
    ax2.set_ylabel("Daily P&L change (USD)")

    plt.tight_layout()
    plt.savefig("delta_hedge_backtest.png")
    print("\nSaved delta_hedge_backtest.png")
