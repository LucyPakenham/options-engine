"""
vol_surface.py
Fetches real option chain data via yfinance, computes implied volatility
across strikes and expiries using the Week 4 solver, and visualizes the
resulting volatility smile/surface.

Week 8 of options-engine project
"""

import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed to enable 3D projection
import yfinance as yf

from implied_vol import implied_volatility

RISK_FREE_RATE = 0.03  # simplification: one constant rate across all expiries

import math


def fetch_option_chain(
    ticker_symbol,
    max_expiries=6,
    min_days_to_expiry=10,
    moneyness_band=0.4,
    min_open_interest=10,
    min_volume=1,
    min_price=0.10,
):
    """
    Pull real option chain data for a ticker across several expiries.

    Uses OTM options only (puts below spot, calls above spot) - the most
    liquid, and avoids mixing calls/puts whose implied vols can genuinely
    diverge on ITM contracts (e.g. American early-exercise premium on ITM
    puts, which the European solver has no way to model).

    min_days_to_expiry: skips very short-dated options (near-zero vega).
    moneyness_band: only keep strikes within this fraction of spot.
    min_open_interest / min_volume: skip illiquid contracts.
    min_price: skips options too cheap to trust (rounding noise dominates).

    Returns (current_price, list_of_quote_dicts).
    """
    ticker = yf.Ticker(ticker_symbol)
    S = ticker.history(period="1d")["Close"].iloc[-1]

    expiries = ticker.options[:max_expiries]
    today = dt.date.today()

    rows = []
    for expiry_str in expiries:
        expiry_date = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
        days_to_expiry = (expiry_date - today).days
        if days_to_expiry < min_days_to_expiry:
            continue

        T = days_to_expiry / 365
        chain = ticker.option_chain(expiry_str)

        for df, option_type in [(chain.calls, "call"), (chain.puts, "put")]:
            for _, row in df.iterrows():
                strike = row["strike"]
                if strike < S * (1 - moneyness_band) or strike > S * (
                    1 + moneyness_band
                ):
                    continue

                if strike < S and option_type != "put":
                    continue
                if strike >= S and option_type != "call":
                    continue

                open_interest = row.get("openInterest", 0) or 0
                volume = row.get("volume", 0) or 0
                if open_interest < min_open_interest and volume < min_volume:
                    continue

                bid = row.get("bid", 0) or 0
                ask = row.get("ask", 0) or 0
                last = row.get("lastPrice", 0) or 0

                if bid > 0 and ask > 0:
                    market_price = (bid + ask) / 2
                elif last > 0:
                    market_price = last
                else:
                    continue

                if market_price < min_price:
                    continue

                rows.append(
                    {
                        "strike": strike,
                        "expiry": expiry_str,
                        "T": T,
                        "market_price": market_price,
                        "option_type": option_type,
                    }
                )

    return S, rows


def build_vol_surface(S, rows):
    """
    Run the Week 4 implied vol solver on every option chain row.
    Returns a list of (strike, T, implied_vol) points, skipping failures.
    """
    surface_points = []

    for row in rows:
        try:
            iv = implied_volatility(
                row["market_price"],
                S,
                row["strike"],
                row["T"],
                RISK_FREE_RATE,
                row["option_type"],
            )
            if 0.01 < iv < 3.0:  # sanity filter: discard nonsensical solver results
                surface_points.append((row["strike"], row["T"], iv))
        except (ValueError, ZeroDivisionError):
            continue  # bad or inconsistent quote - skip rather than crash the whole run

    return surface_points


def plot_smile(surface_points, S, target_T, tolerance=0.05):
    """Plot implied vol vs strike for options near one specific expiry."""
    smile = sorted(
        [(K, iv) for K, T, iv in surface_points if abs(T - target_T) < tolerance]
    )
    if not smile:
        print(f"No data close to T={target_T:.3f}")
        return

    strikes = [K for K, iv in smile]
    ivs = [iv for K, iv in smile]

    plt.figure(figsize=(8, 5))
    plt.plot(strikes, ivs, marker="o")
    plt.axvline(S, color="gray", linestyle="--", label=f"Spot price ({S:.2f})")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title(f"Volatility Smile (T ≈ {target_T:.3f} years)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vol_smile.png")
    print("Saved vol_smile.png")


def plot_surface(surface_points):
    """Plot the full implied volatility surface across strikes and expiries."""
    strikes = [p[0] for p in surface_points]
    Ts = [p[1] for p in surface_points]
    ivs = [p[2] for p in surface_points]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(strikes, Ts, ivs, c=ivs, cmap="viridis")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiry (years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Implied Volatility Surface")
    plt.tight_layout()
    plt.savefig("vol_surface.png")
    print("Saved vol_surface.png")


if __name__ == "__main__":
    TICKER = "AAPL"

    print(f"Fetching option chain for {TICKER}...")
    S, rows = fetch_option_chain(TICKER)
    print(f"Current price: {S:.2f}")
    print(f"Fetched {len(rows)} raw option quotes")

    surface_points = build_vol_surface(S, rows)
    skipped = len(rows) - len(surface_points)
    print(
        f"Successfully computed implied vol for {len(surface_points)} options "
        f"({skipped} skipped due to bad quotes or solver failures)"
    )

    if surface_points:
        nearest_expiry = sorted(set(T for K, T, iv in surface_points))[0]
        plot_smile(surface_points, S, nearest_expiry)
        plot_surface(surface_points)
