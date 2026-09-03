"""
dashboard.py
Interactive dashboard: pricing, Greeks, method comparison, volatility
surface, and delta-hedging backtest - all in one place.

Mirrors the trading-engine's week7.py dashboard style (page layout,
sidebar-driven controls, metric cards, matplotlib charts via st.pyplot),
so the two projects visually read as a matched pair.

Week 10 of options-engine project
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from bs_model import black_scholes
from greeks import delta, gamma, vega, theta, rho
from binomial_tree import binomial_tree_price
from monte_carlo import monte_carlo_price
from vol_surface import fetch_option_chain, build_vol_surface
from delta_hedge_backtest import (
    fetch_price_history,
    realized_volatility,
    run_delta_hedge_backtest,
)

st.set_page_config(page_title="Options Pricing Engine", layout="wide")
st.title("Options Pricing Engine")
st.markdown(
    "Built by Lucy Pakenham | Black-Scholes, Greeks, Binomial Tree, "
    "Monte Carlo & Delta-Hedging"
)

# --- Sidebar: Option Parameters (shared across Pricing/Greeks/Comparison tabs) ---
st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Stock Price (S)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T_days = st.sidebar.slider("Days to Expiry", 1, 365, 90)
T = T_days / 365
r_pct = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0) / 100
sigma_pct = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 25.0) / 100
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

r = r_pct
sigma = sigma_pct

# --- Sidebar: Volatility Surface controls ---
st.sidebar.header("Volatility Surface")
vs_ticker = st.sidebar.text_input("Ticker (Vol Surface)", value="AAPL").upper()
vs_max_expiries = st.sidebar.slider("Max Expiries", 1, 10, 6)

# --- Sidebar: Delta-Hedging Backtest controls ---
st.sidebar.header("Delta-Hedging Backtest")
hedge_ticker = st.sidebar.text_input("Ticker (Hedging)", value="AAPL").upper()
hedge_window = st.sidebar.slider("Hedge Window (days)", 10, 120, 60)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Pricing & Greeks",
        "Method Comparison",
        "Volatility Surface",
        "Delta-Hedging Backtest",
    ]
)

# ============================================================
# TAB 1: Pricing & Greeks
# ============================================================
with tab1:
    st.subheader("Black-Scholes Price & Greeks")

    price = black_scholes(S, K, T, r, sigma, option_type)
    d = delta(S, K, T, r, sigma, option_type)
    g = gamma(S, K, T, r, sigma)
    v = vega(S, K, T, r, sigma)
    th = theta(S, K, T, r, sigma, option_type)
    rh = rho(S, K, T, r, sigma, option_type)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Price", f"${price:.4f}")
    col2.metric("Delta", f"{d:.4f}")
    col3.metric("Gamma", f"{g:.4f}")
    col4.metric("Vega", f"{v:.4f}")
    col5.metric("Theta", f"{th:.4f}")
    col6.metric("Rho", f"{rh:.4f}")

    st.subheader("Payoff at Expiry")
    S_range = np.linspace(S * 0.5, S * 1.5, 100)
    if option_type == "call":
        payoff = np.maximum(S_range - K, 0) - price
    else:
        payoff = np.maximum(K - S_range, 0) - price

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(S_range, payoff, color="steelblue", label="P&L at expiry")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.axvline(S, color="orange", linestyle="--", label=f"Current price (${S:.2f})")
    ax1.set_xlabel("Stock Price at Expiry")
    ax1.set_ylabel("P&L (USD)")
    ax1.legend()
    st.pyplot(fig1)

# ============================================================
# TAB 2: Method Comparison
# ============================================================
with tab2:
    st.subheader("Black-Scholes vs Binomial Tree vs Monte Carlo")

    n_steps = st.slider("Binomial Tree Steps (N)", 50, 2000, 200)
    n_sims = st.slider("Monte Carlo Simulations", 1000, 200000, 50000, step=1000)

    with st.spinner("Pricing across all three methods..."):
        bs_price = black_scholes(S, K, T, r, sigma, option_type)
        tree_euro = binomial_tree_price(
            S, K, T, r, sigma, N=n_steps, option_type=option_type, american=False
        )
        tree_amer = binomial_tree_price(
            S, K, T, r, sigma, N=n_steps, option_type=option_type, american=True
        )
        mc_price, mc_se = monte_carlo_price(
            S, K, T, r, sigma, option_type, n_sims=n_sims, seed=42
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Black-Scholes", f"${bs_price:.4f}")
    col2.metric("Binomial (European)", f"${tree_euro:.4f}")
    col3.metric("Binomial (American)", f"${tree_amer:.4f}")
    col4.metric("Monte Carlo", f"${mc_price:.4f}", f"±{mc_se:.4f} SE")

    st.caption(
        "European tree and Monte Carlo should sit close to the Black-Scholes "
        "price; the American tree price can only be equal or higher, since "
        "early exercise is never a disadvantage."
    )

# ============================================================
# TAB 3: Volatility Surface
# ============================================================
with tab3:
    st.subheader("Implied Volatility Surface")

    if st.button("Fetch Option Chain & Build Surface"):
        with st.spinner(f"Fetching option chain for {vs_ticker}..."):
            try:
                S_spot, rows = fetch_option_chain(
                    vs_ticker, max_expiries=vs_max_expiries
                )
                surface_points = build_vol_surface(S_spot, rows)
            except Exception as e:
                st.error(f"Could not fetch or process option chain: {e}")
                st.stop()

        if not surface_points:
            st.warning("No usable option data found for this ticker.")
        else:
            st.success(
                f"Current price: ${S_spot:.2f} | "
                f"{len(surface_points)} implied vols computed from {len(rows)} raw quotes"
            )

            expiries = sorted(set(T_pt for K_pt, T_pt, iv in surface_points))
            selected_T = st.selectbox(
                "Expiry (years to expiry)", expiries, format_func=lambda x: f"{x:.3f}"
            )

            smile = sorted(
                [(K_pt, iv) for K_pt, T_pt, iv in surface_points if T_pt == selected_T]
            )
            strikes = [k for k, iv in smile]
            ivs = [iv for k, iv in smile]

            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(strikes, ivs, marker="o", color="steelblue")
            ax2.axvline(
                S_spot, color="orange", linestyle="--", label=f"Spot (${S_spot:.2f})"
            )
            ax2.set_xlabel("Strike")
            ax2.set_ylabel("Implied Volatility")
            ax2.legend()
            st.pyplot(fig2)

            fig3 = plt.figure(figsize=(10, 6))
            ax3 = fig3.add_subplot(111, projection="3d")
            all_strikes = [p[0] for p in surface_points]
            all_Ts = [p[1] for p in surface_points]
            all_ivs = [p[2] for p in surface_points]
            ax3.scatter(all_strikes, all_Ts, all_ivs, c=all_ivs, cmap="viridis")
            ax3.set_xlabel("Strike")
            ax3.set_ylabel("Time to expiry (years)")
            ax3.set_zlabel("Implied Volatility")
            st.pyplot(fig3)
    else:
        st.info(
            "Click the button above to fetch live option data and build the surface."
        )

# ============================================================
# TAB 4: Delta-Hedging Backtest
# ============================================================
with tab4:
    st.subheader("Delta-Hedging Backtest")

    if st.button("Run Delta-Hedging Backtest"):
        with st.spinner(f"Fetching price history for {hedge_ticker}..."):
            try:
                prices = fetch_price_history(hedge_ticker)
                hedge_sigma = realized_volatility(prices)
                results = run_delta_hedge_backtest(prices, hedge_sigma)
            except Exception as e:
                st.error(f"Could not run backtest: {e}")
                st.stop()

        hedged_changes = results["hedged_portfolio"].diff().dropna()
        unhedged_changes = results["unhedged_portfolio"].diff().dropna()
        variance_reduction = (1 - hedged_changes.std() / unhedged_changes.std()) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Realized Volatility Used", f"{hedge_sigma:.2%}")
        col2.metric("Final Hedged P&L", f"${results['hedged_portfolio'].iloc[-1]:.2f}")
        col3.metric(
            "Final Unhedged P&L", f"${results['unhedged_portfolio'].iloc[-1]:.2f}"
        )
        col4.metric("Variance Reduction", f"{variance_reduction:.1f}%")

        st.subheader("P&L Over Time")
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        ax4.plot(
            results["date"],
            results["hedged_portfolio"],
            label="Delta-Hedged",
            color="steelblue",
        )
        ax4.plot(
            results["date"],
            results["unhedged_portfolio"],
            label="Unhedged (Naked Short Call)",
            color="orange",
            linestyle="--",
        )
        ax4.set_ylabel("P&L (USD)")
        ax4.legend()
        st.pyplot(fig4)

        st.subheader("Daily Hedging Error")
        fig5, ax5 = plt.subplots(figsize=(12, 3))
        daily_error = results["hedged_portfolio"].diff()
        ax5.plot(results["date"], daily_error, color="steelblue")
        ax5.axhline(0, color="gray", linewidth=0.8)
        ax5.set_ylabel("Daily P&L change (USD)")
        st.pyplot(fig5)
    else:
        st.info(
            "Click the button above to run the delta-hedging backtest on real market data."
        )
