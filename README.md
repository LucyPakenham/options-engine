# Options Pricing Engine

A full options pricing and risk-management system built in Python and C++ — Black-Scholes, the Greeks, implied volatility, binomial trees, Monte Carlo simulation, a real market-data volatility surface, and a delta-hedging backtest. Built over 10 weeks as a companion project to [trading-engine](https://github.com/LucyPakenham/trading-engine).

## Live Demo

Run the interactive dashboard locally:

```bash
pip install -r requirements.txt
python -m streamlit run dashboard.py
```

## Project Structure

```

options-engine/
├── bs_model.py # Black-Scholes pricing (calls & puts)
├── greeks.py # Delta, gamma, vega, theta, rho
├── implied_vol.py # Implied volatility (Newton-Raphson + bisection fallback)
├── binomial_tree.py # CRR binomial tree (European & American exercise)
├── monte_carlo.py # Monte Carlo simulation (pure Python)
├── mc_pricer.cpp # Monte Carlo core, compiled to a DLL
├── monte_carlo_cpp.py # Python ctypes bindings to the C++ core
├── benchmark.py # Python vs C++ performance benchmark
├── vol_surface.py # Real implied volatility surface (yfinance option chains)
├── delta_hedge_backtest.py # Delta-hedging backtest on real historical prices
├── dashboard.py # Interactive Streamlit dashboard (all of the above, in one place)
├── options_engine_notes.md # Detailed weekly build notes and design rationale
└── vol_smile.png, vol_surface.png, delta_hedge_backtest.png
```

## Features

- Black-Scholes pricing for European calls and puts
- Full Greeks — delta, gamma, vega, theta, rho
- Implied volatility solver — Newton-Raphson with a bisection fallback for numerical robustness
- CRR binomial tree — prices American options with early exercise, verified against Black-Scholes
- Monte Carlo simulation in Python and C++ — **~15–21x speedup** from the compiled core
- A real implied volatility surface built from live AAPL option chains, showing the equity volatility skew
- A delta-hedging backtest on real historical prices — **~89% reduction in P&L variance** versus an unhedged position
- One Streamlit dashboard tying all of it together

## Results

**Monte Carlo: Python vs C++**

| N Simulations | Python (s) | C++ (s) | Speedup |
|---|---|---|---|
| 10,000 | 0.0064 | 0.0003 | 20.7x |
| 100,000 | 0.0644 | 0.0042 | 15.2x |
| 1,000,000 | 0.6776 | 0.0317 | 21.4x |
| 5,000,000 | 3.4807 | 0.1690 | 20.6x |

**Delta-Hedging Backtest (AAPL, 60-day window)**

| Metric | Hedged | Unhedged |
|---|---|---|
| Final P&L | -$3.45 | -$7.86 |
| Daily P&L Std Dev | $0.48 | $4.31 |
| **Variance Reduction** | **88.9%** | — |

## Tech Stack

- Python — `math` (no `scipy` dependency, for Windows compatibility), `numpy`, `pandas`, `matplotlib`, `yfinance`, `Streamlit`
- C++ — Monte Carlo core compiled as a shared library (`.dll`), `std::mt19937_64` for random number generation
- `ctypes` — Python bindings to call the C++ core directly

## Key Learnings

- How to numerically solve for implied volatility, and why Newton-Raphson needs a more robust fallback method
- How the CRR binomial tree prices American options and why early exercise is never optimal for a call without dividends
- How to profile and accelerate a numerical Python bottleneck with a compiled C++ core
- How to work with real, messy market data — illiquid quotes, stale prices, and the difference between European and American implied vols — and how to systematically debug it rather than just filtering blindly
- How delta-hedging reduces directional risk, and where its real limitation (gamma risk) shows up in practice
- How to structure two related projects so they read as a coherent, matched body of work

See [`options_engine_notes.md`](options_engine_notes.md) for detailed week-by-week design rationale, debugging notes, and interview talking points.

## Built By

Lucy Pakenham — Engineering Student, Trinity College Dublin
github.com/LucyPakenham