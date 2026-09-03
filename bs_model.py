"""
bs_model.py
Black-Scholes option pricing model.
"""

import numpy as np
import math


def normal_cdf(x):
    """Standard normal cumulative distribution function (replaces scipy's norm.cdf)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes price of a European option.

    S     = current stock price
    K     = strike price
    T     = time to expiry, in years
    r     = risk-free interest rate (decimal, e.g. 0.05 = 5%)
    sigma = volatility (decimal, e.g. 0.25 = 25%)
    option_type = "call" or "put"
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * normal_cdf(d1) - K * np.exp(-r * T) * normal_cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


if __name__ == "__main__":
    S = 100
    K = 105
    T = 0.5
    r = 0.03
    sigma = 0.25

    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")

    print(f"Call price: {call_price:.4f}")
    print(f"Put price:  {put_price:.4f}")
