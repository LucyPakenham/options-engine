"""
greeks.py
Option Greeks: sensitivities of the Black-Scholes price to its inputs

Week 3 of options-engine project
"""

import numpy as np
import math
from bs_model import black_scholes


def normal_cdf(x):
    """Standard normal cumulative distribution function(replaces scipy's norm.cdf)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_pdf(x):
    """Standard normal probability density function."""
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)


def delta(S, K, T, r, sigma, option_type="call"):
    """
    Delta: rate of change of option price with respect to the underlying price

    Call delta is between 0 and 1.
    Put delta is between -1 and 0.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return normal_cdf(d1)
    elif option_type == "put":
        return normal_cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma):
    """
    Gamma: rate of change of delta with respect to the underlying price
    Same value for calls and puts.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return normal_pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Vega: rate of change of option price with the respect to volatility.
    Same value for calls and puts. Quoted per1% change is sigma, so we divide by 100.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * normal_pdf(d1) * np.sqrt(T) / 100


def theta(S, K, T, r, sigma, option_type="call"):
    """
    Theta: rate of change of option price with respect to time (time decay).
    Returned as decay per calender day annualisd value divided by 365.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        theta = -S * normal_pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(
            -r * T
        ) * normal_cdf(d2)
    elif option_type == "put":
        theta = -S * normal_pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(
            -r * T
        ) * normal_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return theta / 365  # Return daily decay


def rho(S, K, T, r, sigma, option_type="call"):
    """
    Rho: rate of change of option price with respect to the risk-free interest rate.
    Differs for calls and puts. Quoted per 1% change in r, so we divide by 100.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return K * T * np.exp(-r * T) * normal_cdf(d2) / 100
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * normal_cdf(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")


if __name__ == "__main__":
    # example option:stock at 100 dollars, strike at 100 dollars, 3months to expire
    # 3% risk-free rate, 25% volatility
    S = 100
    K = 100
    T = 0.25
    r = 0.03
    sigma = 0.25

    print("call Greeks:")
    print(f"delta: {delta(S, K, T, r, sigma, 'call'): .4f}")
    print(f"gamma: {gamma(S, K, T, r, sigma): .4f}")
    print(f"vega: {vega(S, K, T, r, sigma): .4f}")
    print(f"theta: {theta(S, K, T, r, sigma, 'call'): .4f}")
    print(f"rho: {rho(S, K, T, r, sigma, 'call'): .4f}")

    print("\nput Greeks:")
    print(f"delta: {delta(S, K, T, r, sigma, 'put'): .4f}")
    print(f"gamma: {gamma(S, K, T, r, sigma): .4f}")
    print(f"vega: {vega(S, K, T, r, sigma): .4f}")
    print(f"theta: {theta(S, K, T, r, sigma, 'put'): .4f}")
    print(f"rho: {rho(S, K, T, r, sigma, 'put'): .4f}")
