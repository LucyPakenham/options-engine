"""
implied_vol.py
Back out implied volatility from an observed market price using Newton-Raphson,
with bisection as a fallback for robustness.

Week 4 of options-engine project
"""

import math
from bs_model import black_scholes, compute_d1_d2


def raw_vega(S, K, T, r, sigma):
    """Unscaled vega: dPrice/dSigma. Used internally for Newton-Raphson steps."""
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    normal_pdf_d1 = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2)
    return S * normal_pdf_d1 * math.sqrt(T)


def implied_vol_newton(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    initial_guess=0.2,
    tol=1e-6,
    max_iter=100,
):
    """
    Newton-Raphson search for implied volatility.
    Returns None if it fails to converge (caller should fall back to bisection).
    """
    sigma = initial_guess

    for i in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        v = raw_vega(S, K, T, r, sigma)
        if v < 1e-8:
            return None  # vega too small, Newton-Raphson unreliable here

        sigma = sigma - diff / v

        if sigma <= 0:
            return None  # stepped into invalid territory

    return None  # didn't converge within max_iter


def implied_vol_bisection(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    low=1e-6,
    high=5.0,
    tol=1e-6,
    max_iter=200,
):
    """
    Bisection search for implied volatility. Slower but always converges
    if a root exists between low and high.
    """
    price_low = black_scholes(S, K, T, r, low, option_type) - market_price
    price_high = black_scholes(S, K, T, r, high, option_type) - market_price

    if price_low * price_high > 0:
        raise ValueError("Market price not bracketed by given volatility range")

    for i in range(max_iter):
        mid = (low + high) / 2
        price_mid = black_scholes(S, K, T, r, mid, option_type) - market_price

        if abs(price_mid) < tol:
            return mid

        if price_low * price_mid < 0:
            high = mid
        else:
            low = mid
            price_low = price_mid

    return (low + high) / 2


def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """
    Main entry point: try Newton-Raphson first (fast), fall back to
    bisection (robust) if it fails.
    """
    result = implied_vol_newton(market_price, S, K, T, r, option_type)
    if result is not None:
        return result
    return implied_vol_bisection(market_price, S, K, T, r, option_type)


if __name__ == "__main__":
    # Sanity check: price an option at a known sigma, then back out
    # implied vol from that price. We should recover the original sigma.
    S, K, T, r, true_sigma = 100, 100, 0.25, 0.03, 0.25

    market_price = black_scholes(S, K, T, r, true_sigma, "call")
    print(f"Priced call at sigma={true_sigma}: {market_price:.4f}")

    recovered_sigma = implied_volatility(market_price, S, K, T, r, "call")
    print(f"Recovered implied vol: {recovered_sigma:.4f}")

    # Deep out-of-the-money call — small vega, tests the bisection fallback
    otm_price = black_scholes(100, 150, 0.25, 0.03, 0.25, "call")
    print(
        f"OTM price: {otm_price:.4f}, recovered vol: {implied_volatility(otm_price, 100, 150, 0.25, 0.03, 'call'):.4f}"
    )

    # Put option — confirms it works for both option types, not just calls
    put_price = black_scholes(100, 100, 0.25, 0.03, 0.25, "put")
    print(
        f"Put price: {put_price:.4f}, recovered vol: {implied_volatility(put_price, 100, 100, 0.25, 0.03, 'put'):.4f}"
    )

    # High volatility regime
    high_vol_price = black_scholes(100, 100, 0.25, 0.03, 0.80, "call")
    print(
        f"High-vol price: {high_vol_price:.4f}, recovered vol: {implied_volatility(high_vol_price, 100, 100, 0.25, 0.03, 'call'):.4f}"
    )
