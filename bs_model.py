"""
bs_model.py
Black-Scholes option pricing model.
"""

import math


def normal_cdf(x):
    """Standard normal cumulative distribution function (replaces scipy's norm.cdf)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def validate_inputs(S, K, T, r, sigma):
    """
    Guard against inputs that would break the math: log() needs a positive
    domain, and T/sigma sit in a denominator, so zero or negative values
    would divide by zero.
    """
    if S <= 0:
        raise ValueError(f"Stock price S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time to expiry T must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")


def compute_d1_d2(S, K, T, r, sigma):
    """
    Shared d1/d2 calculation, used by pricing, the Greeks, and implied vol.
    Centralizing this means the validation above only needs to exist once.
    """
    validate_inputs(S, K, T, r, sigma)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


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
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
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

    # Put-call parity check: call - put should equal S - K*exp(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = S - K * math.exp(-r * T)
    print(f"Put-call parity check: {parity_lhs:.6f} vs {parity_rhs:.6f}")

    # Validation check: this should raise a ValueError, not crash silently
    try:
        black_scholes(S, K, 0, r, sigma, "call")
    except ValueError as e:
        print(f"Validation working correctly: {e}")
