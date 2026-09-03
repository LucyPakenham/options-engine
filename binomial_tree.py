"""
binomial_tree.py
Cox-Ross-Rubinstein (CRR) binomial tree option pricer.
Supports European and American exercise for calls and puts.

Week 5 of options-engine project
"""

import math


def binomial_tree_price(S, K, T, r, sigma, N=200, option_type="call", american=True):
    """
    Price an option using the CRR binomial tree model.

    S : current stock price
    K : strike price
    T : time to expiry (years)
    r : risk-free rate
    sigma : volatility
    N : number of time steps in the tree (higher = more accurate, slower)
    option_type : "call" or "put"
    american : if True, allows early exercise at every node (American-style);
               if False, exercise only at expiry (European-style)
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)

    if not (0 < p < 1):
        raise ValueError(
            "Risk-neutral probability out of bounds - check inputs (dt too large relative to sigma?)"
        )

    # Terminal stock prices at each of the N+1 final nodes
    stock_prices = [S * (u**j) * (d ** (N - j)) for j in range(N + 1)]

    # Terminal option payoffs
    if option_type == "call":
        values = [max(sp - K, 0) for sp in stock_prices]
    elif option_type == "put":
        values = [max(K - sp, 0) for sp in stock_prices]
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Step backward through the tree
    for step in range(N - 1, -1, -1):
        new_values = []
        for j in range(step + 1):
            continuation = disc * (p * values[j + 1] + (1 - p) * values[j])

            if american:
                stock_price = S * (u**j) * (d ** (step - j))
                if option_type == "call":
                    intrinsic = max(stock_price - K, 0)
                else:
                    intrinsic = max(K - stock_price, 0)
                new_values.append(max(continuation, intrinsic))
            else:
                new_values.append(continuation)

        values = new_values

    return values[0]


if __name__ == "__main__":
    from bs_model import black_scholes

    S, K, T, r, sigma = 100, 100, 0.25, 0.03, 0.25

    euro_call_tree = binomial_tree_price(
        S, K, T, r, sigma, N=200, option_type="call", american=False
    )
    euro_call_bs = black_scholes(S, K, T, r, sigma, "call")
    print(
        f"European call - tree: {euro_call_tree:.4f}, Black-Scholes: {euro_call_bs:.4f}"
    )

    euro_put_tree = binomial_tree_price(
        S, K, T, r, sigma, N=200, option_type="put", american=False
    )
    euro_put_bs = black_scholes(S, K, T, r, sigma, "put")
    print(f"European put - tree: {euro_put_tree:.4f}, Black-Scholes: {euro_put_bs:.4f}")

    amer_call = binomial_tree_price(
        S, K, T, r, sigma, N=200, option_type="call", american=True
    )
    print(f"American call: {amer_call:.4f} (should equal European call, no dividends)")

    amer_put = binomial_tree_price(
        S, K, T, r, sigma, N=200, option_type="put", american=True
    )
    print(
        f"American put: {amer_put:.4f} (should be >= European put: {euro_put_tree:.4f})"
    )
