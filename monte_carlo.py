"""
monte_carlo.py
Monte Carlo simulation for European option pricing via simulated GBM terminal prices.
Compares convergence/accuracy against Black-Scholes and the CRR binomial tree.

Week 6 of options-engine project
"""

import math
import random
from bs_model import black_scholes
from binomial_tree import binomial_tree_price


def monte_carlo_price(S, K, T, r, sigma, option_type="call", n_sims=100000, seed=None):
    """
    Price a European option via Monte Carlo simulation of terminal stock prices
    under geometric Brownian motion (GBM).

    Returns (price, standard_error).
    """
    if seed is not None:
        random.seed(seed)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    payoffs = []
    for _ in range(n_sims):
        z = random.gauss(0, 1)
        S_T = S * math.exp(drift + diffusion * z)
        if option_type == "call":
            payoff = max(S_T - K, 0)
        elif option_type == "put":
            payoff = max(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        payoffs.append(payoff)

    mean_payoff = sum(payoffs) / n_sims
    price = discount * mean_payoff

    # Standard error of the mean, discounted to present value
    variance = sum((p - mean_payoff) ** 2 for p in payoffs) / (n_sims - 1)
    std_error = discount * math.sqrt(variance / n_sims)

    return price, std_error


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 0.25, 0.03, 0.25

    bs_price = black_scholes(S, K, T, r, sigma, "call")
    tree_price = binomial_tree_price(
        S, K, T, r, sigma, N=200, option_type="call", american=False
    )

    print(f"Black-Scholes price: {bs_price:.4f}")
    print(f"Binomial tree price: {tree_price:.4f}")
    print()

    print("Monte Carlo convergence (call option):")
    for n_sims in [1000, 10000, 100000, 500000]:
        mc_price, mc_se = monte_carlo_price(
            S, K, T, r, sigma, "call", n_sims=n_sims, seed=42
        )
        error_vs_bs = abs(mc_price - bs_price)
        print(
            f"  N={n_sims:>7}: price={mc_price:.4f}  std_error={mc_se:.4f}  |diff from BS|={error_vs_bs:.4f}"
        )
