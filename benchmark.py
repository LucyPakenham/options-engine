"""
benchmark.py
Rigorous Python vs C++ benchmark for Monte Carlo option pricing.

Week 7 of options-engine project
"""

import time
from monte_carlo import monte_carlo_price
from monte_carlo_cpp import monte_carlo_price_cpp


def time_function(func, *args, repeats=3, **kwargs):
    """
    Run `func` several times and return (last_result, best_time).
    Taking the minimum across repeats (rather than the average) is standard
    benchmarking practice: it filters out one-off slowdowns from background
    OS activity, giving a closer estimate of the function's true best-case cost.
    """
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return result, min(times)


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 0.25, 0.03, 0.25
    n_sims_list = [10000, 100000, 1000000, 5000000]

    print(f"{'N sims':>10} | {'Python (s)':>12} | {'C++ (s)':>12} | {'Speedup':>10}")
    print("-" * 52)

    for n_sims in n_sims_list:
        (py_price, py_se), py_time = time_function(
            monte_carlo_price, S, K, T, r, sigma, "call", n_sims=n_sims, seed=42
        )
        (cpp_price, cpp_se), cpp_time = time_function(
            monte_carlo_price_cpp, S, K, T, r, sigma, "call", n_sims=n_sims, seed=42
        )

        speedup = py_time / cpp_time if cpp_time > 0 else float("inf")
        print(f"{n_sims:>10} | {py_time:>12.4f} | {cpp_time:>12.4f} | {speedup:>9.1f}x")

    print()
    print(f"Python price at N={n_sims_list[-1]}: {py_price:.4f} (SE={py_se:.4f})")
    print(f"C++    price at N={n_sims_list[-1]}: {cpp_price:.4f} (SE={cpp_se:.4f})")
