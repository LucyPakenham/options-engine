"""
monte_carlo_cpp.py
Python wrapper around the compiled C++ Monte Carlo pricer (mc_pricer.dll),
called via ctypes.

Week 7 of options-engine project
"""

import ctypes
import os

_dll_path = os.path.join(os.path.dirname(__file__), "mc_pricer.dll")
_mc_lib = ctypes.CDLL(_dll_path)

_mc_lib.monte_carlo_cpp.argtypes = [
    ctypes.c_double,  # S
    ctypes.c_double,  # K
    ctypes.c_double,  # T
    ctypes.c_double,  # r
    ctypes.c_double,  # sigma
    ctypes.c_int,  # option_type (0=call, 1=put)
    ctypes.c_longlong,  # n_sims
    ctypes.c_uint,  # seed
    ctypes.POINTER(ctypes.c_double),  # out_price
    ctypes.POINTER(ctypes.c_double),  # out_std_error
]
_mc_lib.monte_carlo_cpp.restype = None


def monte_carlo_price_cpp(
    S, K, T, r, sigma, option_type="call", n_sims=100000, seed=42
):
    """
    Price a European option using the compiled C++ Monte Carlo core.
    Returns (price, standard_error), matching monte_carlo_price()'s interface.
    """
    type_flag = 0 if option_type == "call" else 1

    out_price = ctypes.c_double()
    out_std_error = ctypes.c_double()

    _mc_lib.monte_carlo_cpp(
        S,
        K,
        T,
        r,
        sigma,
        type_flag,
        n_sims,
        seed,
        ctypes.byref(out_price),
        ctypes.byref(out_std_error),
    )

    return out_price.value, out_std_error.value


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 0.25, 0.03, 0.25

    price, se = monte_carlo_price_cpp(S, K, T, r, sigma, "call", n_sims=100000, seed=42)
    print(f"C++ Monte Carlo price: {price:.4f}  std_error: {se:.4f}")
