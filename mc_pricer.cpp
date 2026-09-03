// mc_pricer.cpp
// C++ Monte Carlo core for European option pricing.
// Compiled as a DLL, called from Python via ctypes.

#include <cmath>
#include <random>

extern "C" {

__declspec(dllexport)
void monte_carlo_cpp(double S, double K, double T, double r, double sigma,
                      int option_type, long long n_sims, unsigned int seed,
                      double* out_price, double* out_std_error) {

    double drift = (r - 0.5 * sigma * sigma) * T;
    double diffusion = sigma * std::sqrt(T);
    double discount = std::exp(-r * T);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;

    for (long long i = 0; i < n_sims; i++) {
        double z = normal(rng);
        double S_T = S * std::exp(drift + diffusion * z);

        double payoff;
        if (option_type == 0) {          // call
            payoff = S_T - K;
            if (payoff < 0.0) payoff = 0.0;
        } else {                          // put
            payoff = K - S_T;
            if (payoff < 0.0) payoff = 0.0;
        }

        sum_payoff += payoff;
        sum_payoff_sq += payoff * payoff;
    }

    double mean_payoff = sum_payoff / n_sims;
    double price = discount * mean_payoff;

    double variance = (sum_payoff_sq / n_sims - mean_payoff * mean_payoff)
                       * n_sims / (n_sims - 1);
    double std_error = discount * std::sqrt(variance / n_sims);

    *out_price = price;
    *out_std_error = std_error;
}

} // extern "C"