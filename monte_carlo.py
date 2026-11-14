"""Monte Carlo pricing of European options under geometric Brownian motion."""

from __future__ import annotations

import numpy as np

from ._validation import validate_inputs, validate_option_type


def _terminal_price(S, r, sigma, T, Z):
    """Terminal price S_T under the risk-neutral measure."""
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def _payoff(ST, K, option_type):
    return np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)


def monte_carlo_price(
    S,
    K,
    T,
    r,
    sigma,
    n_simulations: int = 100_000,
    seed: int | None = 42,
    option_type: str = "call",
    antithetic: bool = False,
    return_stderr: bool = False,
):
    """Monte Carlo price of a European option.

    Parameters
    ----------
    n_simulations : int
        TOTAL number of simulated terminal prices. With ``antithetic=True`` they
        are generated as n_simulations/2 pairs (Z, -Z), so that standard and
        antithetic estimators use the same computational budget. An odd number
        is rounded down to the nearest even number.
    seed : int or None
        Seed for ``numpy.random.default_rng``.
    antithetic : bool
        Use antithetic variates for variance reduction.
    return_stderr : bool
        If True, return ``(price, standard_error)``.
    """
    option_type = validate_option_type(option_type)
    validate_inputs(S, K, T, r, sigma)
    if int(n_simulations) != n_simulations or n_simulations < 2:
        raise ValueError("n_simulations must be an integer >= 2")
    n_simulations = int(n_simulations)

    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    if antithetic:
        n_pairs = n_simulations // 2
        Z = rng.standard_normal(n_pairs)
        pair_payoff = 0.5 * (
            _payoff(_terminal_price(S, r, sigma, T, Z), K, option_type)
            + _payoff(_terminal_price(S, r, sigma, T, -Z), K, option_type)
        )
        samples = disc * pair_payoff
    else:
        Z = rng.standard_normal(n_simulations)
        samples = disc * _payoff(_terminal_price(S, r, sigma, T, Z), K, option_type)

    price = samples.mean()
    if return_stderr:
        return price, samples.std(ddof=1) / np.sqrt(samples.size)
    return price


def monte_carlo_call(S, K, T, r, sigma, n_simulations=100_000, seed=42, return_stderr=False):
    """Standard Monte Carlo price of a European call."""
    return monte_carlo_price(
        S, K, T, r, sigma, n_simulations, seed, "call", False, return_stderr
    )


def monte_carlo_put(S, K, T, r, sigma, n_simulations=100_000, seed=42, return_stderr=False):
    """Standard Monte Carlo price of a European put."""
    return monte_carlo_price(
        S, K, T, r, sigma, n_simulations, seed, "put", False, return_stderr
    )


def monte_carlo_call_antithetic(
    S, K, T, r, sigma, n_simulations=100_000, seed=42, return_stderr=False
):
    """Antithetic-variates Monte Carlo price of a European call.

    ``n_simulations`` is the total number of simulated paths (n/2 antithetic pairs).
    """
    return monte_carlo_price(
        S, K, T, r, sigma, n_simulations, seed, "call", True, return_stderr
    )
