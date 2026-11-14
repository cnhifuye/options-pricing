"""Black-Scholes closed-form pricing for European options (no dividends)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ._validation import validate_inputs, validate_option_type


def _d1_d2(S, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type: str = "call"):
    """Black-Scholes price of a European option.

    Parameters
    ----------
    S : float or array
        Current underlying price.
    K : float or array
        Strike price.
    T : float or array
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float or array
        Annualised volatility.
    option_type : {'call', 'put'}
    """
    option_type = validate_option_type(option_type)
    validate_inputs(S, K, T, r, sigma)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes price of a European call option."""
    return black_scholes_price(S, K, T, r, sigma, "call")


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes price of a European put option."""
    return black_scholes_price(S, K, T, r, sigma, "put")
