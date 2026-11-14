"""Analytical Black-Scholes Greeks for European options (no dividends)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ._validation import validate_inputs, validate_option_type
from .black_scholes import _d1_d2


def bs_greeks(S, K, T, r, sigma, option_type: str = "call") -> dict:
    """Return Delta, Gamma, Vega, Theta and Rho.

    Conventions
    -----------
    * Vega is the price change per 1.00 (i.e. 100 percentage points) change in
      volatility; divide by 100 for "per vol point".
    * Theta is per year (calendar-time decay); divide by 365 for per day.
    * Rho is per 1.00 change in the interest rate; divide by 100 for per 1%.
    """
    option_type = validate_option_type(option_type)
    validate_inputs(S, K, T, r, sigma)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * disc * norm.cdf(d2)
        rho = K * T * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * disc * norm.cdf(-d2)
        rho = -K * T * disc * norm.cdf(-d2)

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}
