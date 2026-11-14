"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options."""

from __future__ import annotations

import numpy as np

from ._validation import validate_inputs, validate_option_type


def binomial_price(
    S,
    K,
    T,
    r,
    sigma,
    n_steps: int = 100,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """CRR binomial-tree price of a vanilla option (no dividends).

    Parameters
    ----------
    n_steps : int
        Number of time steps in the tree.
    option_type : {'call', 'put'}
    american : bool
        Allow early exercise at every node. (Without dividends an American call
        is never exercised early, so it equals the European call.)
    """
    option_type = validate_option_type(option_type)
    validate_inputs(S, K, T, r, sigma)
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    n_steps = int(n_steps)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability p={p:.4f} is outside (0, 1): "
            "increase n_steps or check the inputs"
        )
    discount = np.exp(-r * dt)
    sign = 1.0 if option_type == "call" else -1.0

    # Terminal node j (= number of up-moves) has price S * u**j * d**(n-j).
    j = np.arange(n_steps + 1)
    values = np.maximum(sign * (S * u**j * d ** (n_steps - j) - K), 0.0)

    # Backward induction.
    for i in range(n_steps, 0, -1):
        values = discount * (p * values[1 : i + 1] + (1 - p) * values[:i])
        if american:
            j = np.arange(i)
            nodes = S * u**j * d ** (i - 1 - j)
            values = np.maximum(values, np.maximum(sign * (nodes - K), 0.0))

    return float(values[0])


def binomial_call(S, K, T, r, sigma, n_steps: int = 100) -> float:
    """CRR price of a European call."""
    return binomial_price(S, K, T, r, sigma, n_steps, "call", american=False)


def binomial_put(S, K, T, r, sigma, n_steps: int = 100, american: bool = False) -> float:
    """CRR price of a European (default) or American put."""
    return binomial_price(S, K, T, r, sigma, n_steps, "put", american=american)
