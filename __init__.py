"""options_pricing: Black-Scholes, Monte Carlo and binomial-tree option pricing."""

from .binomial_tree import binomial_call, binomial_price, binomial_put
from .black_scholes import black_scholes_call, black_scholes_price, black_scholes_put
from .greeks import bs_greeks
from .monte_carlo import (
    monte_carlo_call,
    monte_carlo_call_antithetic,
    monte_carlo_price,
    monte_carlo_put,
)

__version__ = "0.1.0"

__all__ = [
    "black_scholes_price",
    "black_scholes_call",
    "black_scholes_put",
    "bs_greeks",
    "monte_carlo_price",
    "monte_carlo_call",
    "monte_carlo_put",
    "monte_carlo_call_antithetic",
    "binomial_price",
    "binomial_call",
    "binomial_put",
]
