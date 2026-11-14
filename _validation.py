"""Shared input validation for the pricing models."""

from __future__ import annotations

import numpy as np

_VALID_TYPES = ("call", "put")


def validate_inputs(S, K, T, r, sigma) -> None:
    """Raise ValueError if the market / contract parameters are not admissible.

    S, K, T and sigma must be strictly positive; r may be any finite number
    (negative rates are allowed). Array inputs are checked element-wise.
    """
    for name, value in (("S", S), ("K", K), ("T", T), ("sigma", sigma)):
        if not np.all(np.asarray(value) > 0):
            raise ValueError(f"{name} must be strictly positive, got {value!r}")
    if not np.all(np.isfinite(np.asarray(r, dtype=float))):
        raise ValueError(f"r must be finite, got {r!r}")


def validate_option_type(option_type: str) -> str:
    """Normalise and check ``option_type`` ('call' or 'put')."""
    option_type = str(option_type).lower()
    if option_type not in _VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return option_type
