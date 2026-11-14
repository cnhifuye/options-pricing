"""Regenerate the figures used in the README.

Usage (from the repository root):
    pip install -e ".[dev]"
    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from options_pricing import (
    binomial_call,
    binomial_put,
    black_scholes_call,
    black_scholes_put,
    monte_carlo_price,
)

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

bs_call = black_scholes_call(S, K, T, R, SIGMA)
bs_put = black_scholes_put(S, K, T, R, SIGMA)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Monte Carlo standard error: standard vs antithetic
sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]
se_std, se_anti = [], []
for n in sizes:
    se_std.append(monte_carlo_price(S, K, T, R, SIGMA, n, 42, return_stderr=True)[1])
    se_anti.append(
        monte_carlo_price(S, K, T, R, SIGMA, n, 42, antithetic=True, return_stderr=True)[1]
    )

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(sizes, se_std, "o-", label="Standard MC")
ax.loglog(sizes, se_anti, "s-", label="Antithetic MC")
ref = se_std[0] * np.sqrt(sizes[0] / np.array(sizes))
ax.loglog(sizes, ref, "k--", alpha=0.5, label=r"$1/\sqrt{N}$ reference")
ax.set(xlabel="Number of simulated paths", ylabel="Standard error",
       title="Monte Carlo standard error (European call)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
save(fig, "mc_standard_error.png")

# 2. Binomial convergence (zig-zag around the Black-Scholes value)
steps = np.arange(5, 301)
tree = [binomial_call(S, K, T, R, SIGMA, int(n)) for n in steps]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(steps, tree, lw=1, label="CRR binomial tree")
ax.axhline(bs_call, ls="--", c="k", label="Black-Scholes")
ax.set(xlabel="Number of tree steps", ylabel="Call price",
       title="Binomial tree convergence (European call)")
ax.grid(True, alpha=0.3)
ax.legend()
save(fig, "binomial_convergence.png")

# 3. American vs European put as a function of spot
spots = np.linspace(60, 140, 41)
eu = [black_scholes_put(s, K, T, R, SIGMA) for s in spots]
am = [binomial_put(s, K, T, R, SIGMA, 500, american=True) for s in spots]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(spots, am, label="American put (binomial, 500 steps)")
ax.plot(spots, eu, "--", label="European put (Black-Scholes)")
ax.plot(spots, np.maximum(K - spots, 0), ":", c="gray", label="Intrinsic value")
ax.set(xlabel="Spot price S", ylabel="Put price", title="Early-exercise premium of an American put")
ax.grid(True, alpha=0.3)
ax.legend()
save(fig, "american_vs_european_put.png")
