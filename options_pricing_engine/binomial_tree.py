"""
binomial_tree.py
----------------
N-step Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

Reference: Cox, J., Ross, S. & Rubinstein, M. (1979). "Option Pricing:
A Simplified Approach." Journal of Financial Economics, 7(3), 229-263.

Key insight: As N -> infinity, the binomial price converges to Black-Scholes.
We verify this convergence in the demo script.
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal

OptionType  = Literal["call", "put"]
ExerciseType = Literal["european", "american"]


@dataclass
class BinomialResult:
    """Container for binomial tree pricing output."""
    option_type:   str
    exercise_type: str
    price:         float
    N:             int
    u:             float   # up factor
    d:             float   # down factor
    q:             float   # risk-neutral probability of up move
    stock_tree:    np.ndarray  # shape (N+1, N+1) upper-triangular
    option_tree:   np.ndarray  # shape (N+1, N+1) upper-triangular
    early_exercise: np.ndarray | None  # True where American option is exercised early

    def __str__(self) -> str:
        lines = [
            f"  Option        : {self.option_type.upper()} {self.exercise_type.upper()}",
            f"  Price         : {self.price:.4f}",
            f"  Steps (N)     : {self.N}",
            f"  Up factor (u) : {self.u:.4f}",
            f"  Down factor(d): {self.d:.4f}",
            f"  Risk-neutral q: {self.q:.4f}",
        ]
        return "\n".join(lines)


def price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 100,
    option_type: OptionType = "call",
    exercise_type: ExerciseType = "european",
) -> BinomialResult:
    """
    Price an option using the CRR binomial tree.

    Parameters
    ----------
    S             : Current spot price
    K             : Strike price
    r             : Continuously compounded risk-free rate (annualised)
    sigma         : Annualised volatility
    T             : Time to maturity in years
    N             : Number of time steps (more steps = more accurate)
    option_type   : 'call' or 'put'
    exercise_type : 'european' (only at expiry) or 'american' (any time)

    Algorithm
    ---------
    1. Build the recombining stock price tree (forward pass).
    2. Compute terminal payoffs at all N+1 nodes.
    3. Backward induction: discount expected payoff under Q.
       For American options, take max(intrinsic, continuation value).
    """
    dt = T / N                          # length of each time step
    u  = np.exp(sigma * np.sqrt(dt))    # up factor  (CRR parameterisation)
    d  = 1.0 / u                        # down factor (d = 1/u ensures recombination)
    disc = np.exp(-r * dt)              # one-step discount factor

    # Risk-neutral probability of an up move (no-arbitrage requires d < e^{r dt} < u)
    q = (np.exp(r * dt) - d) / (u - d)

    if not (0 < q < 1):
        raise ValueError(
            f"Risk-neutral probability q={q:.4f} is outside (0,1). "
            "Try fewer steps, shorter T, or different parameters."
        )

    # --- Forward pass: build stock price tree ---
    # stock_tree[i, j] = S * u^j * d^(i-j)  at time step i with j up-moves
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):          # time step
        for j in range(i + 1):     # number of up moves (0 .. i)
            stock_tree[i, j] = S * (u ** j) * (d ** (i - j))

    # --- Terminal payoffs ---
    payoff = (
        np.maximum(stock_tree[N] - K, 0) if option_type == "call"
        else np.maximum(K - stock_tree[N], 0)
    )

    option_tree    = np.zeros((N + 1, N + 1))
    option_tree[N] = payoff

    early_exercise = None
    if exercise_type == "american":
        early_exercise = np.zeros((N + 1, N + 1), dtype=bool)

    # --- Backward induction ---
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (q * option_tree[i + 1, j + 1]
                                   + (1 - q) * option_tree[i + 1, j])
            if exercise_type == "american":
                intrinsic = (
                    max(stock_tree[i, j] - K, 0) if option_type == "call"
                    else max(K - stock_tree[i, j], 0)
                )
                option_tree[i, j] = max(intrinsic, continuation)
                if option_tree[i, j] == intrinsic > 0:
                    early_exercise[i, j] = True
            else:
                option_tree[i, j] = continuation

    return BinomialResult(
        option_type=option_type,
        exercise_type=exercise_type,
        price=option_tree[0, 0],
        N=N,
        u=u,
        d=d,
        q=q,
        stock_tree=stock_tree,
        option_tree=option_tree,
        early_exercise=early_exercise,
    )


def convergence_to_bs(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType = "call",
    step_range: list[int] | None = None,
) -> dict:
    """
    Compute binomial prices for increasing N to demonstrate convergence
    to the Black-Scholes analytical price.

    Returns a dictionary with keys 'steps', 'prices', 'errors', 'bs_price'.
    """
    from black_scholes import price as bs_price

    if step_range is None:
        step_range = [5, 10, 20, 50, 100, 200, 500]

    bs = bs_price(S, K, r, sigma, T, option_type).price
    steps, prices, errors = [], [], []

    for N in step_range:
        bt = price(S, K, r, sigma, T, N, option_type, "european")
        steps.append(N)
        prices.append(bt.price)
        errors.append(abs(bt.price - bs))

    return {"steps": steps, "prices": prices, "errors": errors, "bs_price": bs}
