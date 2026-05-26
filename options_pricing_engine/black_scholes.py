"""
black_scholes.py
----------------
Analytical Black-Scholes pricing for European options.

Reference: Black, F. & Scholes, M. (1973). "The Pricing of Options and
Corporate Liabilities." Journal of Political Economy, 81(3), 637-654.

Model assumptions:
  - Underlying follows Geometric Brownian Motion: dS = mu*S dt + sigma*S dW
  - Constant risk-free rate r and volatility sigma
  - No dividends, no transaction costs
  - Continuous trading, no arbitrage
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


OptionType = Literal["call", "put"]


@dataclass
class BSMResult:
    """Container for Black-Scholes price and all Greeks."""
    option_type: str
    price: float
    delta: float      # dV/dS        — sensitivity to spot
    gamma: float      # d2V/dS2      — rate of change of delta
    theta: float      # dV/dt        — time decay (per calendar day)
    vega: float       # dV/d(sigma)  — sensitivity to vol (per 1% move)
    rho: float        # dV/dr        — sensitivity to rate (per 1% move)
    d1: float
    d2: float

    def __str__(self) -> str:
        lines = [
            f"  Option : {self.option_type.upper()}",
            f"  Price  : {self.price:.4f}",
            f"  Delta  : {self.delta:.4f}",
            f"  Gamma  : {self.gamma:.4f}",
            f"  Theta  : {self.theta:.4f}  (per day)",
            f"  Vega   : {self.vega:.4f}   (per 1% vol)",
            f"  Rho    : {self.rho:.4f}    (per 1% rate)",
        ]
        return "\n".join(lines)


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float):
    """
    Compute d1 and d2 — the standardised log-moneyness terms.

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    if T <= 0:
        raise ValueError("Time to maturity T must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")
    if S <= 0 or K <= 0:
        raise ValueError("Spot S and strike K must be positive.")

    ln_SK = np.log(S / K)
    d1 = (ln_SK + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType = "call",
) -> BSMResult:
    """
    Price a European option and compute all Greeks analytically.

    Parameters
    ----------
    S     : Current spot price of the underlying
    K     : Strike price
    r     : Continuously compounded risk-free rate (annualised)
    sigma : Annualised volatility of the underlying
    T     : Time to maturity in years
    option_type : 'call' or 'put'

    Returns
    -------
    BSMResult with price and Greeks
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    Nd1  = norm.cdf(d1)
    Nd2  = norm.cdf(d2)
    Nnd1 = norm.cdf(-d1)
    Nnd2 = norm.cdf(-d2)
    npd1 = norm.pdf(d1)          # standard normal PDF at d1
    disc = np.exp(-r * T)        # discount factor

    if option_type == "call":
        price_val = S * Nd1 - K * disc * Nd2
        delta     = Nd1
        rho_val   = K * T * disc * Nd2 / 100
    elif option_type == "put":
        price_val = K * disc * Nnd2 - S * Nnd1
        delta     = Nd1 - 1              # = -N(-d1)
        rho_val   = -K * T * disc * Nnd2 / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    # Greeks shared between calls and puts
    gamma = npd1 / (S * sigma * np.sqrt(T))
    vega  = S * npd1 * np.sqrt(T) / 100   # per 1% move in vol
    # Theta: rate of change of value per calendar day
    theta = (
        -(S * npd1 * sigma) / (2 * np.sqrt(T))
        - r * K * disc * (Nd2 if option_type == "call" else -Nnd2)
    ) / 365

    return BSMResult(
        option_type=option_type,
        price=price_val,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho_val,
        d1=d1,
        d2=d2,
    )


def put_call_parity_check(
    S: float, K: float, r: float, sigma: float, T: float
) -> dict:
    """
    Verify Put-Call Parity: C - P = S - K * exp(-rT).

    Returns dictionary showing both sides and the numerical error.
    """
    call = price(S, K, r, sigma, T, "call")
    put  = price(S, K, r, sigma, T, "put")
    lhs  = call.price - put.price
    rhs  = S - K * np.exp(-r * T)
    return {
        "C - P"         : round(lhs, 8),
        "S - K*exp(-rT)": round(rhs, 8),
        "error"         : round(abs(lhs - rhs), 10),
        "parity_holds"  : np.isclose(lhs, rhs, atol=1e-8),
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """
    Recover implied volatility via Newton-Raphson iteration.

    Solves:  BSM(sigma) - market_price = 0
    using the analytical vega as the derivative.
    """
    sigma = 0.2  # initial guess
    for _ in range(max_iter):
        result = price(S, K, r, sigma, T, option_type)
        diff   = result.price - market_price
        if abs(diff) < tol:
            return sigma
        vega_val = result.vega * 100   # undo per-1% scaling
        if abs(vega_val) < 1e-10:
            break
        sigma -= diff / vega_val
        sigma  = max(sigma, 1e-6)      # keep sigma positive
    raise RuntimeError(f"Implied vol did not converge (last sigma={sigma:.4f})")
