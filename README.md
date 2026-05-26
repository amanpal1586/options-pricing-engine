# Option Pricing Engine: From No-Arbitrage to Black-Scholes
### A Computational Study of Binomial Trees, Greeks, and Risk-Neutral Valuation



---

## Overview

This project implements two foundational option pricing models from scratch in Python — the **Black-Scholes-Merton (BSM) analytical formula** and the **Cox-Ross-Rubinstein (CRR) binomial tree** — starting from first principles of mathematical finance.

The goal is not to use an existing pricing library, but to derive and implement every component: the stochastic process, the no-arbitrage argument, risk-neutral valuation, closed-form Greeks, and numerical convergence.

---

## Significance

Most students can recite the Black-Scholes formula. This project proves something harder: that you can **derive it, implement it correctly, verify its mathematical properties, and communicate it clearly** 

- **Stochastic calculus** — GBM dynamics derived via Itô's lemma to produce the log-normal distribution of stock prices
- **No-arbitrage pricing** — the Fundamental Theorem of Asset Pricing applied to construct the risk-neutral measure Q
- **Closed-form Greeks** — all five sensitivities (Δ, Γ, Θ, ν, ρ) derived analytically, not estimated numerically
- **Put-call parity** — verified to machine precision (error = 0.0), confirming mathematical consistency
- **Binomial convergence** — CRR tree prices converge to Black-Scholes at rate O(1/N), proven empirically on a log-log scale
- **American options** — early exercise premium captured via backward induction (no closed-form exists)
- **Implied volatility** — recovered via Newton-Raphson iteration using analytical Vega as the derivative

---

## Project Structure

```
options_engine/
├── black_scholes.py      # Analytical BSM pricing, all Greeks, put-call parity, implied vol
├── binomial_tree.py      # CRR N-step tree for European & American options, convergence
├── visualisations.py     # Greeks surfaces, convergence plots, 3D price surface, tree diagram
└── demo.py               # End-to-end script — reproduces all tables and figures
```

---

## Key Results

| Result | Value |
|---|---|
| European call price (S=100, K=100, r=5%, σ=20%, T=1y) | 10.4506 |
| Put-call parity error | 0.0 (machine precision) |
| Implied vol round-trip error | 0.00e+00 |
| American put early exercise premium | 0.5229 |
| Binomial error at N=500 | 0.00400 → O(1/N) |

---

## Quickstart

```python
from black_scholes import price, implied_volatility
from binomial_tree import price as bt_price

# European call — price + all Greeks in one call
result = price(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type='call')
print(result)
# Option : CALL  |  Price: 10.4506  |  Delta: 0.6368  |  Gamma: 0.0188
# Theta: -0.0176 (per day)  |  Vega: 0.3752 (per 1% vol)  |  Rho: 0.5323

# American put — binomial tree, N=200 steps
amer = bt_price(100, 100, 0.05, 0.20, 1.0, N=200, option_type='put', exercise_type='american')
print(amer.price)   # 6.0864

# Implied volatility recovery
iv = implied_volatility(market_price=10.4506, S=100, K=100, r=0.05, T=1.0)
print(f'IV = {iv:.4f}')   # IV = 0.2000
```

---

## Dependencies

```
numpy
scipy
matplotlib
reportlab
```

Install with: `pip install numpy scipy matplotlib reportlab`

---

## References

1. **Black, F. & Scholes, M. (1973).** The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.
2. **Cox, J., Ross, S. & Rubinstein, M. (1979).** Option Pricing: A Simplified Approach. *Journal of Financial Economics*, 7(3), 229–263.
3. **Merton, R. C. (1973).** Theory of Rational Option Pricing. *Bell Journal of Economics and Management Science*, 4(1), 141–183.
4. **Capinski, M. & Zastawniak, T. (2010).** *Mathematics for Finance: An Introduction to Financial Engineering*. Springer.
5. **Hull, J. C. (2018).** *Options, Futures and Other Derivatives*, 10th Ed. Pearson.
6. **Cvitanic, J. & Zapatero, F. (2007).** *Introduction to the Economics and Mathematics of Financial Markets*. Prentice-Hall.
7. **Chandra, S. et al. (2014).** *Financial Mathematics: An Introduction*. Narosa Publishers.

---


*Built as a financial engineering portfolio project. All results are reproducible by running `python demo.py`.*
