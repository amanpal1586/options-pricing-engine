"""
demo.py
-------
End-to-end demonstration of the option pricing engine.
Run this script to reproduce all results shown in the LaTeX write-up.

Usage:
    python demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import black_scholes as bs
import binomial_tree as bt


# ---- parameters (match the write-up examples) ----
S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

HEADER = lambda s: print(f"\n{'='*55}\n  {s}\n{'='*55}")


# ===== 1. Black-Scholes pricing ========================================== #
HEADER("1. Black-Scholes pricing")

call = bs.price(S, K, r, sigma, T, "call")
put  = bs.price(S, K, r, sigma, T, "put")
print("\n--- European Call ---")
print(call)
print("\n--- European Put ---")
print(put)


# ===== 2. Put-Call Parity verification =================================== #
HEADER("2. Put-Call Parity: C - P = S - K·exp(-rT)")

pcp = bs.put_call_parity_check(S, K, r, sigma, T)
for k, v in pcp.items():
    print(f"  {k:<18}: {v}")


# ===== 3. Binomial tree pricing ========================================== #
HEADER("3. Binomial Tree (CRR) — European Call")

for N in [10, 50, 100, 500]:
    res = bt.price(S, K, r, sigma, T, N, "call", "european")
    print(f"  N={N:<5}  price={res.price:.5f}  (BS={call.price:.5f}  err={abs(res.price-call.price):.5f})")


# ===== 4. American vs European Put ======================================= #
HEADER("4. American vs European Put (early exercise premium)")

eur_put = bt.price(S, K, r, sigma, T, 200, "put", "european")
ame_put = bt.price(S, K, r, sigma, T, 200, "put", "american")
bs_put  = bs.price(S, K, r, sigma, T, "put")

print(f"\n  European put (BS analytical) : {bs_put.price:.5f}")
print(f"  European put (binomial N=200): {eur_put.price:.5f}")
print(f"  American put (binomial N=200): {ame_put.price:.5f}")
print(f"  Early exercise premium       : {ame_put.price - eur_put.price:.5f}")


# ===== 5. Implied volatility recovery ==================================== #
HEADER("5. Implied volatility — round-trip check")

mkt_price = call.price     # use our own BS price as "market"
iv = bs.implied_volatility(mkt_price, S, K, r, T, "call")
print(f"\n  Input sigma     : {sigma:.4f}  ({sigma:.0%})")
print(f"  Recovered IV    : {iv:.4f}  ({iv:.0%})")
print(f"  Round-trip error: {abs(iv - sigma):.2e}")


# ===== 6. Greeks table ==================================================== #
HEADER("6. Greeks summary (ATM call, T=1y)")

print(f"\n  {'Greek':<10} {'Value':>10}  {'Interpretation'}")
print(f"  {'-'*55}")
rows = [
    ("Delta",  call.delta, "Spot sensitivity — hedge ratio"),
    ("Gamma",  call.gamma, "Rate of change of Delta"),
    ("Theta",  call.theta, "Value lost per calendar day"),
    ("Vega",   call.vega,  "Value gained per +1% vol"),
    ("Rho",    call.rho,   "Value gained per +1% rate"),
]
for name, val, interp in rows:
    print(f"  {name:<10} {val:>10.4f}  {interp}")


# ===== 7. Generate all figures =========================================== #
HEADER("7. Generating figures")

try:
    import visualisations as viz
    out = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out, exist_ok=True)
    viz.plot_greeks(savepath=out + "/greeks.png")
    viz.plot_convergence(savepath=out + "/convergence.png")
    viz.plot_price_surface(savepath=out + "/price_surface.png")
    viz.plot_tree_diagram(savepath=out + "/tree_diagram.png")
except Exception as e:
    print(f"  Figure generation skipped: {e}")

print("\nDone. See figures/ directory for all plots.")
