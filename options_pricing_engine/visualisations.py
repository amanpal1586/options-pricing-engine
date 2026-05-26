"""
visualisations.py
-----------------
Publication-quality plots for the option pricing engine.
All figures saved as PNG — drop them straight into your LaTeX write-up.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import black_scholes as bs
import binomial_tree as bt


# ---------- style ---------------------------------------------------------- #
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "lines.linewidth"  : 1.8,
})
COLORS = ["#1a4f8a", "#d94f3d", "#2e8b57", "#e6821e", "#6a3d9a"]


# ---------- 1. Greeks vs Spot ---------------------------------------------- #
def plot_greeks(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, savepath="greeks.png"):
    spots = np.linspace(50, 180, 300)
    greeks = {"delta": [], "gamma": [], "theta": [], "vega": [], "rho": []}

    for s in spots:
        res = bs.price(s, K, r, sigma, T, "call")
        greeks["delta"].append(res.delta)
        greeks["gamma"].append(res.gamma)
        greeks["theta"].append(res.theta)
        greeks["vega"].append(res.vega)
        greeks["rho"].append(res.rho)

    labels = {
        "delta": (r"$\Delta = \partial C / \partial S$",   COLORS[0]),
        "gamma": (r"$\Gamma = \partial^2 C / \partial S^2$", COLORS[1]),
        "theta": (r"$\Theta$ (per day)",                   COLORS[2]),
        "vega" : (r"$\nu$ (per 1\% vol)",                  COLORS[3]),
        "rho"  : (r"$\rho$ (per 1\% rate)",                COLORS[4]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, (key, (label, color)) in zip(axes, labels.items()):
        ax.plot(spots, greeks[key], color=color)
        ax.axvline(K, color="gray", lw=1, ls="--", alpha=0.5, label="ATM (K=100)")
        ax.set_title(label)
        ax.set_xlabel("Spot price $S$")
        ax.legend(fontsize=8)

    axes[-1].axis("off")
    fig.suptitle(
        f"Black-Scholes Greeks  —  K={K}, r={r:.0%}, σ={sigma:.0%}, T={T}y",
        fontsize=13, y=1.01
    )
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {savepath}")


# ---------- 2. Binomial Tree Convergence ----------------------------------- #
def plot_convergence(S=100, K=100, r=0.05, sigma=0.2, T=1.0, savepath="convergence.png"):
    steps = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    conv  = bt.convergence_to_bs(S, K, r, sigma, T, step_range=steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Price convergence
    ax1.plot(conv["steps"], conv["prices"], "o-", color=COLORS[0], ms=5, label="Binomial")
    ax1.axhline(conv["bs_price"], color=COLORS[1], ls="--", lw=1.5, label=f"Black-Scholes = {conv['bs_price']:.4f}")
    ax1.set_xlabel("Number of steps $N$")
    ax1.set_ylabel("Option price")
    ax1.set_title("Binomial price converges to Black-Scholes")
    ax1.legend()

    # Error decay (log-log)
    ax2.loglog(conv["steps"], conv["errors"], "s-", color=COLORS[2], ms=5)
    # Fit a reference O(1/N) line
    N_arr = np.array(conv["steps"], dtype=float)
    ref   = conv["errors"][0] * conv["steps"][0] / N_arr
    ax2.loglog(N_arr, ref, "k--", lw=1, label=r"$O(1/N)$ reference")
    ax2.set_xlabel("Number of steps $N$")
    ax2.set_ylabel("|Binomial − BS|")
    ax2.set_title(r"Pricing error decays as $O(1/N)$")
    ax2.legend()

    fig.suptitle("CRR Binomial Tree Convergence to Black-Scholes", fontsize=13)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {savepath}")


# ---------- 3. Price Surface (S, T) ---------------------------------------- #
def plot_price_surface(K=100, r=0.05, sigma=0.2, savepath="price_surface.png"):
    spots = np.linspace(60, 160, 60)
    mats  = np.linspace(0.05, 2.0, 60)
    S_grid, T_grid = np.meshgrid(spots, mats)

    C_grid = np.vectorize(lambda s, t: bs.price(s, K, r, sigma, t, "call").price)(S_grid, T_grid)
    P_grid = np.vectorize(lambda s, t: bs.price(s, K, r, sigma, t, "put").price)(S_grid, T_grid)

    fig = plt.figure(figsize=(13, 5))
    for idx, (grid, title) in enumerate([(C_grid, "Call price"), (P_grid, "Put price")], 1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        surf = ax.plot_surface(S_grid, T_grid, grid, cmap="viridis", alpha=0.9, linewidth=0)
        ax.set_xlabel("Spot $S$")
        ax.set_ylabel("Maturity $T$ (yr)")
        ax.set_zlabel("Option price")
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

    fig.suptitle(f"BS Price Surface  —  K={K}, r={r:.0%}, σ={sigma:.0%}", fontsize=13)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {savepath}")


# ---------- 4. Small Binomial Tree Diagram (N=4) --------------------------- #
def plot_tree_diagram(S=100, K=100, r=0.05, sigma=0.3, T=1.0, N=4, savepath="tree_diagram.png"):
    result = bt.price(S, K, r, sigma, T, N, "call", "european")
    stree  = result.stock_tree
    otree  = result.option_tree

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, tree, title, fmt in [
        (axes[0], stree, "Stock price tree", ".2f"),
        (axes[1], otree, "Option value tree", ".3f"),
    ]:
        ax.set_xlim(-0.5, N + 0.5)
        ax.set_ylim(-0.5, N + 0.5)
        ax.axis("off")
        ax.set_title(title, pad=10)

        for i in range(N + 1):
            for j in range(i + 1):
                x, y = i, 2 * j - i    # lay out so tree fans out vertically
                val  = tree[i, j]
                color = "#d4edda" if (title.startswith("Option") and val > 0) else "#dce8f5"
                ax.add_patch(plt.Rectangle((x - 0.38, y - 0.28), 0.76, 0.56,
                                           facecolor=color, edgecolor="#555", lw=0.8, zorder=2))
                ax.text(x, y, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8.5, zorder=3)
                # draw branches
                if i < N:
                    for dj in [0, 1]:
                        nx, ny = i + 1, 2 * (j + dj) - (i + 1)
                        ax.plot([x + 0.38, nx - 0.38], [y, ny],
                                color="#888", lw=0.9, zorder=1)

        ax.text(-0.1, N + 0.3, f"N={N} steps  |  u={result.u:.3f}  d={result.d:.3f}  q={result.q:.3f}",
                fontsize=8, color="#555", transform=ax.transData)

    fig.suptitle("CRR Binomial Tree  (N=4 illustration)", fontsize=13)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {savepath}")


# ---------- run all -------------------------------------------------------- #
if __name__ == "__main__":
    out = "options_engine/figures/"
    os.makedirs(out, exist_ok=True)
    print("Generating figures...")
    plot_greeks(savepath=out + "greeks.png")
    plot_convergence(savepath=out + "convergence.png")
    plot_price_surface(savepath=out + "price_surface.png")
    plot_tree_diagram(savepath=out + "tree_diagram.png")
    print("All figures saved.")
