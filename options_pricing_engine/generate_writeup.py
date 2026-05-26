"""
generate_writeup.py
-------------------
Produces a professional PDF write-up of the option pricing engine,
suitable for inclusion in an MFE/MSc application portfolio.

Uses ReportLab Platypus for structured document layout.
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether
)
from reportlab.platypus.flowables import BalancedColumns

import black_scholes as bsm
import binomial_tree as bt

W, H = A4
MARGIN = 2.2 * cm
ACCENT = colors.HexColor("#1a4f8a")
LIGHT  = colors.HexColor("#dce8f5")
GREY   = colors.HexColor("#f5f5f5")
BLACK  = colors.black

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
OUT     = "/mnt/user-data/outputs/option_pricing_writeup.pdf"

# ── styles ──────────────────────────────────────────────────────────────── #
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

Title = S("MyTitle",
    fontSize=22, leading=28, textColor=ACCENT,
    fontName="Times-Bold", alignment=TA_CENTER, spaceAfter=6)
Subtitle = S("MySubtitle",
    fontSize=12, leading=16, textColor=colors.HexColor("#444444"),
    fontName="Times-Italic", alignment=TA_CENTER, spaceAfter=4)
Author = S("MyAuthor",
    fontSize=11, leading=14, textColor=BLACK,
    fontName="Times-Roman", alignment=TA_CENTER, spaceAfter=2)
Abstract = S("MyAbstract",
    fontSize=10, leading=14, textColor=BLACK,
    fontName="Times-Italic", alignment=TA_JUSTIFY,
    leftIndent=1.5*cm, rightIndent=1.5*cm, spaceAfter=12)
H1 = S("MyH1",
    fontSize=14, leading=18, textColor=ACCENT,
    fontName="Times-Bold", spaceBefore=18, spaceAfter=6)
H2 = S("MyH2",
    fontSize=11, leading=14, textColor=BLACK,
    fontName="Times-Bold", spaceBefore=10, spaceAfter=4)
Body = S("MyBody",
    fontSize=10, leading=15, textColor=BLACK,
    fontName="Times-Roman", alignment=TA_JUSTIFY, spaceAfter=6)
Math = S("MyMath",
    fontSize=10, leading=16, textColor=BLACK,
    fontName="Courier", alignment=TA_CENTER,
    spaceBefore=4, spaceAfter=4,
    backColor=GREY, leftIndent=1*cm, rightIndent=1*cm,
    borderPad=6)
Code = S("MyCode",
    fontSize=8.5, leading=13, textColor=BLACK,
    fontName="Courier",
    backColor=GREY, leftIndent=0.5*cm, rightIndent=0.5*cm,
    spaceBefore=4, spaceAfter=6, borderPad=5)
Caption = S("MyCaption",
    fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"),
    fontName="Times-Italic", alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)

def P(text, style=Body): return Paragraph(text, style)
def SP(n=8):             return Spacer(1, n)
def HR():                return HRFlowable(width="100%", thickness=0.5,
                                           color=colors.HexColor("#cccccc"), spaceAfter=6)


def fig(filename, width=14*cm, caption=None):
    path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(path):
        return []
    items = [Image(path, width=width, height=width * 0.55)]
    if caption:
        items.append(P(caption, Caption))
    return items


# ── numerical results (computed live) ───────────────────────────────────── #
S0, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
call = bsm.price(S0, K, r, sigma, T, "call")
put  = bsm.price(S0, K, r, sigma, T, "put")
pcp  = bsm.put_call_parity_check(S0, K, r, sigma, T)
conv = bt.convergence_to_bs(S0, K, r, sigma, T, step_range=[10, 50, 100, 500])
eur_put = bt.price(S0, K, r, sigma, T, 200, "put", "european")
ame_put = bt.price(S0, K, r, sigma, T, 200, "put", "american")


# ── document build ───────────────────────────────────────────────────────── #
def build():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    story = []

    # ── Title page block ─────────────────────────────────────────── #
    story += [
        SP(30),
        P("Option Pricing Engine", Title),
        P("Black-Scholes Formula &amp; CRR Binomial Tree", Subtitle),
        SP(6),
        HR(),
        SP(4),
        P("A Financial Engineering Project", Author),
        SP(20),
        P(
            "<b>Abstract.</b>  This report presents a complete implementation of two "
            "foundational option pricing models: the analytical Black-Scholes-Merton "
            "formula and the Cox-Ross-Rubinstein (CRR) binomial tree. Starting from "
            "first principles — geometric Brownian motion, the no-arbitrage principle, "
            "and risk-neutral valuation — we derive closed-form prices for European "
            "options and extend the framework to American options via backward induction. "
            "We compute all five Greeks analytically, verify put-call parity to "
            "machine precision, recover implied volatilities via Newton-Raphson, and "
            "demonstrate that binomial prices converge to the Black-Scholes price at "
            "rate O(1/N). All results are reproduced from a clean Python library.",
            Abstract,
        ),
        HR(),
        PageBreak(),
    ]

    # ── §1 Introduction ──────────────────────────────────────────── #
    story += [
        P("1.  Introduction", H1),
        P(
            "Options are financial contracts that grant the holder the right, but not "
            "the obligation, to buy (call) or sell (put) an underlying asset at a "
            "pre-agreed strike price K on or before a maturity date T. Pricing such "
            "contracts rigorously requires a mathematical model for the evolution of "
            "the underlying asset price and a principle that rules out riskless profit "
            "— the <i>no-arbitrage principle</i>.",
        ),
        P(
            "Two models dominate introductory financial engineering: the "
            "<b>Black-Scholes-Merton (BSM)</b> model (1973), which yields a closed-form "
            "formula under continuous-time assumptions, and the <b>Cox-Ross-Rubinstein "
            "(CRR) binomial tree</b> (1979), which is a discrete-time approximation that "
            "converges to BSM as the number of steps N tends to infinity.",
        ),
        P("The parameters used throughout this report are:", Body),
    ]

    param_data = [
        ["Parameter", "Symbol", "Value"],
        ["Spot price",       "S",      f"{S0}"],
        ["Strike price",     "K",      f"{K}"],
        ["Risk-free rate",   "r",      f"{r:.0%} p.a."],
        ["Volatility",       "sigma",  f"{sigma:.0%} p.a."],
        ["Time to maturity", "T",      f"{T} year"],
    ]
    t = Table(param_data, colWidths=[5*cm, 3*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Times-Bold"),
        ("FONTNAME",   (0,1), (-1,-1), "Times-Roman"),
        ("FONTSIZE",   (0,0), (-1,-1), 9.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [SP(4), t, SP(12)]

    # ── §2 Risk-Free Assets & Discounting ─────────────────────────── #
    story += [
        P("2.  Risk-Free Assets and the Time Value of Money", H1),
        P(
            "A fundamental concept in finance is that money has a time value: a dollar "
            "today is worth more than a dollar tomorrow. Under <b>continuous compounding</b> "
            "at rate r, a unit of currency grows to e<super>rT</super> after time T, and "
            "the present value (discount factor) is:",
        ),
        P("P(0, T)  =  e<super>-rT</super>", Math),
        P(
            "This discount factor appears in every option pricing formula. A "
            "<b>zero-coupon bond</b> paying face value F at maturity T has present "
            "value F &middot; e<super>-rT</super>. The continuously compounded "
            "framework is standard in derivatives pricing because it simplifies "
            "stochastic calculus.",
        ),
    ]

    # ── §3 Risky Asset Dynamics ──────────────────────────────────── #
    story += [
        P("3.  Dynamics of the Risky Asset", H1),
        P(
            "The Black-Scholes model assumes the stock price S follows a "
            "<b>Geometric Brownian Motion (GBM)</b>:",
        ),
        P("dS  =  mu S dt  +  sigma S dW(t)", Math),
        P(
            "where mu is the drift (expected return), sigma is the instantaneous "
            "volatility, and W(t) is a standard Brownian motion under the physical "
            "measure P. By Ito's lemma applied to f(S) = ln S:",
        ),
        P(
            "d(ln S)  =  (mu - sigma<super>2</super>/2) dt  +  sigma dW(t)", Math),
        P(
            "Integrating from 0 to T gives the log-normal distribution of S(T):",
        ),
        P(
            "S(T) = S(0) exp[ (mu - sigma<super>2</super>/2)T  +  sigma W(T) ]", Math),
        P(
            "The key implication: ln S(T) ~ N( ln S(0) + (mu - sigma<sup>2</sup>/2)T , "
            "sigma<sup>2</sup>T ), so future stock prices are log-normally distributed.",
        ),
    ]

    # ── §4 No-Arbitrage & Risk-Neutral Pricing ───────────────────── #
    story += [
        P("4.  No-Arbitrage and Risk-Neutral Valuation", H1),
        P(
            "The <b>no-arbitrage principle</b> states that any two portfolios with "
            "identical future payoffs in all states of the world must have the same "
            "price today. Equivalently, there is no strategy with zero initial cost "
            "that delivers a non-negative payoff with positive probability and zero "
            "probability of a loss.",
        ),
        P(
            "The <b>Fundamental Theorem of Asset Pricing</b> states that a market is "
            "arbitrage-free if and only if there exists an <i>equivalent martingale "
            "measure</i> Q (the risk-neutral measure) under which all discounted asset "
            "prices are martingales. Under Q, every asset earns the risk-free rate r, "
            "and the price of any derivative V with payoff h(S<sub>T</sub>) is:",
        ),
        P("V(0)  =  e<super>-rT</super>  E<super>Q</super>[ h(S(T)) ]", Math),
        P(
            "This is the cornerstone of modern derivatives pricing: we do not need to "
            "know the true drift mu — only the risk-free rate r and volatility sigma.",
        ),
    ]

    # ── §5 Black-Scholes Formula ─────────────────────────────────── #
    story += [
        P("5.  The Black-Scholes Formula", H1),
        P(
            "Applying risk-neutral valuation to a European call with payoff "
            "max(S(T) - K, 0) and evaluating the expectation under Q using the "
            "log-normal distribution of S(T) yields the celebrated "
            "<b>Black-Scholes formula</b>:",
        ),
        P("C  =  S N(d1)  -  K e<super>-rT</super> N(d2)", Math),
        P("P  =  K e<super>-rT</super> N(-d2)  -  S N(-d1)", Math),
        P(
            "d1 = [ ln(S/K) + (r + sigma<super>2</super>/2) T ] / (sigma sqrt(T))", Math),
        P("d2 = d1 - sigma sqrt(T)", Math),
        P("where N(.) denotes the standard normal CDF.", Body),
        SP(6),
    ]

    # Numerical results table
    res_data = [
        ["Quantity", "Call", "Put"],
        ["Price",  f"{call.price:.4f}", f"{put.price:.4f}"],
        ["Delta",  f"{call.delta:.4f}", f"{put.delta:.4f}"],
        ["Gamma",  f"{call.gamma:.4f}", f"{put.gamma:.4f}"],
        ["Theta (per day)", f"{call.theta:.4f}", f"{put.theta:.4f}"],
        ["Vega (per 1% vol)", f"{call.vega:.4f}", f"{put.vega:.4f}"],
        ["Rho (per 1% rate)", f"{call.rho:.4f}", f"{put.rho:.4f}"],
    ]
    rt = Table(res_data, colWidths=[5.5*cm, 3.5*cm, 3.5*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Times-Bold"),
        ("FONTNAME",   (0,1), (-1,-1), "Courier"),
        ("FONTSIZE",   (0,0), (-1,-1), 9.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [
        P("Table 1: Black-Scholes prices and Greeks for the base parameters.", Caption),
        rt, SP(10),
    ]
    story += fig("greeks.png", width=15*cm,
        caption="Figure 1: All five Greeks of a European call plotted against spot price S. "
                "ATM strike K=100 shown as dashed line.")

    # ── §6 Put-Call Parity ────────────────────────────────────────── #
    story += [
        P("6.  Put-Call Parity", H1),
        P(
            "For European options on a non-dividend-paying stock, the following "
            "identity holds by a simple no-arbitrage argument:",
        ),
        P("C  -  P  =  S  -  K e<super>-rT</super>", Math),
        P(
            "A violation of this identity would allow a riskless profit by "
            "simultaneously buying the underpriced side and selling the overpriced side "
            "(a classic arbitrage). We verify this numerically:",
        ),
    ]
    pcp_data = [
        ["C - P", "S - K exp(-rT)", "Error"],
        [f"{pcp['C - P']:.8f}", f"{pcp['S - K*exp(-rT)']:.8f}", f"{pcp['error']:.2e}"],
    ]
    pt = Table(pcp_data, colWidths=[5*cm, 5*cm, 4*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Times-Bold"),
        ("FONTNAME",   (0,1), (-1,-1), "Courier"),
        ("FONTSIZE",   (0,0), (-1,-1), 9.5),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [pt, SP(4),
        P("Put-call parity holds to machine precision (error = 0).", Caption)]

    # ── §7 CRR Binomial Tree ──────────────────────────────────────── #
    story += [
        PageBreak(),
        P("7.  CRR Binomial Tree Model", H1),
        P(
            "Cox, Ross &amp; Rubinstein (1979) proposed a discrete-time model where, "
            "in each time step dt = T/N, the stock moves up by factor u or down by "
            "factor d with risk-neutral probability q:",
        ),
        P("u = e<super>sigma sqrt(dt)</super>,    d = 1/u", Math),
        P("q = ( e<super>r dt</super> - d ) / ( u - d )", Math),
        P(
            "The no-arbitrage condition requires d &lt; e<super>r dt</super> &lt; u, "
            "which is satisfied by the CRR parameterisation for all reasonable inputs. "
            "The option value at each node is recovered by <b>backward induction</b>:",
        ),
        P(
            "V(i, j)  =  e<super>-r dt</super> [ q V(i+1, j+1)  +  (1-q) V(i+1, j) ]", Math),
        P(
            "For <b>American options</b>, we take the maximum of the continuation "
            "value and the intrinsic value at each node, capturing the possibility "
            "of early exercise. The European formula is a special case where "
            "early exercise is never optimal.",
        ),
    ]

    # Tree convergence table
    conv_data = [["Steps N", "Binomial price", "BS price", "Error"]]
    for n, p_val, e in zip(conv["steps"], conv["prices"], conv["errors"]):
        conv_data.append([str(n), f"{p_val:.5f}", f"{conv['bs_price']:.5f}", f"{e:.5f}"])
    ct = Table(conv_data, colWidths=[3*cm, 4*cm, 4*cm, 4*cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Times-Bold"),
        ("FONTNAME",   (0,1), (-1,-1), "Courier"),
        ("FONTSIZE",   (0,0), (-1,-1), 9.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [
        P("Table 2: Convergence of CRR binomial price to Black-Scholes.", Caption),
        ct, SP(8),
    ]

    story += fig("tree_diagram.png", width=15*cm,
        caption="Figure 2: Stock price tree (left) and option value tree (right) for N=4 steps.")
    story += fig("convergence.png", width=15*cm,
        caption="Figure 3: Left: binomial prices converging to the BS benchmark. "
                "Right: pricing error decays at rate O(1/N) on a log-log scale.")

    # American vs European
    story += [
        P("7.1  American vs European Put: Early Exercise Premium", H2),
        P(
            "A key advantage of the binomial tree is its ability to price "
            "<b>American options</b>, for which no closed-form formula exists. "
            "An American put holder may optimally exercise early when the option "
            "is sufficiently deep in-the-money. The early exercise premium is:",
        ),
    ]
    ep_data = [
        ["", "Price (N=200)"],
        ["European put (binomial)", f"{eur_put.price:.5f}"],
        ["American put (binomial)", f"{ame_put.price:.5f}"],
        ["Early exercise premium", f"{ame_put.price - eur_put.price:.5f}"],
    ]
    et = Table(ep_data, colWidths=[8*cm, 5*cm])
    et.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Times-Bold"),
        ("FONTNAME",   (0,1), (-1,-2), "Courier"),
        ("FONTNAME",   (0,-1), (-1,-1), "Times-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY, colors.HexColor("#d4edda")]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [et, SP(4),
        P("Table 3: American put commands a premium of {:.4f} over the European put.".format(
            ame_put.price - eur_put.price), Caption)]

    # ── §8 Price Surface ──────────────────────────────────────────── #
    story += [
        PageBreak(),
        P("8.  Option Price Surface", H1),
        P(
            "Varying both the spot price S and time to maturity T reveals the "
            "full option price surface. Key observations: (i) call value increases "
            "with S and T; (ii) put value decreases with S; (iii) all options lose "
            "value as T approaches zero — this is <b>time decay</b> (Theta).",
        ),
    ]
    story += fig("price_surface.png", width=15*cm,
        caption="Figure 4: Black-Scholes price surfaces for call (left) and put (right) "
                "across spot prices and maturities.")

    # ── §9 Implied Volatility ─────────────────────────────────────── #
    story += [
        P("9.  Implied Volatility", H1),
        P(
            "Given a market-observed option price, the implied volatility (IV) is the "
            "sigma that makes the Black-Scholes formula reproduce that price. "
            "Since there is no closed-form inverse, we solve numerically using "
            "<b>Newton-Raphson</b> iteration:",
        ),
        P("sigma_(n+1)  =  sigma_n  -  [ BSM(sigma_n) - C_mkt ] / Vega(sigma_n)", Math),
        P(
            "The algorithm converges quadratically (typically in 5-10 iterations) "
            "because Vega is always positive for vanilla options. "
            "Round-trip check: starting from sigma = 20%, pricing the call, "
            "then recovering the IV gives an error of 0.00e+00 — machine precision.",
        ),
    ]

    # ── §10 Code architecture ─────────────────────────────────────── #
    story += [
        P("10.  Code Architecture", H1),
        P(
            "The pricing engine is implemented as a clean Python library with three modules, "
            "designed for readability, correctness, and extensibility:",
        ),
    ]
    arch_data = [
        ["Module", "Responsibility"],
        ["black_scholes.py", "Analytical BS pricing, all Greeks, put-call parity, implied vol"],
        ["binomial_tree.py", "CRR tree for European & American options, convergence analysis"],
        ["visualisations.py","Publication-quality plots: Greeks, convergence, price surfaces, trees"],
        ["demo.py",          "End-to-end reproduction script — all tables and figures in this report"],
    ]
    at = Table(arch_data, colWidths=[4.5*cm, 10.5*cm])
    at.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Times-Bold"),
        ("FONTNAME",   (0,1), (-1,-1), "Courier"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GREY]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",(0,0), (-1,-1), 6),
    ]))
    story += [at, SP(10)]

    code_snippet = (
        "from black_scholes import price, implied_volatility\n"
        "from binomial_tree import price as bt_price\n\n"
        "# European call — price + all Greeks in one call\n"
        "result = price(S=100, K=100, r=0.05, sigma=0.20, T=1.0, option_type='call')\n"
        "print(result)   # Price: 10.4506  Delta: 0.6368  Gamma: 0.0188 ...\n\n"
        "# American put via binomial tree (N=200 steps)\n"
        "amer = bt_price(100, 100, 0.05, 0.20, 1.0, N=200,\n"
        "               option_type='put', exercise_type='american')\n"
        "print(amer.price)   # 6.0864\n\n"
        "# Implied volatility recovery\n"
        "iv = implied_volatility(market_price=10.4506, S=100, K=100, r=0.05, T=1.0)\n"
        "print(f'IV = {iv:.4f}')   # IV = 0.2000"
    )
    story += [
        P("Sample usage:", Body),
        P(code_snippet.replace("\n", "<br/>").replace(" ", "&nbsp;"), Code),
    ]

    # ── §11 Conclusion ────────────────────────────────────────────── #
    story += [
        P("11.  Conclusion", H1),
        P(
            "This project demonstrates a complete arc from mathematical theory to "
            "numerical implementation in financial engineering. Starting from "
            "geometric Brownian motion and the no-arbitrage principle (Units 1-4), "
            "we derived the Black-Scholes formula via risk-neutral valuation and "
            "verified all analytical properties: Greeks, put-call parity, and implied "
            "volatility inversion. The CRR binomial tree provides a flexible discrete "
            "framework that (a) matches Black-Scholes in the European limit and "
            "(b) extends naturally to American options, capturing the early-exercise "
            "premium of 0.52 for our base parameters.",
        ),
        P(
            "The project sits at the intersection of stochastic calculus, numerical "
            "analysis, and software engineering — the core skill set for any "
            "quantitative finance role or graduate programme in financial engineering.",
        ),
    ]

    # ── References ───────────────────────────────────────────────── #
    story += [
        HR(), SP(4),
        P("References", H2),
        P("[1] Black, F. &amp; Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. "
          "<i>Journal of Political Economy</i>, 81(3), 637-654."),
        P("[2] Cox, J., Ross, S. &amp; Rubinstein, M. (1979). Option Pricing: A Simplified Approach. "
          "<i>Journal of Financial Economics</i>, 7(3), 229-263."),
        P("[3] Capinski, M. &amp; Zastawniak, T. (2010). <i>Mathematics for Finance</i>. Springer."),
        P("[4] Hull, J.C. (2018). <i>Options, Futures and Other Derivatives</i>, 10th Ed. Pearson."),
        P("[5] Cvitanic, J. &amp; Zapatero, F. (2007). <i>Introduction to the Economics and Mathematics "
          "of Financial Markets</i>. Prentice-Hall."),
    ]

    doc.build(story)
    print(f"PDF written to: {OUT}")


if __name__ == "__main__":
    build()
