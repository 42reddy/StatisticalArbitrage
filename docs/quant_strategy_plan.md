# Quantitative Strategy Directions — Small Capital, Daily Frequency

> **Context:** Physics background, ~few thousand USD capital, daily frequency, high relative transaction costs,
> India-based (NSE/BSE access), existing expertise in OU processes, Bayesian optimization, walk-forward
> validation. Currently running a HAL/BDL stat arb model in paper trading.

---

## Framing the Problem Correctly

Before picking a strategy, the binding constraints need to be explicit:

| Constraint | Implication |
|---|---|
| Small capital (~$2–5K USD) | Cannot absorb large drawdowns. Kelly sizing is aggressive — use fractional Kelly. |
| High relative transaction costs | Strategies must trade infrequently. Each round-trip on NSE eats ~0.05–0.1% in STT + brokerage. At daily freq, that's ~12–25% annualized just in costs. |
| No colocation / low latency infra | Sub-minute alpha is inaccessible. Daily close-to-close or end-of-day signals only. |
| India market access | NSE equity futures, commodity futures (MCX), and equity cash market are the realistic universe. |
| Interpretable math only | Every parameter must have a prior intuition. No black-box ensembles. |

The strategies below are chosen specifically because they **survive transaction costs at daily frequency**, have
**small minimum capital requirements**, and have **mathematical structure you can own end-to-end**.

---

## Direction 1: Cross-Sectional Mean Reversion in Equity Futures

### The Core Idea

Among a basket of liquid NSE equity futures (say, 30–50 stocks in the F&O segment), stocks that have
deviated strongly from the cross-sectional mean over 1–5 days tend to partially revert. This is the
**short-term reversal anomaly** — one of the oldest documented return patterns, present in every equity
market studied, and especially strong in emerging markets where retail overreaction is high.

This is *not* the same as momentum (which works over 1–12 months). It is the **opposite**: you fade
recent extreme moves.

### The Math

**Signal construction:**

Define the cross-sectional z-score for stock i on day t:

```
z_i(t) = [r_i(t-k:t) - mean_j(r_j(t-k:t))] / std_j(r_j(t-k:t))
```

where `r_i` is the cumulative return over the past `k` days (typically k = 1 to 5), and the mean/std
are taken across all stocks in the universe on that day. You go **long** stocks with the most negative
z-score and **short** stocks with the most positive z-score.

**Why it works (mechanism):**
- Retail-driven overreaction creates short-term price pressure that dissipates over 2–5 days.
- Market makers and short-term arbitrageurs provide the supply/demand shock; you are the mean-reversion
  force on the other side.
- The OU analogy is exact: each stock's deviation from the cross-sectional mean follows an approximate
  OU process with a very fast mean-reversion speed (half-life of 2–4 days).

**Position sizing — inverse volatility weighting:**

```
w_i = sign(-z_i) * (1/σ_i) / Σ_j (1/σ_j)
```

This is **risk parity at the single-stock level** — each position contributes equal volatility, not
equal notional. Elegant and well-motivated: you are not betting more on a noisy stock just because it
has a large price deviation.

**Turnover and cost control:**

The key problem with short-term reversal is that it is *high turnover* if implemented naively. You must
build in a **transaction cost buffer**: only trade if |z_i| > threshold τ, where τ is calibrated so
that expected alpha exceeds round-trip costs. Concretely:

```
Trade only if |z_i| > τ, where τ is chosen so E[alpha | trade] > 2 * cost_per_trade
```

At NSE, with ~0.05% round-trip for futures: set τ empirically by backtesting at different thresholds
on 5-minute or EOD data. Typically τ ≈ 1.5 to 2.0 σ works.

### Implementation Path

1. Pull EOD futures prices for the NSE F&O segment (~150 stocks). Free sources: NSE Bhavcopy (daily
   CSV from nseindia.com), or use `nsepy` / `jugaad-data` Python libraries.
2. Filter to liquid contracts only: open interest > ₹50 crore, to avoid illiquidity slippage.
3. Compute rolling z-scores over lookback windows [1, 2, 3, 5] days. Walk-forward optimize the
   lookback using your existing Bayesian optimization framework.
4. Rank and select top/bottom decile. Size positions using inverse-vol weights.
5. Backtest on 3 years of data. Hold out the last 6 months strictly as OOS.

### What Makes This Suitable

- **No leverage needed to generate returns.** Long-short structure reduces market beta.
- **The universe is large enough** (~150 F&O stocks) to have meaningful breadth. Sharpe scales with
  breadth even if individual signal IC is low.
- **Your OU toolkit applies directly** — you can formally estimate mean-reversion speed per stock and
  use it to weight signals.
- **Walk-forward validation** is standard here and you already know how to do it.

### Key Risks

- **Crowding:** Short-term reversal strategies in NSE are used by prop desks. In periods of market
  stress, everyone unwinds together, causing drawdowns.
- **Borrow costs for short side:** In the cash segment, shorting requires futures or SLB (Securities
  Lending and Borrowing). Futures short is simpler but introduces roll cost.
- **Regime sensitivity:** In trending markets (strong bull run), reversal signals fail consistently.
  Add a regime filter: if the index (Nifty 50) is trending strongly (ADX > 25 or returns > 2σ over
  10 days), reduce position size by 50%.

---

## Direction 2: Calendar Spread Mean Reversion in Commodity Futures

### The Core Idea

A calendar spread is the price difference between two futures contracts on the same underlying but
different expiries — say, MCX Crude Oil Near-Month minus MCX Crude Oil Next-Month. This spread is
governed by cost-of-carry theory and mean-reverts around a fundamental value driven by storage costs,
convenience yield, and interest rates.

This is a **natural extension of your HAL/BDL stat arb** into a single-instrument structure where
the cointegration relationship is theoretically motivated rather than empirically hunted.

### The Math

Under cost-of-carry:

```
F(T2) - F(T1) = F(T1) * [e^(r + u - y) * (T2 - T1) - 1]
```

where `r` is the risk-free rate, `u` is storage cost, `y` is convenience yield, and T2 > T1.

In practice, the spread `S(t) = F(T1,t) - F(T2,t)` fluctuates around this theoretical value. You
model deviations as an OU process:

```
dS = κ(μ - S)dt + σ dW
```

This is *exactly* your existing framework. Estimate κ (mean-reversion speed), μ (long-run mean),
and σ (diffusion) using MLE or your existing Bayesian approach. Entry and exit thresholds are set
at ±1.5σ and ±0.5σ of the OU stationary distribution, just as in pair trading.

**Key difference from equity pair trading:** The spread S(t) here has a *structural* anchor — the
cost-of-carry equation gives you a prior on μ. You are not purely relying on statistical stationarity
that might break. When storage costs or convenience yield shifts (oil glut, harvest cycles), you can
update μ explicitly.

### Implementation Path

1. Get MCX futures data: MCX provides EOD bhavcopy (free). Python library `mcxpy` or direct CSV
   download. Focus on: Crude Oil, Gold, Silver, Natural Gas — all have near and next-month contracts.
2. For each commodity, compute the near-next spread daily.
3. Fit OU parameters using a rolling 60-day window. Use your existing MLE/Bayesian estimator.
4. Generate signals: enter long spread when S < μ - 1.5σ, enter short spread when S > μ + 1.5σ.
   Exit at μ ± 0.5σ.
5. Walk-forward validation: 180-day training, 30-day test, roll forward monthly.

### What Makes This Suitable

- **Exact match to your existing skillset.** OU process, Bayesian parameter estimation, walk-forward
  validation — everything maps 1:1 from HAL/BDL.
- **Low transaction costs.** You are trading two legs of the *same* commodity. Many brokers offer
  calendar spread margins (SPAN reduced margin) — you may need only 30–50% of the normal margin.
- **Structural justification.** The mean-reversion in a calendar spread is *theoretically* grounded,
  not just empirically fitted. This makes it more robust out-of-sample.
- **Small capital.** MCX Crude Oil mini contracts (100 barrels) require ~₹15,000–20,000 margin. Gold
  mini is similar. This fits your capital budget.

### Key Risks

- **Delivery risk near expiry.** Commodity futures can behave erratically in the last 5 days before
  expiry. Always roll out of near-month contracts 5 days before expiry.
- **Structural regime shifts.** Convenience yield can shift suddenly (OPEC cuts, harvest failure).
  Track the cost-of-carry implied μ and update it; do not use a static μ over long periods.
- **Liquidity in next-month contracts.** Next-month MCX contracts are sometimes thinly traded.
  Check open interest before entering — require OI > 500 lots.

---

## Direction 3: Index Arbitrage via ETF-Futures Basis

### The Core Idea

Nifty 50 futures must, by no-arbitrage, trade at a premium to the Nifty 50 spot index equal to
the cost of carry (risk-free rate minus dividend yield, prorated for time to expiry). In practice,
this **basis** fluctuates around its fair value. When it deviates significantly — futures too rich or
too cheap relative to the theoretical fair value — it is a mean-reversion opportunity.

The implementation: trade the basis between **Nifty 50 futures** and **Nifty 50 ETF** (e.g., Nippon
India Nifty BeES, the oldest and most liquid Indian ETF with AUM > ₹5,000 crore).

### The Math

**Theoretical futures price:**

```
F_fair(t, T) = S(t) * e^(r - d)(T - t)
```

where S(t) is the ETF NAV (proxy for spot), r is the 91-day T-bill rate, d is the dividend yield
of Nifty 50, and (T - t) is time to expiry in years.

**Basis:**

```
Basis(t) = F_actual(t) - F_fair(t, T)
```

This should hover near zero. When |Basis(t)| > threshold (set empirically at ~0.3–0.5% of spot),
it is tradeable.

**The OU model on basis:**

```
dB = κ(0 - B)dt + σ dW
```

Here μ = 0 (or a small positive value for cost friction). Estimate κ and σ from historical daily
basis data. The mean-reversion is often *faster* than equity pairs — half-life of 1–3 days — which
means this is a very short-duration trade, reducing exposure time.

**Signal:**
- Basis > +threshold: Futures are too rich. Short futures, buy ETF (cash).
- Basis < -threshold: Futures are too cheap. Buy futures, sell ETF (or hold existing ETF as the
  long leg and overlay the futures short).

### Why This Is Different from "Index Arb for HFTs"

Pure index arbitrage (replicating the index with all 50 stocks) is HFT territory. This strategy is
different: you use the ETF as the spot proxy, which trades in a single transaction. The trade is
daily-frequency — you compute the basis at EOD, decide, and execute at next open. No speed required.

Execution lag of hours is fine because the basis at daily frequency has autocorrelation — it does
not snap back within minutes at the EOD level.

### Implementation Path

1. Data: Pull Nifty 50 futures EOD (NSE bhavcopy) + Nifty BeES ETF EOD price + 91-day T-bill rate
   (RBI website, updated weekly). Compute daily F_fair and Basis.
2. Estimate OU parameters on 90-day rolling basis time series.
3. Entry: |Basis| > 1.5σ_basis. Exit: |Basis| < 0.3σ_basis or at expiry roll.
4. Sizing: Keep the ETF leg and futures leg notionally equal. Start with 1 lot of Nifty futures
   (~₹5–6 lakh notional, ~₹50,000 margin) and equivalent ETF holding.
5. Walk-forward test on 3 years of data. Roll futures 5 days before expiry.

### What Makes This Suitable

- **Near-perfect cointegration.** ETF and futures on the same index — this is as close as you can
  get to a theoretically justified pair. No model risk on the spread structure.
- **Very fast mean-reversion** reduces overnight exposure. You are rarely in a trade for more than
  3–5 days.
- **No stock-picking, no sector risk.** The only risk is the basis itself, not the direction of the
  index.
- **Margin efficiency.** Nifty futures margins are ~8–10% of notional. The ETF is your cash holding,
  not a cost — you own it anyway. The only margined leg is the futures.

### Key Risks

- **Dividend timing.** When a large Nifty constituent goes ex-dividend, the futures price adjusts
  instantaneously but the ETF NAV adjusts with a day's lag. Do not confuse this for a basis
  opportunity. Track the Nifty dividend calendar.
- **Roll cost.** Monthly rolling of futures costs ~0.05–0.1% each roll. Factor this into the
  threshold calibration.
- **ETF tracking error.** Nifty BeES has small but non-zero tracking error (~0.05% per day).
  This creates noise in the basis signal. Use a 3-day moving average of the basis to smooth it.

---

## Comparative Summary

| | Cross-Sec Reversal | Calendar Spread | ETF-Futures Basis |
|---|---|---|---|
| **Anomaly type** | Behavioral overreaction | Cost-of-carry deviation | No-arb basis deviation |
| **Universe** | 50–150 NSE F&O stocks | MCX commodities (4–5) | Nifty 50 only |
| **Signal math** | Cross-sectional z-score | OU process on spread | OU process on basis |
| **OU connection** | Per-stock mean reversion | Direct carry-adjusted OU | Exact OU on basis |
| **Trade duration** | 2–5 days | 3–10 days | 1–5 days |
| **Min. capital needed** | ₹1–2 lakh (multi-leg) | ₹30–50K (2 legs, MCX mini) | ₹1–1.5 lakh (1 lot + ETF) |
| **Transaction cost sensitivity** | High (many legs) | Low (2 legs same asset) | Low (2 legs same index) |
| **Theory anchor** | Empirical, behavioral | Cost-of-carry equation | No-arbitrage identity |
| **Extends your HAL/BDL work** | Partially | Directly | Directly |
| **Biggest risk** | Crowding in stress | Delivery, regime shift | Dividend timing errors |

---

## Recommended Sequencing

### Phase 1 — Months 1–2 (While HAL/BDL is still in paper trading)

Start with **Direction 3 (ETF-Futures Basis)**. It has the clearest theory, the fewest moving parts,
and the math is a direct port of your OU estimator. The data is free and clean. Build the estimator,
run the walk-forward backtest, then paper trade it alongside HAL/BDL. This gives you a second model
in paper trading with very little new infrastructure.

### Phase 2 — Months 3–4

Add **Direction 2 (Calendar Spreads)** on MCX Crude and Gold. Again, your OU framework applies
directly. Two commodities gives you 2 independent signals. Calibrate the OU parameters separately
per commodity and use independent position limits.

### Phase 3 — Months 5–8

Once Directions 2 and 3 are live (or near live), approach **Direction 1 (Cross-Sectional
Reversal)**. It requires more infrastructure (universe management, rolling liquidity filter, 
multi-leg execution) and is more sensitive to transaction costs. Build it last, when you have
a clearer picture of your actual execution costs from real trading.

---

## Notes on Capital Allocation Across Strategies

With a few thousand USD (~₹2–4 lakh), a simple allocation that works:

- Keep **50% in liquid cash or liquid fund** at all times. This is your drawdown buffer and your
  margin top-up reserve. Do not deploy it into trades.
- Allocate **25% to Direction 3** (ETF basis trade): primarily sits in BeES ETF with an occasional
  futures overlay. This is low-risk and the ETF is a productive use of cash anyway.
- Allocate **25% to Direction 2** (calendar spread): MCX mini contracts are small enough to fit here.

Direction 1 is added only when capital grows above ₹5–6 lakh, because multi-leg execution in F&O
requires meaningful notional to dilute transaction costs.

**Position sizing rule (across all strategies):** Use **half-Kelly** as the upper bound. If Kelly
says bet 8% of capital on a signal, bet 4%. The Kelly criterion assumes you know the true edge
distribution — you don't. Half-Kelly gives up some expected return in exchange for dramatically
lower ruin probability. For someone starting out, avoiding ruin is more important than maximizing
growth rate.

---

## Common Infrastructure You Can Build Once and Reuse

Since all three strategies share OU-based signal generation:

```python
# Shared OU parameter estimator (MLE)
def fit_ou(spread: np.ndarray, dt: float = 1.0) -> dict:
    """
    Fit OU process: dX = kappa*(mu - X)*dt + sigma*dW
    Uses maximum likelihood on discretized Euler scheme.
    Returns: {kappa, mu, sigma, half_life, z_score}
    """
    n = len(spread)
    x = spread[:-1]
    y = spread[1:]
    # OLS on discretized OU: y = a + b*x + eps
    b, a = np.polyfit(x, y, 1)
    kappa = -np.log(b) / dt
    mu = a / (1 - b)
    residuals = y - (a + b * x)
    sigma = np.std(residuals) * np.sqrt(2 * kappa / (1 - b**2))
    half_life = np.log(2) / kappa
    z = (spread[-1] - mu) / (sigma / np.sqrt(2 * kappa))
    return dict(kappa=kappa, mu=mu, sigma=sigma, half_life=half_life, z_score=z)
```

This function is the same for Direction 2 (spread = calendar spread), Direction 3 (spread = basis),
and per-stock in Direction 1 (spread = stock deviation from cross-sectional mean). Build it once,
test it thoroughly, and plug it into each strategy module.

---

*Last updated: April 2026*
