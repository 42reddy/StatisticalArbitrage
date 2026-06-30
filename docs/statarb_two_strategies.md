# Two Comprehensive Stat Arb Strategies
### For a Retail Quant — Small Capital, Daily Frequency, NSE Universe

> **Baseline context:** You already run an OU-based log-price spread model on HAL/BDL (defence PSU
> pair). Both strategies below use identical mathematical machinery — OU process, z-score entry/exit,
> Bayesian parameter optimization, walk-forward validation — but differ fundamentally in how the
> spread is *constructed* and *what gives it the right to mean-revert*. That structural difference
> is what makes them genuinely distinct strategies, not just different pairs.

---

---

# STRATEGY A
# Sector-Residual Mean Reversion

---

## 1. Core Intuition

In your HAL/BDL model, the spread is:

```
Spread(t) = log(HAL) − β × log(BDL)
```

You are removing the common factor (defence sector movement) from one stock using the other as
a proxy for that factor. The residual is what you trade.

This strategy makes that logic explicit and generalises it. Instead of using one stock as a proxy
for another, you use the *actual sector index* as the common factor. For every stock in your
universe, you run:

```
r_stock(t) = α + β × r_sector(t) + ε(t)
```

The residual ε(t) is the stock's daily return that cannot be explained by its sector moving up or
down. If HDFC Bank rises 2% on a day the Nifty Bank index rises 2%, ε ≈ 0 — the stock just moved
with its sector. If HDFC Bank rises 2% while the sector is flat, ε ≈ 2% — the stock has done
something idiosyncratic.

You cumulate these daily residuals into a **residual price series**:

```
R(t) = Σ_{s=1}^{t} ε(s)      [cumulated residual log-price]
```

This series has no trend by construction (you've removed the sector drift). It mean-reverts to zero.
You fit an OU process to R(t) and trade it exactly as you trade your existing spread.

**Why does it mean-revert structurally, not just statistically?**

A stock cannot permanently deviate from its sector without a fundamental reason. If it drifts up
persistently relative to its sector, either: (a) there is a real fundamental repricing happening
(earnings revision, M&A, regulatory change), in which case you should not trade it back, or (b) it
is noise, positioning, or a temporary liquidity imbalance, in which case it reverts. Your OU
half-life estimate is the mechanism that distinguishes fast-reverting noise from slow-moving
fundamental repricing — you only enter when the half-life is short enough that reversion will happen
before transaction costs eat the position.

---

## 2. Universe Construction

**Step 1 — Choose sectors with internal homogeneity.**

Not all sectors produce tradeable residuals. You want sectors where:
- The sector index explains a large fraction of individual stock variance (R² of the sector
  regression > 0.40). Low R² means the stock is driven by idiosyncratic factors anyway, and
  the residual has no clean mean-reversion anchor.
- There are at least 5 liquid stocks in the sector (so you have rotation opportunities).
- The stocks are all in the NSE F&O segment (so you can short without borrowing complications).

**Recommended starting sectors on NSE:**

| Sector | Index to use | Candidate stocks | Typical R² |
|---|---|---|---|
| Private Banking | Nifty Bank | HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK, INDUSINDBK | 0.55–0.75 |
| Public Sector Banking | Nifty PSU Bank | SBIN, BANKBARODA, PNB, CANARABANK, UNIONBANK | 0.60–0.80 |
| IT Services | Nifty IT | TCS, INFY, WIPRO, HCLTECH, TECHM | 0.50–0.70 |
| Pharma | Nifty Pharma | SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, AUROPHARMA | 0.35–0.55 |
| Auto | Nifty Auto | MARUTI, TATAMOTORS, M&M, BAJAJ-AUTO, EICHERMOT | 0.40–0.60 |

**Start with PSU Banking and IT.** PSU Bank stocks have the highest R² (government policy is the
dominant common factor), giving the cleanest residuals. IT stocks have high R² driven by USD/INR
and US demand cycles. Both sectors have liquid F&O contracts.

**Step 2 — Liquidity filter.**

For each candidate stock, require:
- F&O open interest > ₹100 crore (ensures you can exit without significant market impact)
- Average daily volume (cash + futures) > ₹50 crore
- Futures lot value < ₹5 lakh (keeps single-lot exposure manageable for small capital)

Recheck this filter monthly. Stocks rotate in and out of liquidity thresholds.

---

## 3. Spread Construction in Detail

For each stock i in sector S, on each trading day t:

**Step 1 — Estimate the rolling sector beta.**

Use a rolling OLS window of 60 trading days (approximately 3 months):

```python
import numpy as np

def rolling_sector_beta(stock_returns: np.ndarray,
                        sector_returns: np.ndarray,
                        window: int = 60) -> np.ndarray:
    """
    Returns rolling OLS beta of stock on sector index.
    Uses only past data — no lookahead.
    """
    betas = np.full(len(stock_returns), np.nan)
    for t in range(window, len(stock_returns)):
        y = stock_returns[t-window:t]
        x = sector_returns[t-window:t]
        # OLS: beta = cov(x,y)/var(x)
        betas[t] = np.cov(x, y)[0, 1] / np.var(x)
    return betas
```

Why 60 days? Short enough to capture regime changes in sector sensitivity (a bank acquiring an NBFC
changes its beta). Long enough to have stable OLS estimates (you need ~30+ observations for OLS
to be reliable, 60 gives comfortable stability).

**Step 2 — Compute daily residuals.**

```python
def compute_residuals(stock_log_prices: np.ndarray,
                      sector_log_prices: np.ndarray,
                      beta: np.ndarray) -> np.ndarray:
    stock_ret = np.diff(stock_log_prices)
    sector_ret = np.diff(sector_log_prices)
    # alpha is negligible at daily frequency, set to zero
    residuals = stock_ret - beta[1:] * sector_ret
    return residuals
```

Setting alpha to zero at daily frequency is justified: daily alpha (≈ risk premium / 252) is
numerically negligible compared to the residual signal. Estimating alpha adds noise without
adding information.

**Step 3 — Cumulate residuals into a tradeable spread.**

```python
def cumulated_residual(residuals: np.ndarray,
                       reset_window: int = 120) -> np.ndarray:
    """
    Cumulate residuals over a rolling window.
    Reset every `reset_window` days to prevent drift accumulation.
    """
    R = np.zeros(len(residuals))
    for t in range(len(residuals)):
        start = max(0, t - reset_window)
        R[t] = np.sum(residuals[start:t+1])
    return R
```

The reset window is important. If you simply cumulate ε(t) from the beginning of time, you pick
up low-frequency drift (the beta estimation is imperfect; small systematic errors accumulate over
years). Resetting every 120 days means R(t) is always the 6-month cumulated idiosyncratic
deviation — a cleaner signal.

**Step 4 — Fit OU process to R(t).**

Use your existing `fit_ou()` function directly. Feed it R(t) instead of the log-price spread.
The OU parameters have identical interpretation:
- κ (mean-reversion speed): how quickly the stock reverts to its sector after a deviation
- μ (long-run mean): should be near zero by construction; if it is persistently non-zero, the
  beta estimation is biased — check your OLS
- σ (diffusion): magnitude of idiosyncratic noise
- Half-life = ln(2)/κ: use this as a sanity check. Reject any stock where half-life > 30 days
  (mean reversion is too slow to trade profitably before transaction costs dominate)

---

## 4. Signal Generation and Trade Selection

Unlike your HAL/BDL model (which has one spread and trades it whenever the z-score threshold is
crossed), this strategy has **N stocks across M sectors**, each with its own residual and its own
OU z-score. You need a ranking and selection mechanism.

**Step 1 — Compute z-scores across the universe.**

At end of each trading day, for each stock i:

```python
z_i(t) = (R_i(t) - mu_i) / sigma_i_stationary
```

where `sigma_i_stationary = sigma_i / sqrt(2 * kappa_i)` is the stationary standard deviation
of the OU process (not the rolling std of R, which is non-stationary during entry/exit).

**Step 2 — Apply OU quality filter before trading.**

Only consider stock i for trading if:
- ADF p-value on R_i(last 90 days) < 0.10 (residual is stationary)
- Half-life: 3 days < hl_i < 25 days (fast enough to trade, not so fast that you can't execute)
- R² of the sector regression > 0.35 (sector explains enough variance that the residual is
  meaningful)
- |z_i(t)| > entry threshold (signal is strong enough to overcome transaction costs)

The half-life bounds are critical. Below 3 days, you're competing with intraday momentum traders
and your EOD execution is too slow. Above 25 days, the position ties up capital for too long
and costs accumulate.

**Step 3 — Rank and select.**

Sort all stocks passing the filter by |z_i|, descending. Select the top K stocks, where K is
determined by your capital:

```
K = floor(available_capital / (min_lot_notional × 2))
```

The factor of 2 accounts for both legs (long the residual = long stock + short sector ETF as hedge).
With ₹2 lakh available capital and typical lot notional of ₹3–5 lakh, K = 1–2 positions at a time.
That is fine — concentrate in the highest-conviction signals.

**Step 4 — Direction of trade.**

- z_i < −entry_threshold: residual is below OU mean. Stock has underperformed its sector.
  **Long the stock, short the sector ETF** (hedge ratio = β_i × notional_stock).
- z_i > +entry_threshold: residual is above OU mean. Stock has outperformed its sector.
  **Short the stock, long the sector ETF.**

The sector ETF leg is the key difference from your HAL/BDL model. Instead of pairing with another
individual stock, you are pairing with the sector index itself. This hedge is *cleaner* because
the sector index is diversified — no single-stock event can blow it up.

**Available sector ETFs on NSE:**
- Nifty Bank: Nippon India Bank BeES (BANKBEES)
- Nifty PSU Bank: SBI PSU Bank ETF or Nippon PSU Bank ETF
- Nifty IT: Nippon India ETF Nifty IT (ITBEES)
- Nifty Pharma: Nippon India Pharma ETF

---

## 5. Position Sizing

This strategy requires more careful sizing than a single pair because you are running multiple
simultaneous positions across different stocks.

**Per-position sizing:**

```python
def position_size(capital: float,
                  stock_price: float,
                  lot_size: int,
                  sigma_stationary: float,
                  z_score: float,
                  kappa: float,
                  max_positions: int = 2,
                  half_kelly_fraction: float = 0.5) -> int:
    """
    Returns number of lots to trade.
    Uses Kelly criterion scaled by OU parameters.
    """
    # Expected holding period return (OU mean reversion to zero)
    # E[return] ≈ |z| * sigma_stationary (distance to mean × vol)
    expected_return = abs(z_score) * sigma_stationary

    # Expected variance over holding period (1/kappa days)
    holding_period = 1.0 / kappa  # in days
    expected_var = sigma_stationary**2 * holding_period

    # Kelly fraction
    kelly_f = expected_return / expected_var

    # Half-Kelly, further divided by max_positions (capital sharing)
    adjusted_f = (half_kelly_fraction * kelly_f) / max_positions

    # Notional to deploy
    notional = capital * adjusted_f
    lots = max(1, int(notional / (stock_price * lot_size)))
    return lots
```

**Sector ETF hedge sizing:**

The ETF leg is sized to neutralise the sector exposure:

```
ETF_units = (stock_lots × lot_size × stock_price × β_i) / ETF_NAV
```

Round to the nearest ETF unit. ETFs trade in single units, so this is always achievable.

**Capital allocation rule:**

Never deploy more than 60% of capital across all open positions simultaneously. Keep 40% as:
- 20% in liquid fund or overnight FD (productive cash, earns ~6.5% annualised)
- 20% as margin buffer (futures margins can increase intraday; you need headroom)

---

## 6. Exit Logic

**Primary exit — OU mean reversion:**
Close when z_i crosses back through exit_threshold toward zero (same as HAL/BDL model).
Set exit_threshold = 0.2σ. You do not wait for full reversion to zero — this is the
well-known result that the optimal OU exit is before the mean, not at it, because the
reversion slows near the mean and holding costs accumulate.

**Secondary exit — Stop loss:**
Close if z_i moves further from zero and crosses stop_threshold = 3.0σ in the adverse direction.
This is a *regime change signal* — the residual is not reverting; something fundamental has changed
(earnings surprise, regulatory action, M&A announcement). Exit immediately.

**Tertiary exit — Time stop:**
Close any position that has been open for more than max_hold = 20 trading days, regardless of
z-score. A position that has not reverted in 20 days is not reverting on this cycle. The OU
model has failed for this instance. Take whatever P&L is available and exit.

**Event-driven exit — Mandatory:**
Implement an earnings calendar filter. Before entering any position, check if the stock has an
earnings announcement within the next 10 days. If yes, skip the entry. Earnings releases create
large, non-mean-reverting residual spikes that will trigger your z-score entry, but the subsequent
move is fundamental, not noise. Entering around earnings is one of the most common ways retail
stat arb models lose money.

```python
def earnings_blackout(entry_date, earnings_dates: list,
                      pre_blackout: int = 10,
                      post_blackout: int = 2) -> bool:
    """Returns True if entry is blacklisted due to upcoming/recent earnings."""
    for ed in earnings_dates:
        days_to = (ed - entry_date).days
        if -post_blackout <= days_to <= pre_blackout:
            return True
    return False
```

---

## 7. Parameter Optimization and Walk-Forward Validation

**Parameters to optimize (per stock, per sector):**

| Parameter | Search range | Notes |
|---|---|---|
| beta_window | [40, 90] | Rolling OLS window for sector beta |
| reset_window | [60, 180] | Cumulated residual reset period |
| z_entry | [1.2, 2.5] | Entry threshold (can be asymmetric long/short) |
| z_exit | [0.1, 0.5] | Exit threshold |
| z_stop | [2.5, 4.0] | Stop loss threshold |
| max_hold | [10, 30] | Time stop in days |

**What NOT to optimize per-stock:**
Do not optimize the OU fitting window or the liquidity filter thresholds. These are structural
parameters shared across all stocks. Optimizing them per-stock introduces enormous overfitting
risk because you have at most 50–100 trades per stock in a 5-year history.

**Walk-forward setup:**

Use the same WFV structure as your HAL/BDL model, but at the **sector level**, not the stock
level:

```
Training fold   : 18 months of data
Test fold       : 3 months of OOS
Step size       : 3 months (anchored expanding window)
Minimum folds   : 8 (requires ~42 months of data)
```

Optimize one set of parameters per sector. Apply that parameter set uniformly to all stocks in
the sector during the test fold. This forces genuine generalization — you cannot learn
stock-specific quirks in training.

**Objective function:**

Use the same composite score you are currently using, but add a breadth penalty:

```
Score = Sharpe_OOS − λ × (1 / N_trades)
```

where N_trades is the total number of trades taken in the test fold and λ = 0.5. This penalizes
parameter sets that trigger very few trades (which often means the threshold is too conservative
and the backtest P&L is based on 3–4 lucky trades — not a generalizable result).

---

## 8. Risk Management at Portfolio Level

Because this strategy runs multiple simultaneous positions (potentially across multiple sectors),
you need portfolio-level risk controls that do not exist in your single-pair HAL/BDL model.

**Sector concentration limit:**
Never have more than 2 open positions in the same sector simultaneously. If 3 stocks in PSU Banking
all show extreme z-scores at the same time, this is likely a sector-wide event (RBI policy, credit
cycle), not idiosyncratic noise. Pick the top 2 by |z| and ignore the third.

**Correlation monitoring:**
Compute the pairwise correlation of open position residuals daily. If two open positions have
residual correlation > 0.6, they are responding to the same factor — you effectively have a
concentrated bet. Reduce the smaller position by 50%.

**Beta neutrality check:**
At the portfolio level, verify that the net market beta is near zero:

```
Portfolio beta = Σ_i (w_i × β_stock_i) − Σ_j (w_j × β_ETF_j)
```

If |Portfolio beta| > 0.3, your hedging is imperfect — either the rolling beta is stale or an
ETF is an imperfect proxy. Adjust ETF hedge quantities.

**Maximum drawdown circuit breaker:**
If the strategy's total P&L drawdown exceeds 15% of allocated capital in any rolling 30-day window,
halt new entries for 10 trading days. This is a forced pause to assess whether a regime change
has occurred. Resume only after re-running the OU diagnostics on all stocks and confirming that
residuals are still stationary. Do not override this rule.

---

## 9. Data Sources and Infrastructure

**Price data:**
- NSE Bhavcopy (EOD): free, direct download from nseindia.com. Available by 7pm each trading day.
- Python: `jugaad-data` or `nsepy` libraries for programmatic download.
- Sector index data: NSE publishes index EOD values alongside Bhavcopy.

**Earnings calendar:**
- NSE website publishes board meeting dates (which usually coincide with earnings releases).
- `jugaad-data` has an earnings calendar API.
- Cross-check with BSE corporate filings for exact dates.

**ETF NAV:**
- NSE publishes intraday and EOD iNAV for all ETFs. Use EOD NAV for sizing calculations.
- BANKBEES, ITBEES: highly liquid, bid-ask spread < 0.05%. Acceptable for EOD execution.

**Execution:**
- Enter and exit at next-day market open using limit orders placed at open ± 0.1%.
  Do not use market orders — the ETF and futures open can gap significantly.
- For the stock leg: use equity cash market or single-stock futures. Cash market is simpler
  but requires capital upfront. Futures require margin but free up cash for other positions.
- Zerodha Kite API or Upstox API for order placement. Both have Python SDKs.

---

## 10. What Can Go Wrong — Honest Assessment

**Beta instability:** If a stock's sector beta changes rapidly (e.g., ICICI Bank acquiring a
large NBFC), your rolling beta will lag behind the structural change. The residual will appear
to deviate when it is actually the beta that has shifted. Mitigation: monitor rolling beta for
sudden changes (> 0.2 shift in 10 days) and pause trading that stock.

**Sector ETF tracking error:** BANKBEES has a tracking error of ~0.10% per day. Over a 15-day
hold, this accumulates to ~1.5% unexplained P&L variation. This is not large enough to destroy
the strategy but is large enough to distort your z-score calculations. Use the ETF's published
iNAV, not its market price, for the sector regression — the iNAV is closer to the true index value.

**Crowding in large-cap stocks:** HDFCBANK and ICICIBANK are the most-traded stocks on NSE.
Many prop desks run residual mean-reversion on them. When z-scores are extreme, you may be on
the same side as many other players — entry is fine, but exit can be difficult if everyone
exits simultaneously. Prefer smaller stocks within each sector where crowding is lower, even
if R² is slightly lower.

**Illiquid ETF legs:** PSU Bank ETFs are less liquid than BANKBEES. Bid-ask spreads can widen
to 0.2–0.3% on low-volume days. This adds ~0.4–0.6% round-trip cost to each trade. Factor this
into your threshold calibration — for PSU banking positions, set z_entry 0.2σ higher than for
private banking to compensate.

---

---

# STRATEGY B
# Fundamental Value Spread (Accounting-Anchored OU)

---

## 1. Core Intuition

Your HAL/BDL spread is constructed from log-prices and tested for statistical stationarity.
The spread mean-reverts because the two companies share fundamental drivers — but you are
*inferring* this from price history, not from the fundamentals themselves.

This strategy builds the spread *directly* from financial statement data — specifically from
valuation ratios — so the mean has an economic justification, not just a statistical one.

The fundamental premise: two companies in the same sector with similar business models, similar
return-on-equity profiles, and similar growth rates should trade at similar valuation multiples.
If Company A's P/B ratio is 2.5× and Company B's is 1.8×, and this gap is historically 0.2×,
then the current gap of 0.7× is either:
(a) Justified by a real fundamental divergence (ROE has diverged, growth has diverged), or
(b) A pricing inefficiency that will correct as the market re-rates the cheaper company upward
    or the expensive company downward.

You construct the spread from this valuation ratio gap, model it as an OU process, and trade
the deviation. Critically, the OU equilibrium mean μ is not estimated statistically — it is
*computed* from fundamental data. This makes the mean more stable across regimes than a
purely statistical estimate.

---

## 2. Choosing the Right Valuation Metric

Not all valuation ratios are equally useful for OU modeling. The criteria:

1. **Stationarity:** The ratio spread must be stationary. P/E ratios are notoriously non-stationary
   (they swing wildly with earnings cycles). P/B is more stable because book value changes slowly.
2. **Availability:** Must be computable from publicly available quarterly data. No Bloomberg needed.
3. **Comparability:** Both companies must use the same accounting standard (both Indian GAAP or
   both Ind-AS — not mixed) and have similar capital structures (ratio must not be distorted by
   one company's debt level).

**Recommended primary metric: Price-to-Book (P/B)**

```
PB_i(t) = Price_i(t) / Book_Value_Per_Share_i(latest quarter)
```

P/B is ideal because:
- Book value is audited quarterly and changes slowly (typically < 5% per quarter for stable businesses)
- P/B spread between comparable companies is highly stationary historically
- It is available for free from screener.in, BSE corporate filings, and NSE financial data

**Recommended secondary metric: EV/EBITDA**

Use EV/EBITDA for capital-intensive sectors (steel, cement, infrastructure) where book value
does not capture asset value well. EV/EBITDA requires computing enterprise value (market cap
+ net debt) and trailing 12-month EBITDA. Both are available from quarterly balance sheets.

**Metric to avoid: P/E**

P/E is non-stationary across cycles. A PSU bank's P/E swings from 5× to 40× across credit
cycles, and the spread between two PSU banks' P/E ratios is not stationary either. Do not use P/E
as the primary spread metric for OU modeling.

---

## 3. Pair Selection — Fundamental Similarity Criteria

Unlike Strategy A (which uses any liquid stock in a sector), Strategy B requires pairs where
the fundamental similarity justification is *strong*. You must be able to articulate *why* the
valuation spread should mean-revert, not just observe that it has historically.

**Step 1 — Same sector, same sub-segment.**

PSU banks vs private banks are different businesses despite being in the same sector. The P/B
spread between SBI and HDFC Bank does not mean-revert because their ROE structures are
permanently different (HDFC Bank consistently earns 16–18% ROE; SBI earns 10–13%). Only pair
within sub-segments:
- SBI ↔ Bank of Baroda (both large PSU banks with similar government ownership and NPA cycles)
- HDFC Bank ↔ ICICI Bank (both large private banks competing for the same customer segment)
- TCS ↔ Infosys (both Tier-1 IT exporters with similar revenue mix and margin profile)
- Sun Pharma ↔ Dr. Reddy's (both large pharma with branded generic + API businesses)

**Step 2 — ROE proximity test.**

The primary reason two companies should trade at similar P/B ratios is similar Return on Equity.
The Gordon Growth relationship links P/B to ROE:

```
P/B = (ROE − g) / (ke − g)
```

where g is the sustainable growth rate and ke is the cost of equity. If ROE_A ≈ ROE_B and both
companies have similar risk (similar ke), then P/B_A ≈ P/B_B.

Before including a pair, verify:
- 5-year average ROE_A and ROE_B differ by less than 4 percentage points
- ROE correlation between the two companies > 0.60 over the past 5 years (they respond to the
  same ROE drivers — interest rates, credit cycles, demand cycles)

This ROE test is your cointegration justification. It replaces the Johansen/ADF test as the
first-pass screen. You can then run ADF on the P/B spread as a secondary confirmation.

**Step 3 — Capital structure similarity.**

If Company A has 50% debt/equity and Company B has 10% debt/equity, their P/B ratios will
diverge for capital structure reasons unrelated to fundamental value. Apply a leverage filter:
- For banks: Tier 1 Capital Ratio should be within 2 percentage points
- For non-banks: Net Debt/Equity should be within 0.5× of each other

**Recommended pairs to start with:**

| Pair | Sector | Spread metric | Justification |
|---|---|---|---|
| SBI / Bank of Baroda | PSU Banking | P/B | Same government ownership, NPA cycles correlated |
| HDFC Bank / ICICI Bank | Private Banking | P/B | Same customer segment, similar ROE |
| TCS / Infosys | IT Services | P/B or EV/EBITDA | Tier-1 IT duopoly, correlated USD revenue |
| ONGC / Oil India | Upstream Oil | EV/EBITDA | Both government-owned upstream E&P |
| Coal India / NMDC | Mining PSU | EV/EBITDA | Both commodity PSUs under same ministry |
| HAL / BDL | Defence PSU | P/B | You already know this pair — extend it |

Start with one pair (SBI/Bank of Baroda) before running the full universe.

---

## 4. Spread Construction

**Step 1 — Compute daily P/B ratio for each company.**

```python
import pandas as pd
import numpy as np

def compute_pb_ratio(prices: pd.Series,
                     book_values: pd.Series) -> pd.Series:
    """
    prices: daily EOD closing price
    book_values: quarterly book value per share (forward-filled to daily)
    Returns: daily P/B ratio series
    """
    # Forward-fill quarterly book value to daily frequency
    bvps_daily = book_values.reindex(prices.index, method='ffill')
    pb = prices / bvps_daily
    return pb
```

**Why forward-fill?** Book value is announced quarterly with a lag (typically 30–45 days after
quarter end). You use the most recently announced book value, forward-filled until the next
announcement. This is the only information available in real time — no lookahead.

**Step 2 — Compute the P/B spread.**

```python
def pb_spread(pb_A: pd.Series, pb_B: pd.Series) -> pd.Series:
    """
    Spread = PB_A - PB_B
    Note: this is an arithmetic spread, not log-ratio.
    Log-ratio (log PB_A / PB_B) is also valid and sometimes more stationary.
    Test both empirically on your specific pair.
    """
    return pb_A - pb_B
```

**Log-ratio vs arithmetic spread:** For P/B ratios close to each other (both in the range 1–3×),
the arithmetic spread and log-ratio give similar results. For pairs with very different P/B levels
(e.g., one at 0.5× and one at 3×), use the log-ratio — it handles multiplicative scaling better
and is more likely to be stationary.

**Step 3 — Compute the fundamental fair-value mean.**

This is the key difference from purely statistical stat arb. Instead of estimating μ purely from
historical spread data, you compute a **fundamental implied mean** from the ROE relationship:

```python
def fundamental_pb_spread(roe_A: float, roe_B: float,
                           ke: float = 0.12,
                           g: float = 0.07) -> float:
    """
    Theoretical P/B from Gordon Growth: PB = (ROE - g) / (ke - g)
    Spread of theoretical P/B values is the fundamental anchor for mu.

    roe_A, roe_B: trailing 12-month ROE (as decimal, e.g., 0.16 for 16%)
    ke: cost of equity (use CAPM estimate or sector average, ~12% for Indian banks)
    g: sustainable growth rate (~7% for large Indian banks, in line with nominal GDP)
    """
    pb_theoretical_A = (roe_A - g) / (ke - g)
    pb_theoretical_B = (roe_B - g) / (ke - g)
    return pb_theoretical_A - pb_theoretical_B
```

In practice:
- Use trailing 12-month ROE updated quarterly (when new results are announced)
- The ke estimate (cost of equity) is the most uncertain input. Use 12% as a baseline for
  Indian large-caps and update annually using the CAPM: ke = rf + β × ERP, where rf ≈ 7%
  (10-year G-Sec yield) and ERP ≈ 5% for India.
- g is the long-run nominal growth rate sustainable by the company. For large PSU banks, use
  nominal GDP growth (~12% nominal, i.e., 7% real + 5% inflation). For IT companies, use
  USD revenue growth rate (~8–10%).

The fundamental mean μ_fundamental is a slowly moving target — it updates quarterly. Between
quarterly updates, you use the previous quarter's value. The statistical mean μ_statistical
from OU estimation serves as a secondary reference.

**Combined mean estimate:**

```python
def combined_mu(mu_fundamental: float,
                mu_statistical: float,
                w_fundamental: float = 0.6) -> float:
    """
    Weighted combination of fundamental and statistical means.
    w_fundamental = 0.6 gives primary weight to the theory-based mean.
    """
    return w_fundamental * mu_fundamental + (1 - w_fundamental) * mu_statistical
```

The 0.6/0.4 weighting is a starting point — you can Bayesian-optimize this weight as one of
your parameters, with a prior centered at 0.6 (reflecting your theoretical confidence in the
fundamental anchor).

---

## 5. OU Estimation on the P/B Spread

Apply your existing `fit_ou()` function to the P/B spread, with one modification: demean the
spread by the fundamental mean before fitting, so that the OU process is centered at zero rather
than at the raw spread level.

```python
def fit_ou_fundamental(spread: np.ndarray,
                       mu_fundamental: float,
                       dt: float = 1.0/252) -> dict:
    """
    Fit OU process to (spread - mu_fundamental).
    dt = 1/252 because spread is a valuation ratio, not a log-price.
    """
    demeaned = spread - mu_fundamental
    # Use daily dt (1/252 years) since P/B is a continuous ratio
    return fit_ou(demeaned, dt=dt)
```

Note the `dt = 1/252` argument. Your existing `fit_ou()` likely uses `dt = 1` (one day in
trading-day units). For a P/B spread, the OU parameters have different natural units — κ should
be interpreted in years^-1 rather than days^-1. Adjust your half-life output accordingly:

```
half_life_days = ln(2) / (kappa × 252)    [if kappa is in years^-1]
half_life_days = ln(2) / kappa             [if kappa is in days^-1]
```

Be consistent in your units. The numbers are the same; it is just a matter of what κ means.

---

## 6. Trade Timing — The Quarterly Update Cycle

This strategy has a natural event calendar that Strategy A and your HAL/BDL model do not have:
quarterly earnings releases. These create the two most important moments in the strategy:

**Moment 1 — Post-earnings fundamental mean update.**

When Company A announces quarterly results:
1. Download updated book value per share from the filing
2. Compute updated trailing 12-month ROE
3. Recompute μ_fundamental with the new ROE figures
4. Recompute the OU z-score with the updated mean

If the fundamental mean shifts significantly (> 0.2× in P/B terms), the old z-score may
be misleading. Example: SBI announces better-than-expected NPA recovery, lifting its ROE
from 12% to 14%. This justifies a higher P/B for SBI. The P/B spread that looked like an
extreme deviation (SBI trading rich relative to BoB) may now be at fair value given the
improved ROE. Do not enter a short-SBI trade until you have updated the fundamental mean.

**Moment 2 — Earnings blackout.**

Do NOT enter new positions in the 10 days before either company's results announcement and
2 days after. Earnings releases create P/B spikes that are not mean-reverting noise — they
are information arrivals. Your model cannot distinguish an earnings-driven P/B gap from a
noise-driven P/B gap in the days around results.

```python
def get_earnings_blackout_dates(earnings_dates: list,
                                pre: int = 10,
                                post: int = 2) -> set:
    blackout = set()
    for ed in earnings_dates:
        for d in range(-post, pre + 1):
            blackout.add(ed + pd.Timedelta(days=d))
    return blackout
```

This blackout means you will have approximately 40–50 blackout days per year per pair
(4 quarters × 2 companies × ~6 days each). This is a significant constraint but a necessary one.

---

## 7. Signal Generation

**Entry conditions (all must be satisfied):**

1. Not in earnings blackout window for either company
2. Latest quarterly book values are available (not stale — if BoB hasn't reported in 50+ days,
   your book value is likely outdated)
3. ADF p-value on the 90-day P/B spread < 0.15 (spread is stationary in current window)
4. |z_score| > z_entry_threshold (spread has deviated sufficiently)
5. Half-life < 40 days (reversion expected within reasonable horizon)
6. Fundamental mean and statistical mean agree on direction (both suggest the same company
   is cheaper on a relative basis). If they disagree, skip the trade.

**Condition 6 is the single most important filter in this strategy.** It is your protection
against entering a trade where the statistical signal says "this is cheap" but the fundamental
analysis says "this company has permanently re-rated lower." When the two means disagree in
direction, one of them is wrong — and you cannot be sure which. Wait for agreement.

**Entry direction:**
- Spread > combined_mu + z_entry × σ_stationary: Company A is overvalued relative to B on P/B.
  **Short A, Long B** (not necessarily in equal notional — see sizing below).
- Spread < combined_mu − z_entry × σ_stationary: Company A is undervalued relative to B.
  **Long A, Short B.**

---

## 8. Hedge Ratio and Position Sizing

Unlike log-price stat arb where the hedge ratio is estimated via OLS (your β = 0.8303), the
hedge ratio in this strategy has a direct fundamental interpretation.

**Hedge ratio from dollar-neutral construction:**

The simplest approach: make the trade dollar-neutral. If you go long ₹1 lakh notional of Company
A, go short ₹1 lakh notional of Company B. The hedge ratio in units is:

```
units_A / units_B = (price_B / price_A) × (lot_size_B / lot_size_A)
```

This is not the same as your current β-hedge (which adjusts for differential vol). The
dollar-neutral hedge is more appropriate here because you are trading a *valuation multiple*,
not a *log-price spread* — you want equal capital exposure on both sides, not equal volatility.

**Vol-adjusted alternative:**

If the two companies have significantly different volatilities (σ_A / σ_B > 1.5), use
vol-adjusted sizing:

```
notional_A / notional_B = σ_B / σ_A
```

This equalizes the volatility contribution from each leg, which is theoretically cleaner for
a strategy where you are uncertain about which leg will move.

**Capital per pair:**

With small capital (₹2–3 lakh total), run one pair at a time. Allocate 40% of capital to the
long leg and 40% to the short leg. Keep 20% as a buffer. At lot-level granularity, you will
often be constrained to 1 lot on each side — accept this; do not over-leverage to achieve
precise sizing.

---

## 9. Parameter Optimization

**Parameters specific to this strategy (beyond the shared OU parameters):**

| Parameter | Search range | Notes |
|---|---|---|
| z_entry | [1.0, 2.5] | P/B spreads are less volatile than log-price spreads; lower thresholds work |
| z_exit | [0.1, 0.4] | |
| z_stop | [2.5, 4.0] | |
| w_fundamental | [0.3, 0.8] | Weight on fundamental mean vs statistical mean |
| ke (cost of equity) | [0.10, 0.15] | Bayesian prior: N(0.12, 0.01) |
| g (growth rate) | [0.05, 0.10] | Bayesian prior: N(0.07, 0.01) |
| OU_window | [60, 180] | Window for OU parameter estimation |
| earnings_blackout_pre | [5, 15] | Days before earnings to stop entering |

**Critical point on ke and g optimization:** Do not optimize ke and g primarily to maximize
Sharpe. These are economic parameters with real-world priors. Use Bayesian optimization with
strong priors: ke ~ N(0.12, 0.01) and g ~ N(0.07, 0.01). If the optimizer wants ke = 0.08
to maximize Sharpe, reject that — the prior says the cost of equity for an Indian bank is
not 8% in the current rate environment. Constrain the optimizer to physically meaningful ranges.

**Walk-forward setup:**

```
Training fold   : 24 months  [longer than Strategy A because quarterly data is sparse]
Test fold       : 6 months OOS
Step size       : 6 months
Minimum folds   : 4 (requires 36 months of data)
```

The longer training fold is necessary because you have far fewer trades here (~8–15 per year
per pair) than in Strategy A. You need enough training trades to estimate parameters reliably.
4 OOS folds on 4.5 years of data is the minimum for meaningful walk-forward inference.

**Expect fewer trades and accept it.** 8–15 trades per year per pair is not a problem —
it keeps transaction costs low and each trade has a genuine fundamental underpinning. Quality
over quantity is more important here than in Strategy A.

---

## 10. Robustness Checks Specific to This Strategy

**Check 1 — ROE stability across the backtest period.**

Before running the full backtest, plot the 5-year ROE of both companies. If ROE of one company
underwent a structural shift during the backtest (e.g., SBI's ROE collapsed from 15% to 5%
during the NPA crisis of 2015–2018 and then recovered), your fundamental mean is non-stationary
across the training period. In this case, restrict your training data to the post-stabilization
period (2019 onwards for PSU banks) even if it gives you fewer folds.

**Check 2 — Book value discontinuity from accounting changes.**

India transitioned from Indian GAAP to Ind-AS in 2016–2018 for most listed companies. Book
values under Ind-AS are not comparable to book values under Indian GAAP (fair value adjustments
and expected credit loss provisioning change the numbers significantly). Your training data
must use a consistent accounting standard. Either:
- Start the backtest from April 2018 (when Ind-AS adoption was complete for most large banks), or
- Normalize historical book values using the restated figures published by companies in their
  transition-year annual reports.

The easiest path: start from April 2019 (a full year post-transition for clean data).

**Check 3 — Spread stationarity across regimes.**

Run the ADF test on the P/B spread separately for three sub-periods:
- Pre-COVID (up to Dec 2019)
- COVID period (Jan 2020 – Mar 2021)
- Post-COVID (Apr 2021 onwards)

If the spread is non-stationary in one sub-period but stationary in others, your model will
have regime-dependent performance. Note this in your monitoring — add a regime filter that
reduces position size when rolling ADF p-value exceeds 0.15.

---

## 11. Live Monitoring Protocol

Once deployed (even in paper trading), monitor these quantities daily:

**Daily checks (automated, 5 minutes):**
```
1. Current P/B spread vs combined_mu: how many σ away?
2. Rolling ADF p-value (90-day): is the spread still stationary?
3. Rolling OU half-life: has it drifted above 40 days?
4. Open position z-score: has any open trade hit the stop threshold?
5. Earnings blackout status: any upcoming results in next 10 days?
```

**Weekly checks (manual, 30 minutes):**
```
1. Has either company released new quarterly results? → Update book value, recompute mu_fundamental
2. Has the ROE gap changed meaningfully (> 1 pp)? → Reconsider fundamental mean weight
3. Any regulatory announcements, M&A, management changes? → Evaluate if spread mean has shifted
4. Portfolio beta: is the net market exposure near zero?
```

**Quarterly review (manual, half day):**
```
1. Full re-run of walk-forward validation on expanded dataset
2. Check if Bayesian-optimized parameters have shifted significantly → investigate why
3. Review all stopped-out trades: were they regime changes (accept) or parameter failures (investigate)?
4. Check whether pair's fundamental similarity still holds (ROE proximity, capital structure)
5. Consider adding one new pair or dropping one pair that has lost fundamental similarity
```

---

## 12. Honest Assessment — What Can Go Wrong

**Permanent re-rating risk (most dangerous):** If the P/B spread widens because one company
has genuinely and permanently re-rated — not because of noise, but because the market has
correctly identified a structural deterioration — your model will repeatedly enter against
the trend and stop out repeatedly. This is the fundamental risk of *any* mean-reversion strategy,
but it is more acute here because valuation re-ratings can last years, not days. The protection
is Condition 6 (fundamental and statistical means must agree). If ROE divergence explains the
P/B gap, do not trade.

**Data lag risk:** You are working with quarterly book values that arrive 30–45 days after
quarter end. During this lag, the price is moving but your P/B denominator is stale. The P/B
ratio is artificially volatile in the 6 weeks before a results announcement. Widen your
z_entry threshold by 0.3σ in the 30 days before results to compensate for this stale-data noise.

**Lot size constraint:** Nifty 50 stock futures have lot values of ₹3–8 lakh. With ₹2 lakh
capital, you cannot run this strategy in futures — you need the margin to be < 40% of capital
per position. Two options: (a) use the cash segment for the long leg and single-stock futures
for the short leg, or (b) wait until capital grows to ₹5+ lakh before running this strategy.
Capital constraint is real — do not lever up to force lot sizes.

---

---

# Comparison and Sequencing

## Side-by-Side

| Dimension | HAL/BDL (existing) | Strategy A: Sector Residual | Strategy B: Fundamental P/B |
|---|---|---|---|
| **Spread construction** | Log-price ratio, OLS β | Regression residual vs sector index | Valuation ratio gap (P/B or EV/EBITDA) |
| **Mean anchor** | Statistical (OU μ from history) | Statistical (zero by construction) | Fundamental (Gordon Growth model) |
| **Universe** | Fixed pair | Rotating N stocks, M sectors | Fixed pairs, chosen by ROE proximity |
| **Trades per year** | ~11 | ~20–30 (across universe) | ~8–15 per pair |
| **Avg hold** | 7–10 days | 5–15 days | 10–25 days |
| **Cointegration type** | Price cointegration | Residual stationarity | Valuation ratio stationarity |
| **New infrastructure** | — | Sector ETF trading, rolling OLS | Quarterly data pipeline, fundamental model |
| **Capital min.** | ₹2 lakh | ₹2–3 lakh | ₹4–5 lakh (lot size constraints) |
| **Regime robustness** | Medium (depends on pair staying linked) | Higher (sector hedge reduces factor exposure) | Higher (fundamental anchor slows drift) |
| **Data sources** | NSE Bhavcopy | NSE Bhavcopy + ETF NAV | NSE Bhavcopy + screener.in quarterly |
| **Biggest risk** | Pair decoupling | Beta instability, ETF tracking error | Permanent re-rating, data lag |

## Build Sequence

```
Now          : HAL/BDL in paper trading → real money at small scale
Month 2–3    : Build Strategy A infrastructure (rolling OLS, ETF data pipeline)
               Paper trade Strategy A on PSU Banking sector (3–4 stocks)
Month 4–5    : Strategy A live at small scale (1–2 positions at a time)
               Begin quarterly data collection for Strategy B (screener.in, BSE filings)
Month 6–8    : Build Strategy B on SBI/Bank of Baroda P/B spread
               Paper trade Strategy B through at least one quarterly earnings cycle
Month 9–12   : Strategy B live at small scale IF capital has grown to ₹5+ lakh
               All three strategies running simultaneously with independent capital allocations
```

The sequencing ensures each strategy is independently validated before live deployment.
Strategy A builds naturally on your existing OU framework. Strategy B requires the additional
quarterly data pipeline, which takes time to build and verify, and higher minimum capital due
to lot size constraints.

The long-term goal: three independently-motivated mean-reversion edges with low correlation to
each other. HAL/BDL is sector-pair. Strategy A is sector-residual. Strategy B is fundamental-ratio.
Each fails in different market regimes, so the combined portfolio is more robust than any single
strategy.

---

*Written: May 2026*
