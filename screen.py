import numpy as np
import pandas as pd
import yfinance as yf
import itertools
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats as scipy_stats


TICKERS = [

    # ── DEFENCE & HEAVY ENGINEERING ──
    "BEL.NS",
    "BDL.NS",
    "HAL.NS",
    "BHEL.NS",
    "BEML.NS",

]

START          = '2012-01-01'
END            = '2026-03-19'
MIN_BARS       = 1000          # minimum clean bars required
MAX_HALF_LIFE  = 45            # days — hard reject above this
MIN_HALF_LIFE  = 3             # days — too fast, likely noise
ADF_THRESHOLD  = 0.05          # p-value
HURST_MAX      = 0.48          # below 0.5 = mean-reverting


# ─────────────────────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────────────────────
def load_prices(tickers, start, end):
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)['Close']

    # Handle single ticker edge case
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    # Clean
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw[raw > 0]          # drop zero/negative prices
    raw = raw.dropna(how='all')
    raw = raw.ffill().bfill()   # fill isolated NaNs from holidays
    raw = raw.dropna()

    print(f"  {len(raw)} clean bars  "
          f"({raw.index[0].date()} → {raw.index[-1].date()})")
    return raw


# ─────────────────────────────────────────────────────────────
#  STAT FUNCTIONS
# ─────────────────────────────────────────────────────────────
def estimate_beta(p1, p2):
    """OLS hedge ratio: log(T1) ~ α + β·log(T2)"""
    lp1 = np.log(p1)
    lp2 = np.log(p2)
    mask = np.isfinite(lp1) & np.isfinite(lp2)
    lp1, lp2 = lp1[mask], lp2[mask]
    res = OLS(lp1, add_constant(lp2)).fit()
    return float(res.params.iloc[1])


def compute_spread(p1, p2, beta):
    """β-adjusted log spread"""
    s = np.log(p1) - beta * np.log(p2)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def adf_test(spread):
    stat, p, _, _, crit, _ = adfuller(spread, autolag='AIC')
    return p, stat, crit


def hurst_exponent(spread):
    lags = range(2, min(60, len(spread) // 4))
    tau  = [np.std(np.subtract(spread.values[l:],
                               spread.values[:-l])) for l in lags]
    h, *_ = scipy_stats.linregress(np.log(list(lags)), np.log(tau))
    return h


def half_life(spread):
    combined = pd.concat([spread, spread.shift(1)], axis=1).dropna()
    combined.columns = ['s', 's_lag']
    res  = OLS(combined['s'], add_constant(combined['s_lag'])).fit()
    phi  = min(float(res.params.iloc[1]), 1 - 1e-8)
    kappa = -np.log(max(phi, 1e-8)) * 252
    hl   = np.log(2) / kappa * 252
    return hl


def johansen_test(df_pair):
    try:
        joh      = coint_johansen(df_pair, det_order=0, k_ar_diff=1)
        passed   = bool(joh.lr1[0] > joh.cvt[0, 1])
        trace    = float(joh.lr1[0])
        crit     = float(joh.cvt[0, 1])
        return passed, trace, crit
    except Exception:
        return False, np.nan, np.nan


def zscore_stability(spread, window=60):
    """
    Rolling z-score mean and std stability.
    A stable pair should have rolling mean close to 0
    and rolling std close to 1 consistently.
    Returns: std of rolling mean (lower = more stable equilibrium)
    """
    roll_mean = spread.rolling(window).mean().dropna()
    roll_std  = spread.rolling(window).std().dropna()
    mean_stability = roll_mean.std()    # low = equilibrium is stable
    std_stability  = roll_std.std()     # low = vol regime is stable
    return mean_stability, std_stability


def correlation(p1, p2):
    return np.corrcoef(p1, p2)[0,1]


# ─────────────────────────────────────────────────────────────
#  SCREEN A SINGLE PAIR
# ─────────────────────────────────────────────────────────────
def screen_pair(t1, t2, prices):
    if t1 not in prices.columns or t2 not in prices.columns:
        return None

    p1 = prices[t1].dropna()
    p2 = prices[t2].dropna()

    # Align
    combined = pd.concat([p1, p2], axis=1).dropna()
    if len(combined) < MIN_BARS:
        return None
    p1, p2 = combined.iloc[:, 0], combined.iloc[:, 1]

    try:
        beta    = estimate_beta(p1, p2)
        spread  = compute_spread(p1, p2, beta)

        adf_p, adf_stat, _  = adf_test(spread)
        hurst               = hurst_exponent(spread)
        hl                  = half_life(spread)
        joh_pass, joh_tr, joh_cr = johansen_test(
            pd.concat([np.log(p1), np.log(p2)], axis=1))
        mean_stab, std_stab = zscore_stability(spread)
        corr                = correlation(p1, p2)

        # ── Scoring ──────────────────────────────────────
        # Each criterion scores 0 or 1, weighted by importance
        # Primary:   ADF, half-life range, Johansen
        # Secondary: Hurst, z-score stability, correlation
        score = 0
        score += 3 if adf_p < ADF_THRESHOLD else 0
        score += 3 if MIN_HALF_LIFE < hl < MAX_HALF_LIFE else (
                 1 if hl < MAX_HALF_LIFE * 2 else 0)
        score += 2 if joh_pass else 0
        score += 1 if hurst < HURST_MAX else 0
        score += 1 if mean_stab < 0.5 else 0   # stable equilibrium

        # Hard disqualifiers
        viable = (hl > MIN_HALF_LIFE and
                  hl < MAX_HALF_LIFE * 3 and   # not hopelessly slow
                  adf_p < 0.15 and             # at least marginal stationarity
                  hurst < 0.5)                 # must be mean-reverting

        return dict(
            t1=t1, t2=t2,
            beta=round(beta, 4),
            n_bars=len(combined),
            adf_p=round(adf_p, 4),
            hurst=round(hurst, 4),
            half_life=round(hl, 1),
            johansen=joh_pass,
            joh_trace=round(joh_tr, 2),
            mean_stability=round(mean_stab, 4),
            std_stability=round(std_stab, 4),
            correlation=round(corr, 3),
            score=score,
            viable=viable,
        )

    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────
#  PRINT RESULTS
# ─────────────────────────────────────────────────────────────
def print_results(results):
    if not results:
        print("\n  No viable pairs found.")
        return

    # Split into viable and marginal
    viable   = [r for r in results if r['viable']]
    marginal = [r for r in results if not r['viable']]

    def _header():
        print(f"\n  {'Pair':<35}  {'β':>6}  {'ADF_p':>6}  "
              f"{'Hurst':>6}  {'HL(d)':>6}  {'Joh':>4}  "
              f"{'Corr':>5}  {'Score':>5}  {'Verdict'}")
        print("  " + "─" * 100)

    def _row(r, tag=''):
        joh_str  = '✓' if r['johansen'] else '✗'
        adf_str  = f"{r['adf_p']:.4f}"
        hl_str   = f"{r['half_life']:.1f}"
        pair_str = f"{r['t1'].replace('.NS','')} / {r['t2'].replace('.NS','')}"
        flags    = []
        if r['adf_p'] < ADF_THRESHOLD:    flags.append('ADF✓')
        if r['hurst'] < HURST_MAX:        flags.append('H✓')
        if r['half_life'] < MAX_HALF_LIFE: flags.append('HL✓')
        if r['johansen']:                  flags.append('Joh✓')
        verdict = ' '.join(flags) if flags else '—'

        print(f"  {pair_str:<35}  {r['beta']:>6.3f}  {adf_str:>6}  "
              f"{r['hurst']:>6.3f}  {hl_str:>6}  {joh_str:>4}  "
              f"{r['correlation']:>5.3f}  {r['score']:>5}  {verdict}")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║  PAIRWISE STAT SCREEN
║  Criteria: ADF p<{ADF_THRESHOLD}  |  Hurst<{HURST_MAX}  |  HL {MIN_HALF_LIFE}–{MAX_HALF_LIFE}d  |  Johansen
║  Score: ADF=3pts  HL=3pts  Johansen=2pts  Hurst=1pt  Stability=1pt  (max 10)
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝""")

    if viable:
        print(f"\n  ── VIABLE PAIRS ({len(viable)}) ── score ≥ 5, all soft criteria met\n")
        _header()
        for r in sorted(viable, key=lambda x: -x['score']):
            _row(r)

    if marginal:
        print(f"\n  ── MARGINAL / FAILING PAIRS ({len(marginal)}) ──\n")
        _header()
        for r in sorted(marginal, key=lambda x: -x['score']):
            _row(r)

    # Top recommendation
    if viable:
        best = sorted(viable, key=lambda x: -x['score'])[0]
        print(f"""
  ── TOP RECOMMENDATION ──
  Pair       : {best['t1']} / {best['t2']}
  Beta       : {best['beta']}
  Half-life  : {best['half_life']}d
  ADF p      : {best['adf_p']}  {'✓' if best['adf_p'] < ADF_THRESHOLD else '✗'}
  Hurst      : {best['hurst']}  {'✓' if best['hurst'] < HURST_MAX else '✗'}
  Johansen   : {'✓ pass' if best['johansen'] else '✗ fail'}
  Score      : {best['score']} / 10
  → Run full Bayesian optimisation on this pair first.""")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    prices  = load_prices(TICKERS, START, END)
    pairs   = list(itertools.combinations(TICKERS, 2))
    results = []

    print(f"\nScreening {len(pairs)} pairs...")
    i = 0
    for t1, t2 in pairs:
        i+=1
        if i%50==0:
            print(f"done {i} pairs")
        r = screen_pair(t1, t2, prices)
        if r is not None:
            results.append(r)

    print_results(results)

    # ── Save full results to CSV for inspection ──
    if results:
        df_out = pd.DataFrame(results).sort_values('score', ascending=False)
        df_out.to_csv('pair_screen_results.csv', index=False)
        print(f"\n  Full results saved → pair_screen_results.csv")
