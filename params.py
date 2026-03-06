# Core parameters — tuned for 14.5d half-life
PARAMS = dict(
    # ── Windows ──
    slow_window    = 20,
    medium_window  = 30,
    fast_span      = 10,
    vol_window     = 20,

    # ── Asymmetric entry thresholds ──
    # Long spread  = Brent premium collapsed (historically rarer, faster)
    # Short spread = Brent premium extended  (more common, slower to revert)
    z_entry_long   = 1.5,    # tighter — these moves are sharp, catch early
    z_entry_short  = 1.5,    # wider  — premium extensions linger longer

    # ── Asymmetric exits ──
    z_exit_long    = 0.2,    # exit long faster — mean reversion can overshoot
    z_exit_short   = 0.4,    # exit short slower — premium tends to compress gradually

    # ── Asymmetric stops ──
    z_stop_long    = 3.5,    # wider stop long — dislocation events are violent
    z_stop_short   = 2.8,    # tighter stop short — extended premiums rarely blow out

    # ── Pyramid ──
    z_add          = 2.0,
    vol_cap        = 1.5,
    max_hold       = 25,

    # ── Kept for compatibility ──
    z_entry        = 1.25,   # fallback if features/signals not yet updated
    z_exit         = 0.30,
    z_stop         = 3.0,

    # ── Autocorrelation regime filter ──
    autocorr_window    = 20,
    autocorr_threshold = 0.1,
    ou_adapt_span      = 252,
)