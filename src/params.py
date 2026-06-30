PARAMS = dict(

    T1 = 'BAJFINANCE.NS',
    T2 = 'KOTAKBANK.NS',

    start = '2015-01-01',
    # ── Windows ──
    slow_window    = 23,
    medium_window  = 13,
    fast_span      = 10,
    vol_window     = 20,

    # ── entry thresholds (symmetric — long/short kept for backtest.py
    #    / features.py compatibility, but always equal) ──
    z_entry_long   = 2.4,
    z_entry_short  = 2.4,

    # ── exits ──
    z_exit_long    = 0.18,
    z_exit_short   = 0.18,

    # ── stops ──
    z_stop_long    = 4.5,
    z_stop_short   = 4.5,

    vol_cap        = 1.5,
    max_hold       = 25,

    # ── Kept for compatibility ──
    z_entry        = 1.1,
    z_exit         = 0.18,
    z_stop         = 2.9,

    # ── Autocorrelation regime filter ──
    autocorr_window    = 60,
    autocorr_threshold = 0.1,
    ou_adapt_span      = 252,

    # ── Falling-knife guard (long entries only) ──

    fk_enabled          = True,
    fk_rsi_slope_max    = -8.410,
    fk_lr_mom_slope_max = -0.029,
    fk_z_fast_slope_max = -0.784,
    fk_votes_min        = 2,

    # ── Exhaustion guard (short entries only) ──

    se_enabled          = True,
    se_z_fast_slope_min = 0.840,
    se_z_fast_accel_min = 0.670,
    se_rsi_slope_min    = 9.277,
    se_votes_min        = 2,
)
