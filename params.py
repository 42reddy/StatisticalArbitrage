PARAMS = dict(

    T1 = 'HAL.NS',
    T2 = 'BDL.NS',
    # ── Windows ──
    slow_window    = 20,
    medium_window  = 30,
    fast_span      = 10,
    vol_window     = 20,

    # ── entry thresholds ──
    z_entry_long   = 1.5,
    z_entry_short  = 1.5,

    # ── exits ──
    z_exit_long    = 0.2,
    z_exit_short   = 0.4,

    # ── Asymmetric stops ──
    z_stop_long    = 3.5,
    z_stop_short   = 2.8,

    # ── Pyramid ──
    z_add          = 2.0,
    vol_cap        = 1.5,
    max_hold       = 25,

    # ── Kept for compatibility ──
    z_entry        = 1.25,
    z_exit         = 0.30,
    z_stop         = 3.0,

    # ── Autocorrelation regime filter ──
    autocorr_window    = 20,
    autocorr_threshold = 0.1,
    ou_adapt_span      = 252,
)
