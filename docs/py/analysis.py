"""One synthetic control analysis of a city in the Real-Time Crime Index.

This is the code the website runs (in the browser, through Pyodide), and
the tests run the same file natively. ``Panel`` holds the city-month crime
counts built by ``scripts/build_data.py``; ``run`` fits one treated city
and returns everything the page draws, as plain lists and dicts.

The model is fit on monthly rates per 100,000 residents, the treated city's
rate regressed on the donor cities' rates over the pre-period, with the
estimators and conformal intervals in ``synth_methods`` (from SynthPower).
"""

import json
import time
from datetime import date

import numpy as np

import synth_methods as m

# Outcome: (label, component crimes). Violent and property are the sums of
# their components, missing if any component is missing (as in CrimeDecomp).
CRIMES = {
    "violent": ("Violent crime", ["murder", "rape", "robbery", "assault"]),
    "murder": ("Murder", ["murder"]),
    "rape": ("Rape", ["rape"]),
    "robbery": ("Robbery", ["robbery"]),
    "assault": ("Aggravated assault", ["assault"]),
    "property": ("Property crime", ["burglary", "theft", "motor"]),
    "burglary": ("Burglary", ["burglary"]),
    "theft": ("Theft", ["theft"]),
    "motor": ("Motor vehicle theft", ["motor"]),
}

ESTIMATORS = {
    "lasso": "Lasso with intercept",
    "lasso_noint": "Lasso without intercept",
    "synth": "Classic synthetic control",
}

INTERVALS = {
    "rolling": "Rolling-origin forecast errors",
    "jackknife": "Jackknife, independent cumulative draws",
    "block": "Jackknife, block sums",
}

# Model settings when a config leaves them out. The city, outcome and start
# date have no default; docs/examples.json has complete example configs.
DEFAULTS = {
    "city": None,
    "crime": None,
    "start": None,  # intervention start date, YYYY-MM-DD
    "partial": "pre",  # the month the intervention starts in: "pre", "post", or "drop"
    "first": None,  # first month of the analysis window, default the first month of data
    "last": None,  # last month, default the latest month of data
    "estimator": "lasso",
    "penalty": "cv",  # "cv" or "fixed"
    "alpha": 1.0,  # lasso penalty when penalty == "fixed"
    "interval": "rolling",
    "min_train": 36,  # months in the first rolling-origin training window
    "level": 0.95,
    "min_pop": 0,  # smallest donor population
    "exclude": [],  # donor cities to leave out, by agency ID
    "cumsim": 1000,  # simulations for the independent-draws cumulative band
    "seed": 10,
}

MIN_PRE = 12
MAX_CACHE = 8


class AnalysisError(ValueError):
    """A configuration that cannot be fit; the message is shown to the user."""


def _month_add(ym, k):
    y, mo = int(ym[:4]), int(ym[5:7])
    t = y * 12 + (mo - 1) + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def _month_label(ym):
    return date(int(ym[:4]), int(ym[5:7]), 1).strftime("%b %Y")


class Panel:
    def __init__(self, data):
        if isinstance(data, str):
            data = json.loads(data)
        self.source = data.get("source", {})
        self.dates = list(data["dates"])
        self.cities = data["cities"]
        self.ids = [c["id"] for c in self.cities]
        self.index = {c: i for i, c in enumerate(self.ids)}
        self.labels = np.array([c["label"] for c in self.cities], dtype=object)
        self.pop = np.array([c["pop"] for c in self.cities], dtype=float)
        self.counts = {k: np.array(v, dtype=float) for k, v in data["counts"].items()}
        self._rates = {}
        self._cache = {}

    def rates(self, crime):
        """Cities by months, monthly crimes per 100,000 residents."""
        if crime not in self._rates:
            comps = CRIMES[crime][1]
            total = sum(self.counts[k] for k in comps)
            self._rates[crime] = total / self.pop[:, None] * 1e5
        return self._rates[crime]


def resolve_periods(dates, cfg):
    """Analysis window months, the number of pre-period months, and the partial month."""
    first = cfg["first"] or dates[0]
    last = cfg["last"] or dates[-1]
    try:
        start = date.fromisoformat(cfg["start"])
    except (TypeError, ValueError):
        raise AnalysisError("Enter the intervention start date as YYYY-MM-DD.")
    start_month = f"{start.year:04d}-{start.month:02d}"
    partial = start_month if start.day > 1 else None
    first_post = _month_add(start_month, 1) if partial and cfg["partial"] in ("pre", "drop") else start_month
    window = [d for d in dates if first <= d <= last]
    if partial and cfg["partial"] == "drop":
        window = [d for d in window if d != partial]
    T0 = sum(d < first_post for d in window)
    H = len(window) - T0
    if H < 1:
        raise AnalysisError(f"No data after the intervention: the data end in {_month_label(dates[-1])} "
                            f"and the post-period would start in {_month_label(first_post)}.")
    if T0 < MIN_PRE:
        raise AnalysisError(f"Only {T0} pre-intervention months in the window; at least {MIN_PRE} are needed.")
    return window, T0, partial, first_post


def _fitter(cfg, X_pre, y_pre):
    """f(X, y) -> fit, with the lasso penalty chosen once on the pre-period."""
    if cfg["estimator"] == "synth":
        return m.fit_synth, None
    intercept = cfg["estimator"] == "lasso"
    if cfg["penalty"] == "fixed":
        alpha = float(cfg["alpha"])
        if not alpha > 0:
            raise AnalysisError("The fixed lasso penalty must be greater than 0.")
    else:
        alpha = float(m.lasso_alpha(X_pre, y_pre, intercept=intercept))
    return (lambda X, y: m.fit_lasso(X, y, alpha, intercept=intercept)), alpha


def _clean(x):
    """JSON-safe copy: numpy to lists, non-finite numbers to None."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run(panel, config=None, progress=None):
    """Fit one analysis. ``config`` overrides ``DEFAULTS``; returns a dict."""
    say = progress or (lambda msg: None)
    cfg = {**DEFAULTS, **(config or {})}
    for k, what in [("city", "a treated city"), ("crime", "an outcome"), ("start", "the intervention start date")]:
        if not cfg[k]:
            raise AnalysisError(f"Choose {what}.")
    if cfg["crime"] not in CRIMES:
        raise AnalysisError(f"Unknown outcome {cfg['crime']!r}.")
    if cfg["estimator"] not in ESTIMATORS:
        raise AnalysisError(f"Unknown estimator {cfg['estimator']!r}.")
    if cfg["interval"] not in INTERVALS:
        raise AnalysisError(f"Unknown interval method {cfg['interval']!r}.")
    if cfg["city"] not in panel.index:
        raise AnalysisError(f"Unknown city {cfg['city']!r}.")
    level = float(cfg["level"])
    if not 0.5 <= level < 1:
        raise AnalysisError("The interval level must be between 50% and 99.9%.")
    a = 1 - level
    timing = {}
    t_start = time.perf_counter()

    window, T0, partial, first_post = resolve_periods(panel.dates, cfg)
    H = len(window) - T0
    cols = np.array([panel.dates.index(d) for d in window])
    R = panel.rates(cfg["crime"])[:, cols]
    tr = panel.index[cfg["city"]]
    y = R[tr]
    if np.isnan(y).any():
        miss = [_month_label(window[i]) for i in np.where(np.isnan(y))[0]]
        raise AnalysisError(f"{panel.labels[tr]} is missing {CRIMES[cfg['crime']][0].lower()} data for "
                            f"{len(miss)} month(s) in the window ({', '.join(miss[:6])}"
                            f"{', ...' if len(miss) > 6 else ''}).")

    excluded = set(cfg["exclude"] or []) - {cfg["city"]}
    small = panel.pop < float(cfg["min_pop"] or 0)
    missing = np.isnan(R).any(axis=1)
    base = np.array([i != tr and panel.ids[i] not in excluded for i in range(len(panel.ids))])
    donors = np.where(base & ~small & ~missing)[0]
    if len(donors) < 2:
        raise AnalysisError("Fewer than two donor cities are left after the filters.")
    X = R[donors].T

    key = (cfg["city"], cfg["crime"], tuple(window), T0, cfg["estimator"], cfg["penalty"],
           float(cfg["alpha"]) if cfg["penalty"] == "fixed" else None, tuple(donors))
    ent = panel._cache.get(key)
    if ent is None:
        if cfg["estimator"] == "synth":
            say("Fitting the synthetic control")
        else:
            say("Choosing the lasso penalty by cross-validation" if cfg["penalty"] == "cv" else "Fitting the lasso")
        t = time.perf_counter()
        fitter, alpha = _fitter(cfg, X[:T0], y[:T0])
        fit = fitter(X[:T0], y[:T0])
        timing["fit"] = time.perf_counter() - t
        ent = {"fitter": fitter, "alpha": alpha, "fit": fit, "rolling": {}}
        panel._cache[key] = ent
        while len(panel._cache) > MAX_CACHE:
            panel._cache.pop(next(iter(panel._cache)))
    fitter, fit = ent["fitter"], ent["fit"]

    pred = fit.predict(X)
    gap = y - pred
    dif = gap[T0:]
    pre = gap[:T0]
    rmse = float(np.sqrt(np.mean(pre**2)))
    r2 = float(1 - np.sum(pre**2) / np.sum((y[:T0] - y[:T0].mean()) ** 2))

    # Conformal intervals
    if cfg["interval"] in ("jackknife", "block"):
        if "jackknife" not in ent:
            say(f"Jackknife: refitting {T0} times, leaving out one pre-period month each time")
            t = time.perf_counter()
            ent["jackknife"] = m.jackknife(fitter, X[:T0], y[:T0])
            timing["jackknife"] = time.perf_counter() - t
        r = ent["jackknife"]
        q = np.full(H, m.conformal_quantile(r, a))
        if cfg["interval"] == "jackknife":
            cum, clo, chi = m.cum_band_iid(dif, r, a, nsim=int(cfg["cumsim"]), rng=int(cfg["seed"]))
        else:
            cum, clo, chi = m.cum_band_block(dif, r, a)
        n_ref = np.full(H, T0)
    else:
        min_train = int(cfg["min_train"])
        if not 2 <= min_train < T0:
            raise AnalysisError(f"The minimum training window must be between 2 and {T0 - 1} months.")
        if min_train not in ent["rolling"]:
            say(f"Rolling-origin forecasts: refitting {T0 - min_train} times")
            t = time.perf_counter()
            ent["rolling"][min_train] = m.rolling_origin_errors(fitter, X[:T0], y[:T0], H, min_train)
            timing["rolling"] = time.perf_counter() - t
        E = ent["rolling"][min_train]
        (plo, _), (clo, chi) = m.rolling_origin_bands(dif, E, a)
        q = dif - plo
        cum = np.cumsum(dif)
        n_ref = (~np.isnan(E)).sum(axis=0)

    post_pred = pred[T0:]
    pop = panel.pop[tr]
    pred_total = float(post_pred.sum())

    def pct(v):
        return 100 * v / pred_total if pred_total > 0 else np.nan

    # Donor weights, largest first
    w = np.asarray(fit.w, dtype=float)
    nz = np.where(w > 1e-6)[0]
    nz = nz[np.argsort(-w[nz])]
    pre_mean = X[:T0].mean(axis=0)
    weights = [{"id": panel.ids[donors[j]], "label": panel.labels[donors[j]], "pop": panel.pop[donors[j]],
                "coef": w[j], "pre_mean": pre_mean[j], "contrib": w[j] * pre_mean[j]} for j in nz]

    timing["total"] = time.perf_counter() - t_start
    out = {
        "config": cfg,
        "crime_label": CRIMES[cfg["crime"]][0],
        "estimator_label": ESTIMATORS[cfg["estimator"]],
        "interval_label": INTERVALS[cfg["interval"]],
        "city": {"id": cfg["city"], "label": panel.labels[tr], "pop": pop},
        "dates": window,
        "T0": T0,
        "H": H,
        "first_post": first_post,
        "partial": partial if cfg["partial"] != "post" else None,
        "obs": y,
        "pred": pred,
        "gap": gap,
        "point_q": q,
        "pred_lo": post_pred - q,
        "pred_hi": post_pred + q,
        "dif": dif,
        "dif_lo": dif - q,
        "dif_hi": dif + q,
        "cum": cum,
        "cum_lo": clo,
        "cum_hi": chi,
        "n_ref": n_ref,
        "fit": {
            "rmse": rmse,
            "r2": r2,
            "alpha": ent["alpha"],
            "intercept": fit.intercept,
            "n_donors": len(donors),
            "n_nonzero": len(nz),
        },
        "summary": {
            "obs_total": float(y[T0:].sum()),
            "pred_total": pred_total,
            "cum": cum[-1],
            "cum_lo": clo[-1],
            "cum_hi": chi[-1],
            "pct": pct(cum[-1]),
            "pct_lo": pct(clo[-1]),
            "pct_hi": pct(chi[-1]),
            "count": cum[-1] * pop / 1e5,
            "count_lo": clo[-1] * pop / 1e5,
            "count_hi": chi[-1] * pop / 1e5,
            "excludes_zero": bool(clo[-1] > 0 or chi[-1] < 0),
        },
        "weights": weights,
        "donors": {
            "ids": [panel.ids[i] for i in donors],
            "labels": panel.labels[donors].tolist(),
            "pops": panel.pop[donors],
            "rates": np.round(R[donors], 3),
        },
        "dropped": {
            "excluded": [panel.labels[i] for i in np.where(~base)[0] if i != tr],
            "missing": [panel.labels[i] for i in np.where(base & missing)[0]],
            "small": int((base & ~missing & small).sum()),
        },
        "timing": timing,
    }
    return _clean(out)


def run_json(panel, config_json, progress=None):
    """JSON in, JSON out, for the web worker. Errors come back as {"error": msg}."""
    try:
        return json.dumps(run(panel, json.loads(config_json), progress))
    except AnalysisError as e:
        return json.dumps({"error": str(e)})
