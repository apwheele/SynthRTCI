"""Synthetic control estimators, placebo tests and conformal inference.

Copied from ``src/synthpower/methods.py`` in
https://github.com/apwheele/SynthPower, keeping the estimators, the placebo
tests and the conformal methods (the block permutation test is left out).
The one change is an ``intercept`` option for the lasso; with
``intercept=True`` (the default) the code is the same as SynthPower's.

Estimators of the counterfactual for a treated unit:

- ``synth``: the classic synthetic control, donor weights that are
  non-negative and sum to one, chosen to minimize pre-period squared error.
- ``lasso``: a lasso regression of the treated series on the donor series,
  with non-negative coefficients and an intercept (Wheeler 2019).

Intervals and tests:

- ``placebo_pvalues`` and ``placebo_band``: in-space placebos (Abadie et
  al. 2010), with a cumulative band whose width comes from the placebos.
- ``jackknife``: leave-one-period-out prediction errors in the pre-period,
  with cumulative bands built from those errors either independently per
  period (``cum_band_iid``, as in Wheeler 2023) or as consecutive blocks
  (``cum_band_block``).
- ``rolling_origin_errors``: out-of-sample forecast errors from rolling
  origins inside the pre-period, and the pointwise and cumulative bands
  built from them (``rolling_origin_bands``).
"""

import warnings

import numpy as np
from scipy.optimize import nnls
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, lasso_path

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Estimators. Each ``fit_*`` returns an object with ``predict(X)``.


class SynthFit:
    def __init__(self, w):
        self.w = w
        self.intercept = 0.0

    def predict(self, X):
        return X @ self.w


def fit_synth(X, y, penalty=1e3):
    """Weights w >= 0 with sum(w) = 1 minimizing ||y - Xw||^2.

    Solved as non-negative least squares with an extra, heavily weighted row
    that enforces the sum-to-one constraint.
    """
    s = np.abs(y).mean()
    s = s if s > 0 else 1.0
    Xa = np.vstack([X / s, penalty * np.ones(X.shape[1])])
    ya = np.append(y / s, penalty)
    w, _ = nnls(Xa, ya, maxiter=50 * X.shape[1])
    return SynthFit(w / w.sum())


class LassoFit:
    def __init__(self, model):
        self.model = model
        self.w = model.coef_
        self.intercept = model.intercept_

    def predict(self, X):
        return self.model.predict(X)


def lasso_alpha(X, y, n_alphas=100, cv=5, max_iter=10000, intercept=True):
    """Penalty chosen by cross-validation, as in LassoSynth.suggest_alpha.

    Same 5-fold LassoCV as the original, except that penalties so large that
    no donor gets a weight (an intercept-only model) are not allowed.
    """
    lcv = LassoCV(positive=True, fit_intercept=intercept, alphas=n_alphas, cv=cv, max_iter=max_iter)
    lcv.fit(X, y)
    if np.any(fit_lasso(X, y, lcv.alpha_, max_iter, intercept).w > 0):
        return lcv.alpha_
    mse = lcv.mse_path_.mean(axis=1)
    Xc, yc = (X - X.mean(axis=0), y - y.mean()) if intercept else (X, y)
    _, coefs, _ = lasso_path(Xc, yc, alphas=lcv.alphas_, positive=True, max_iter=max_iter)
    ok = (coefs > 0).any(axis=0)
    if not ok.any():
        return lcv.alphas_[-1]
    return lcv.alphas_[ok][np.argmin(mse[ok])]


def fit_lasso(X, y, alpha, max_iter=10000, intercept=True):
    m = Lasso(alpha=alpha, positive=True, fit_intercept=intercept, max_iter=max_iter)
    m.fit(X, y)
    return LassoFit(m)


# ---------------------------------------------------------------------------
# Placebo inference


def gaps(fitter, Y, unit, donors, T0):
    """Observed minus synthetic for ``unit`` using ``donors`` (column indices)."""
    X = Y[:, donors]
    y = Y[:, unit]
    fit = fitter(X[:T0], y[:T0])
    return y - fit.predict(X)


def placebo_stats(gap, T0):
    pre, post = gap[:T0], gap[T0:]
    rmspe_pre = np.sqrt(np.mean(pre**2))
    return {
        "ratio": np.sqrt(np.mean(post**2)) / rmspe_pre,
        "cum": np.abs(post.sum()) / rmspe_pre,
    }


def placebo_pvalues(treated_stats, placebo_stats_list):
    """p = (1 + #placebos at least as extreme) / (N placebos + 1)."""
    n = len(placebo_stats_list)
    out = {}
    for k, v in treated_stats.items():
        other = np.array([p[k] for p in placebo_stats_list])
        out[k] = (1 + np.sum(other >= v)) / (n + 1)
    return out


def placebo_band(gap, placebo_gaps, T0, alpha=0.05):
    """Cumulative band whose width comes from the placebos.

    At each post-period horizon, placebo cumulative gaps are scaled by each
    placebo's pre-period RMSPE, and the conformal quantile of those scaled
    values is multiplied by the treated unit's pre-period RMSPE. The band
    excludes zero at the last period exactly when the placebo test on
    |cumulative gap| / pre-period RMSPE rejects.
    """
    P = np.asarray(placebo_gaps)
    scale = np.sqrt(np.mean(P[:, :T0] ** 2, axis=1))
    pcum = np.cumsum(P[:, T0:], axis=1) / scale[:, None]
    q = np.array([conformal_quantile(pcum[:, h], alpha) for h in range(pcum.shape[1])])
    cum = np.cumsum(gap[T0:])
    width = q * np.sqrt(np.mean(gap[:T0] ** 2))
    return cum, cum - width, cum + width


# ---------------------------------------------------------------------------
# Jackknife conformal inference, weights fit on the pre-period only


def jackknife(fitter, X_pre, y_pre):
    """Leave-one-period-out prediction errors in the pre-period."""
    T0 = len(y_pre)
    r = np.empty(T0)
    idx = np.arange(T0)
    for t in range(T0):
        keep = idx != t
        fit = fitter(X_pre[keep], y_pre[keep])
        r[t] = y_pre[t] - fit.predict(X_pre[t : t + 1])[0]
    return r


def conformal_quantile(scores, alpha):
    """Finite-sample conformal quantile of the absolute scores.

    Infinite when there are too few scores to reach the 1 - alpha level.
    """
    a = np.sort(np.abs(scores))
    n = len(a)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    return np.inf if k > n else a[k - 1]


def cum_band_iid(dif, r, alpha=0.05, nsim=2000, rng=None, match_pointwise=True):
    """Cumulative band from independent draws of +/- jackknife errors.

    This is the LassoSynth.effects() approach: each post period gets its own
    independent draw from the symmetrized error distribution, and the band
    comes from percentiles of the simulated cumulative sums.

    In the first post period the cumulative effect is that period's gap, so
    the band should equal the pointwise conformal interval. The simulated
    percentiles skip the finite-sample adjustment and come out a little
    narrower (about 7% with 20 pre-periods). With ``match_pointwise`` (the
    default), the half-widths in every period are scaled by the factor that
    makes the first period equal the pointwise interval. Set it to False for
    the original percentiles.
    """
    rng = np.random.default_rng(rng)
    cs = np.concatenate([np.abs(r), -np.abs(r)])
    draws = rng.choice(cs, size=(nsim, len(dif)))
    cum = np.cumsum(dif)
    sims = np.cumsum(dif[None, :] + draws, axis=1)
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
    if not match_pointwise:
        return cum, lo, hi
    half = (hi - lo) / 2
    scale = conformal_quantile(r, alpha) / half[0] if half[0] > 0 else np.inf
    width = half * scale
    return cum, cum - width, cum + width


def block_sums(r, h):
    """All sums of h cyclically consecutive values of r."""
    T = len(r)
    rr = np.concatenate([r, r[: h - 1]]) if h > 1 else r
    c = np.concatenate([[0.0], np.cumsum(rr)])
    return c[h : h + T] - c[:T]


def cum_band_block(dif, r, alpha=0.05):
    """Cumulative band from sums of consecutive jackknife errors.

    For horizon h, the reference distribution is every sum of h consecutive
    pre-period errors (wrapping around), plus their negatives.
    """
    cum = np.cumsum(dif)
    lo = np.empty(len(dif))
    hi = np.empty(len(dif))
    for h in range(1, len(dif) + 1):
        s = block_sums(r, h)
        q = conformal_quantile(s, alpha)
        lo[h - 1], hi[h - 1] = cum[h - 1] - q, cum[h - 1] + q
    return cum, lo, hi


def rolling_origin_errors(fitter, X_pre, y_pre, H, min_train=10):
    """Out-of-sample forecast errors from rolling origins within the pre-period.

    For each origin s (min_train <= s < T0), fit on periods 0..s-1 and
    forecast periods s..s+H-1 that are still inside the pre-period. Returns
    an (origins by H) array, NaN where the forecast would pass the end of
    the pre-period.
    """
    T0 = len(y_pre)
    origins = range(min_train, T0)
    E = np.full((len(origins), H), np.nan)
    for i, s in enumerate(origins):
        fit = fitter(X_pre[:s], y_pre[:s])
        k = min(H, T0 - s)
        E[i, :k] = y_pre[s : s + k] - fit.predict(X_pre[s : s + k])
    return E


def rolling_origin_bands(dif, E, alpha=0.05):
    """Pointwise and cumulative conformal bands from rolling-origin errors.

    Year h uses the h-step-ahead forecast errors (pointwise) or the sums of
    the first h forecast errors (cumulative) from every origin with a full
    h-year forecast. Infinite when too few origins reach year h.
    """
    H = len(dif)
    cum = np.cumsum(dif)
    qp = np.array([conformal_quantile(E[~np.isnan(E[:, h]), h], alpha) for h in range(H)])
    S = np.cumsum(E, axis=1)
    qc = np.array([conformal_quantile(S[~np.isnan(S[:, h]), h], alpha) for h in range(H)])
    return (dif - qp, dif + qp), (cum - qc, cum + qc)
