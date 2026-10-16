"""Tests for the analysis code the website runs (docs/py/).

The comparison with SynthPower needs a local copy of
https://github.com/apwheele/SynthPower, found through the SYNTHPOWER_SRC
environment variable or next to this repository (../SynthPower/src);
those tests are skipped without it.

The data change with every RTCI update: a month is added, past months are
revised, and agencies leave the sample (when their latest month is late) or
join it. Nothing here should depend on that. The examples' agencies, for
instance, are replaced by a stand-in city when they are not in the current
sample, with a warning.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "py"))

import analysis as A  # noqa: E402
import synth_methods as m  # noqa: E402

# RTCI_DATA: test against another data file, e.g. one with agencies removed
DATA = Path(os.environ.get("RTCI_DATA", ROOT / "docs" / "data" / "rtci.json"))
EXAMPLES = json.loads((ROOT / "docs" / "examples.json").read_text())
SYNTHPOWER = Path(os.environ.get("SYNTHPOWER_SRC", ROOT.parent / "SynthPower" / "src"))
PANEL = A.Panel(DATA.read_text())


def example_config(key):
    """An example's settings, with a stand-in treated city if the example's
    city is not in the current RTCI sample: the largest city with complete
    data for the outcome that the example does not leave out."""
    cfg = dict(EXAMPLES[key]["config"])
    if cfg["city"] not in PANEL.index:
        complete = ~np.isnan(PANEL.rates(cfg["crime"])).any(axis=1)
        ok = [i for i in np.where(complete)[0] if PANEL.ids[i] not in cfg["exclude"]]
        stand_in = PANEL.ids[max(ok, key=lambda i: PANEL.pop[i])]
        warnings.warn(f"{cfg['city']} ({key} example) is not in the current RTCI sample; testing with {stand_in}")
        cfg["city"] = stand_in
    return cfg


MEMPHIS = example_config("memphis")
TREATED = MEMPHIS["city"]
LA = example_config("la_gascon")


@pytest.fixture(scope="module")
def panel():
    return PANEL


@pytest.fixture(scope="module")
def synthpower():
    if not (SYNTHPOWER / "synthpower" / "lassosynth.py").exists():
        pytest.skip(f"SynthPower source not found at {SYNTHPOWER}")
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(SYNTHPOWER))
    from synthpower import lassosynth, methods

    return lassosynth, methods


def memphis(panel, crime="violent"):
    """Treated series and donor matrix for the default Memphis example."""
    R = panel.rates(crime)
    tr = panel.index[TREATED]
    excl = set(MEMPHIS["exclude"]) | {TREATED}
    donors = [i for i in range(len(panel.ids)) if panel.ids[i] not in excl and not np.isnan(R[i]).any()]
    T0 = panel.dates.index("2025-10")
    return R[tr], R[donors].T, T0, donors


# ---------------------------------------------------------------------------
# Data file


def test_data_file(panel):
    d = json.loads(DATA.read_text())
    months = d["dates"]
    y, mo = map(int, months[0].split("-"))
    expect = [f"{(y * 12 + mo - 1 + k) // 12:04d}-{(y * 12 + mo - 1 + k) % 12 + 1:02d}" for k in range(len(months))]
    assert months == expect
    assert len({c["label"] for c in d["cities"]}) == len(d["cities"])
    assert all(c["pop"] > 0 for c in d["cities"])
    for crime in ["murder", "rape", "robbery", "assault", "burglary", "theft", "motor"]:
        v = panel.counts[crime]
        assert v.shape == (len(d["cities"]), len(months))
        ok = ~np.isnan(v)
        assert (v[ok] >= 0).all()


def test_examples_in_sample(panel):
    """Agencies come and go between RTCI samples, so a missing example agency
    is a warning, not a failure (the page marks that example unavailable, and
    ignores left-out cities that are not in the data)."""
    for key, ex in EXAMPLES.items():
        assert {"city", "crime", "start"} <= set(ex["config"]) <= set(A.DEFAULTS)
        missing = [c for c in [ex["config"]["city"]] + ex["config"]["exclude"] if c not in panel.index]
        if missing:
            warnings.warn(f"Not in the current RTCI sample ({key} example): {', '.join(missing)}")


def test_combined_crimes(panel):
    parts = ["murder", "rape", "robbery", "assault"]
    total = sum(panel.counts[k] for k in parts)
    np.testing.assert_allclose(panel.rates("violent"), total / panel.pop[:, None] * 1e5)
    # Missing if any component is missing
    miss = np.isnan(np.stack([panel.counts[k] for k in parts])).any(axis=0)
    assert np.array_equal(np.isnan(panel.rates("violent")), miss)


# ---------------------------------------------------------------------------
# Periods and errors


@pytest.mark.parametrize("start,partial,first_post,is_partial,drop", [
    ("2025-10-01", "pre", "2025-10", False, False),
    ("2025-09-29", "pre", "2025-10", True, False),
    ("2025-09-29", "post", "2025-09", True, False),
    ("2025-09-29", "drop", "2025-10", True, True),
])
def test_resolve_periods(panel, start, partial, first_post, is_partial, drop):
    cfg = {**A.DEFAULTS, **MEMPHIS, "start": start, "partial": partial}
    window, T0, part, fp = A.resolve_periods(panel.dates, cfg)
    assert fp == first_post
    assert (part is not None) == is_partial
    assert ("2025-09" not in window) == drop
    assert window[T0] == first_post
    assert all(d < first_post for d in window[:T0])


def test_window(panel):
    window, T0, _, _ = A.resolve_periods(panel.dates, {**A.DEFAULTS, **MEMPHIS, "first": "2019-01", "last": "2026-03"})
    assert window[0] == "2019-01" and window[-1] == "2026-03"
    assert window[T0] == "2025-10"


@pytest.mark.parametrize("cfg,msg", [
    ({"start": "2030-01-15"}, "No data after the intervention"),
    ({"start": "2017-06-01"}, "pre-intervention months"),
    ({"start": "not a date"}, "YYYY-MM-DD"),
    ({"city": "XX0000000"}, "Unknown city"),
    ({"crime": "arson"}, "Unknown outcome"),
    ({"penalty": "fixed", "alpha": 0}, "greater than 0"),
    ({"min_train": 500}, "minimum training window"),
    ({"exclude": [], "min_pop": 1e9}, "Fewer than two donor"),
])
def test_errors(panel, cfg, msg):
    with pytest.raises(A.AnalysisError, match=msg):
        A.run(panel, {**MEMPHIS, **cfg})


def test_required_settings(panel):
    for k in ["city", "crime", "start"]:
        with pytest.raises(A.AnalysisError, match="Choose"):
            A.run(panel, {**MEMPHIS, k: None})
    with pytest.raises(A.AnalysisError, match="Choose a treated city"):
        A.run(panel, {})


def test_missing_treated_data():
    d = json.loads(DATA.read_text())
    i = [c["id"] for c in d["cities"]].index(TREATED)
    d["counts"]["rape"][i][d["dates"].index("2024-03")] = None
    with pytest.raises(A.AnalysisError, match="missing rape data for 1 month"):
        A.run(A.Panel(json.dumps(d)), {**MEMPHIS, "crime": "rape"})


def test_run_json(panel):
    out = json.loads(A.run_json(panel, json.dumps({**MEMPHIS, "interval": "block"})),
                     parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert out["city"]["label"] == panel.labels[panel.index[TREATED]]
    assert len(out["obs"]) == len(out["dates"]) == out["T0"] + out["H"]
    err = json.loads(A.run_json(panel, json.dumps({**MEMPHIS, "start": "2030-01-15"})))
    assert "error" in err


# ---------------------------------------------------------------------------
# Results


def test_donor_pool(panel):
    out = A.run(panel, {**MEMPHIS, "interval": "block"})
    _, X, _, donors = memphis(panel)
    assert out["fit"]["n_donors"] == len(donors) == X.shape[1]
    assert TREATED not in out["donors"]["ids"]
    assert not set(MEMPHIS["exclude"]) & set(out["donors"]["ids"])
    assert "CA0190000" in MEMPHIS["exclude"]  # LA County Sheriff
    assert sorted(out["dropped"]["excluded"]) == sorted(
        panel.labels[panel.index[c]] for c in MEMPHIS["exclude"] if c != TREATED and c in panel.index)
    # Left-out cities that are not in the data (not in this RTCI sample) are ignored
    gone = A.run(panel, {**MEMPHIS, "interval": "block", "exclude": MEMPHIS["exclude"] + ["XX9999999"]})
    assert gone["donors"]["ids"] == out["donors"]["ids"]


def test_summary_consistent(panel):
    out = A.run(panel, MEMPHIS)
    s, pop = out["summary"], out["city"]["pop"]
    dif = np.array(out["obs"][out["T0"]:]) - np.array(out["pred"][out["T0"]:])
    np.testing.assert_allclose(out["dif"], dif)
    np.testing.assert_allclose(out["cum"], np.cumsum(dif))
    assert s["cum"] == pytest.approx(dif.sum())
    assert s["pct"] == pytest.approx(100 * dif.sum() / s["pred_total"])
    assert s["count"] == pytest.approx(dif.sum() * pop / 1e5)
    assert s["cum_lo"] < s["cum"] < s["cum_hi"]
    # Weights reproduce the prediction
    X = np.array(out["donors"]["rates"]).T
    w = dict(zip(out["donors"]["ids"], range(X.shape[1])))
    pred = out["fit"]["intercept"] + sum(x["coef"] * X[:, w[x["id"]]] for x in out["weights"])
    np.testing.assert_allclose(pred, out["pred"], atol=0.05)  # donor rates are rounded to 3 decimals


def test_estimators(panel):
    synth = A.run(panel, {**MEMPHIS, "estimator": "synth", "interval": "block"})
    assert sum(w["coef"] for w in synth["weights"]) == pytest.approx(1, abs=1e-6)
    assert synth["fit"]["intercept"] == 0 and synth["fit"]["alpha"] is None
    noint = A.run(panel, {**MEMPHIS, "estimator": "lasso_noint", "interval": "block"})
    assert noint["fit"]["intercept"] == 0
    fixed = A.run(panel, {**MEMPHIS, "penalty": "fixed", "alpha": 5, "interval": "block"})
    assert fixed["fit"]["alpha"] == 5


def test_level_and_infinite_bands(panel):
    wide = A.run(panel, {**MEMPHIS, "level": 0.99, "interval": "jackknife"})
    narrow = A.run(panel, {**MEMPHIS, "level": 0.8, "interval": "jackknife"})
    # A 99% rolling band needs 99+ forecasts per horizon; T0 - 36 - H + 1 reach the last month
    out99 = A.run(panel, {**MEMPHIS, "level": 0.99})
    assert out99["n_ref"][-1] == max(0, out99["T0"] - 36 - out99["H"] + 1) < 99
    assert out99["summary"]["cum_lo"] is None
    assert wide["summary"]["cum_hi"] - wide["summary"]["cum_lo"] > narrow["summary"]["cum_hi"] - narrow["summary"]["cum_lo"]
    # Too few rolling origins for a 95% band: T0 - 90 - H + 1 reach the last month
    few = A.run(panel, {**MEMPHIS, "min_train": 90})
    assert few["n_ref"][-1] == max(0, few["T0"] - 90 - few["H"] + 1)
    assert few["cum_lo"][-1] is None and few["summary"]["excludes_zero"] is False


def test_la_gascon_example(panel):
    """A fixed past window (December 2020 to November 2024) and its donor pool.

    The estimate is not tested: it moves with each RTCI update, since the
    sample's agencies (the donor pool) change and past months are revised.
    SynthPower's Los Angeles lasso result is +911 per 100,000 (+16.7%); the
    site gave +895 (16.4%) with the June 2026 sample and +780 (14.0%) with
    August 2026.
    """
    out = A.run(panel, LA)
    assert (out["T0"], out["H"], out["first_post"]) == (47, 48, "2020-12")
    assert out["dates"][0] == "2017-01" and out["dates"][-1] == "2024-11"
    # Donors: every other city with complete theft data in the window, less those left out
    cols = [panel.dates.index(d) for d in out["dates"]]
    complete = ~np.isnan(panel.rates("theft")[:, cols]).any(axis=1)
    expect = [panel.ids[i] for i in np.where(complete)[0] if panel.ids[i] not in LA["exclude"] + [LA["city"]]]
    assert sorted(out["donors"]["ids"]) == sorted(expect)
    s = out["summary"]
    assert np.isfinite([s["cum"], s["pct"], s["cum_lo"], s["cum_hi"]]).all()


def test_intercept_default_matches_synthpower_lasso(panel):
    """With an intercept, the lasso helpers are SynthPower's code unchanged."""
    y, X, T0, _ = memphis(panel)
    a = m.lasso_alpha(X[:T0], y[:T0])
    assert a == m.lasso_alpha(X[:T0], y[:T0], intercept=True)
    fit = m.fit_lasso(X[:T0], y[:T0], a)
    assert fit.intercept != 0 and (fit.w >= 0).all()


# ---------------------------------------------------------------------------
# Same answers as SynthPower


def test_methods_match_synthpower(panel, synthpower):
    _, sp = synthpower
    y, X, T0, _ = memphis(panel)
    a = m.lasso_alpha(X[:T0], y[:T0])
    assert a == sp.lasso_alpha(X[:T0], y[:T0])
    f_ours = lambda X_, y_: m.fit_lasso(X_, y_, a)  # noqa: E731
    f_sp = lambda X_, y_: sp.fit_lasso(X_, y_, a)  # noqa: E731
    np.testing.assert_array_equal(f_ours(X[:T0], y[:T0]).w, f_sp(X[:T0], y[:T0]).w)
    r = m.jackknife(f_ours, X[:T0], y[:T0])
    np.testing.assert_allclose(r, sp.jackknife(f_sp, X[:T0], y[:T0]))
    H = len(y) - T0
    E = m.rolling_origin_errors(f_ours, X[:T0], y[:T0], H, 36)
    np.testing.assert_allclose(E, sp.rolling_origin_errors(f_sp, X[:T0], y[:T0], H, 36))
    dif = y[T0:] - f_ours(X[:T0], y[:T0]).predict(X[T0:])
    for ours, theirs in [(m.cum_band_iid(dif, r, 0.05, 1000, 10), sp.cum_band_iid(dif, r, 0.05, 1000, 10)),
                         (m.cum_band_block(dif, r), sp.cum_band_block(dif, r)),
                         (m.rolling_origin_bands(dif, E), sp.rolling_origin_bands(dif, E))]:
        np.testing.assert_allclose(np.array(ours, dtype=float), np.array(theirs, dtype=float))
    np.testing.assert_allclose(m.fit_synth(X[:T0], y[:T0]).w, sp.fit_synth(X[:T0], y[:T0]).w)


def test_run_matches_lassosynth(panel, synthpower):
    """The website's numbers equal SynthPower's LassoSynth class on the same data."""
    lassosynth, _ = synthpower
    import pandas as pd

    y, X, T0, donors = memphis(panel)
    wide = pd.DataFrame(X, columns=[panel.ids[i] for i in donors])
    wide[TREATED] = y
    s = lassosynth.Synth(wide, TREATED, post=T0)
    s.suggest_alpha()
    s.fit()
    s.rolling(min_train=36)
    eff = s.effects(alpha=0.05, cumsim=1000)

    jk = A.run(panel, {**MEMPHIS, "interval": "jackknife"})
    assert jk["fit"]["alpha"] == s.alpha
    assert jk["fit"]["rmse"] == pytest.approx(s.stats["RMSE"])
    assert jk["fit"]["r2"] == pytest.approx(s.stats["RSquare"])
    np.testing.assert_allclose(jk["pred"][T0:], eff["Pred"])
    np.testing.assert_allclose(jk["pred_lo"], eff["Low"])
    np.testing.assert_allclose(jk["pred_hi"], eff["High"])
    np.testing.assert_allclose(jk["cum"], eff["CumDif"])
    np.testing.assert_allclose(jk["cum_lo"], eff["CumDifLow"])
    np.testing.assert_allclose(jk["cum_hi"], eff["CumDifHig"])
    # zero=False: the default filter (Coef >= 1e-6) also drops a negative intercept
    wt = s.weights_table(zero=False)
    wt = wt[(wt["Group"] == "Intercept") | (wt["Coef"] > 1e-6)]
    ours = {w["id"]: w["coef"] for w in jk["weights"]}
    theirs = dict(zip(wt["Group"], wt["Coef"]))
    assert theirs.pop("Intercept") == pytest.approx(jk["fit"]["intercept"])
    assert ours.keys() == theirs.keys()
    for k in ours:
        assert ours[k] == pytest.approx(theirs[k])

    ro = A.run(panel, {**MEMPHIS, "interval": "rolling", "min_train": 36})
    np.testing.assert_allclose(ro["pred_lo"], eff["RollLow"])
    np.testing.assert_allclose(ro["pred_hi"], eff["RollHig"])
    np.testing.assert_allclose(ro["cum_lo"], eff["RollCumLow"])
    np.testing.assert_allclose(ro["cum_hi"], eff["RollCumHig"])


# ---------------------------------------------------------------------------
# Placebo intervals and the block-band guard


def _big():
    """A small placebo pool keeps the cross-validated placebo tests quick:
    the five largest Memphis donors, whichever cities are in the sample."""
    _, _, _, donors = memphis(PANEL)
    return sorted((PANEL.pop[i] for i in donors), reverse=True)[4]


BIG = _big()


def test_placebo_split_matches_single_run(panel):
    """Splitting placebos across workers (as the page does) gives the same result."""
    cfg = {**MEMPHIS, "interval": "placebo", "placebo_min_pop": BIG}
    fresh = A.Panel(DATA.read_text())
    need = A.run(fresh, {**cfg, "defer_placebos": True})["need_placebos"]
    assert 2 <= len(need) < 20
    gaps = {}
    for chunk in (need[:2], need[2:]):
        gaps.update(A.placebo_gaps(A.Panel(DATA.read_text()), cfg, chunk))
    split = A.run(fresh, {**cfg, "placebo_gaps": gaps})
    whole = A.run(A.Panel(DATA.read_text()), cfg)
    assert split["placebo"] == whole["placebo"]
    assert split["placebo"]["n"] == len(need)
    np.testing.assert_allclose(np.array(split["dif_lo"], dtype=float), np.array(whole["dif_lo"], dtype=float))
    assert "placebo_gaps" not in split["config"]


def test_placebo_fixed_alpha_and_synth(panel):
    for extra in [{"penalty": "fixed", "alpha": 10}, {"estimator": "synth"}]:
        out = A.run(panel, {**MEMPHIS, **extra, "interval": "placebo"})
        n = out["placebo"]["n"]
        assert n > 19 and out["n_ref"][0] == n
        assert 1 / (n + 1) <= out["placebo"]["p_cum"] <= 1
        assert out["cum_lo"][0] is not None and out["dif_lo"][0] is not None


def test_placebo_errors(panel):
    with pytest.raises(A.AnalysisError, match="placebo"):
        A.run(panel, {**MEMPHIS, "interval": "placebo", "placebo_min_pop": 1e9})


def test_block_band_guard(panel):
    out = A.run(panel, {**LA, "interval": "block"})
    T0, H = out["T0"], out["H"]
    ok = [T0 - h + 1 >= 19 for h in range(1, H + 1)]  # 19 blocks for a 95% band
    assert [v is not None for v in out["cum_lo"]] == ok
    assert out["n_ref"][0] == T0 and out["n_ref"][-1] == T0 - H + 1


def test_placebo_matches_lassosynth(panel, synthpower):
    lassosynth, _ = synthpower
    import pandas as pd

    y, X, T0, donors = memphis(panel)
    wide = pd.DataFrame(X, columns=[panel.ids[i] for i in donors])
    wide[TREATED] = y
    pool = [panel.ids[i] for i in donors if panel.pop[i] >= BIG]
    s = lassosynth.Synth(wide, TREATED, post=T0)
    s.suggest_alpha()
    s.fit()
    p = s.placebos(pool=pool)
    eff = s.effects(alpha=0.2)
    out = A.run(panel, {**MEMPHIS, "interval": "placebo", "placebo_min_pop": BIG, "level": 0.8})
    assert out["placebo"]["n"] == len(pool)
    assert out["placebo"]["p_cum"] == pytest.approx(p["cum"])
    assert out["placebo"]["p_ratio"] == pytest.approx(p["ratio"])
    np.testing.assert_allclose(out["cum_lo"], eff["PlaceboCumLow"])
    np.testing.assert_allclose(out["cum_hi"], eff["PlaceboCumHig"])
