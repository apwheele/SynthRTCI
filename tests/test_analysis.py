"""Tests for the analysis code the website runs (docs/py/).

The comparison with SynthPower needs a local copy of
https://github.com/apwheele/SynthPower, found through the SYNTHPOWER_SRC
environment variable or next to this repository (../SynthPower/src);
those tests are skipped without it.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "py"))

import analysis as A  # noqa: E402
import synth_methods as m  # noqa: E402

DATA = ROOT / "docs" / "data" / "rtci.json"
EXAMPLES = json.loads((ROOT / "docs" / "examples.json").read_text())
MEMPHIS = EXAMPLES["memphis"]["config"]
SYNTHPOWER = Path(os.environ.get("SYNTHPOWER_SRC", ROOT.parent / "SynthPower" / "src"))


@pytest.fixture(scope="module")
def panel():
    return A.Panel(DATA.read_text())


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
    tr = panel.index["TNMPD0000"]
    excl = set(MEMPHIS["exclude"])
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
    for ex in EXAMPLES.values():
        for cid in [ex["config"]["city"]] + ex["config"]["exclude"]:
            assert cid in panel.index


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


def test_missing_treated_data(panel):
    R = panel.rates("rape")
    tr = next(i for i in range(len(panel.ids)) if np.isnan(R[i]).any())
    with pytest.raises(A.AnalysisError, match="missing rape data"):
        A.run(panel, {**MEMPHIS, "city": panel.ids[tr], "crime": "rape"})


def test_run_json(panel):
    out = json.loads(A.run_json(panel, json.dumps({**MEMPHIS, "interval": "block"})),
                     parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert out["city"]["label"] == "Memphis, TN"
    assert len(out["obs"]) == len(out["dates"]) == out["T0"] + out["H"]
    err = json.loads(A.run_json(panel, json.dumps({**MEMPHIS, "start": "2030-01-15"})))
    assert "error" in err


# ---------------------------------------------------------------------------
# Results


def test_donor_pool(panel):
    out = A.run(panel, {**MEMPHIS, "interval": "block"})
    _, X, _, donors = memphis(panel)
    assert out["fit"]["n_donors"] == len(donors) == X.shape[1]
    assert "TNMPD0000" not in out["donors"]["ids"]
    assert not set(MEMPHIS["exclude"]) & set(out["donors"]["ids"])
    assert "CA0190000" in MEMPHIS["exclude"]  # LA County Sheriff
    assert sorted(out["dropped"]["excluded"]) == sorted(
        panel.labels[panel.index[c]] for c in MEMPHIS["exclude"] if c != "TNMPD0000")


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
    # A 99% rolling band needs 99+ forecasts per horizon; 61 reach month 9
    assert A.run(panel, {**MEMPHIS, "level": 0.99})["summary"]["cum_lo"] is None
    assert wide["summary"]["cum_hi"] - wide["summary"]["cum_lo"] > narrow["summary"]["cum_hi"] - narrow["summary"]["cum_lo"]
    # Too few rolling origins for a 95% band: 105 - 90 - 9 + 1 = 7 reach month 9
    few = A.run(panel, {**MEMPHIS, "min_train": 90})
    assert few["n_ref"][-1] == 7
    assert few["cum_lo"][-1] is None and few["summary"]["excludes_zero"] is False


def test_la_gascon_example(panel):
    """Close to SynthPower's Los Angeles lasso result (+911 per 100,000, +16.7%).

    Not identical: the site uses the population in the RTCI crime file (as
    CrimeDecomp does) and keeps Hoover, AL and Pontiac, MI, which SynthPower
    dropped for lacking names in the RTCI agency file.
    """
    out = A.run(panel, EXAMPLES["la_gascon"]["config"])
    assert (out["T0"], out["H"], out["first_post"]) == (47, 48, "2020-12")
    assert out["fit"]["n_donors"] == 579
    assert out["summary"]["cum"] == pytest.approx(911, rel=0.03)
    assert out["summary"]["pct"] == pytest.approx(16.7, abs=0.5)
    assert out["summary"]["excludes_zero"]


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
    E = m.rolling_origin_errors(f_ours, X[:T0], y[:T0], 9, 36)
    np.testing.assert_allclose(E, sp.rolling_origin_errors(f_sp, X[:T0], y[:T0], 9, 36))
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
    wide["TNMPD0000"] = y
    s = lassosynth.Synth(wide, "TNMPD0000", post=T0)
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


BIG = 1_500_000  # a small placebo pool keeps the cross-validated placebo tests quick


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
    out = A.run(panel, {**EXAMPLES["la_gascon"]["config"], "interval": "block"})
    T0, H = out["T0"], out["H"]
    ok = [T0 - h + 1 >= 19 for h in range(1, H + 1)]  # 19 blocks for a 95% band
    assert [v is not None for v in out["cum_lo"]] == ok
    assert out["n_ref"][0] == T0 and out["n_ref"][-1] == T0 - H + 1


def test_placebo_matches_lassosynth(panel, synthpower):
    lassosynth, _ = synthpower
    import pandas as pd

    y, X, T0, donors = memphis(panel)
    wide = pd.DataFrame(X, columns=[panel.ids[i] for i in donors])
    wide["TNMPD0000"] = y
    pool = [panel.ids[i] for i in donors if panel.pop[i] >= BIG]
    s = lassosynth.Synth(wide, "TNMPD0000", post=T0)
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
