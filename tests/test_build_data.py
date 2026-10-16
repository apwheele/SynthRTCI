"""Tests for scripts/build_data.py with small made-up RTCI files.

Every monthly RTCI sample adds a month, revises past months, and changes
which agencies are in the sample (agencies whose latest month is late drop
out; new agencies join). These check that the site data follow along.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "docs" / "py"))

import analysis as A  # noqa: E402
import build_data as B  # noqa: E402

S3 = "https://crimeindex-data.s3.us-east-1.amazonaws.com/downloads"


def months(first, last):
    return [(p.year, p.month) for p in pd.period_range(first, last, freq="M")]


def rows(agency, first, last, sample=1, pop=100_000, size="all", count=10):
    return [{"id": agency, "size": size, "year": y, "month": m, "population": pop, "agencies": 1,
             "sample": sample, **{f"{c}_total": count + (y * 12 + m) % 5 for c in B.BASE_CRIMES}}
            for y, m in months(first, last)]


def write_crime(path, data):
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Finding the newest sample


def test_latest_crime_sample():
    links = ["archive/crime_202606.csv", "archive/crime_202608_1792092221.csv",
             "archive/crime_202607_1789575398.csv", "archive/clearance_202609_1792092877.csv",
             "historical_crime_1930_2025_1790092670.csv", "archive/crime_202608_1792092221.csv"]
    html = " ".join(f'<a href="{S3}/{f}">' for f in links)
    assert B.latest_crime_sample(html) == f"{S3}/archive/crime_202608_1792092221.csv"


def test_latest_crime_sample_errors():
    with pytest.raises(ValueError, match="No monthly crime sample"):
        B.latest_crime_sample("<html></html>")
    with pytest.raises(ValueError, match="one crime sample for 202608"):
        B.latest_crime_sample(f"{S3}/archive/crime_202608_1.csv {S3}/archive/crime_202608_2.csv")


# ---------------------------------------------------------------------------
# A new sample must keep the history


def test_check_history(tmp_path):
    prior = [f"{y:04d}-{m:02d}" for y, m in months("2017-01", "2026-08")]
    good = write_crime(tmp_path / "good.csv", rows("A", "2017-01", "2026-09") + rows("B", "2017-01", "2026-08", sample=0))
    assert B.check_history(good, prior) == ("2017-01", "2026-09")
    assert B.check_history(good) == ("2017-01", "2026-09")
    # Same months again (a revision) is fine
    assert B.check_history(write_crime(tmp_path / "same.csv", rows("A", "2017-01", "2026-08")), prior)


@pytest.mark.parametrize("data,msg", [
    (rows("A", "2017-01", "2026-07"), "ends in 2026-07"),
    (rows("A", "2020-01", "2026-09"), "starts in 2020-01"),
    (rows("A", "2017-01", "2026-09", sample=0), "no sample agencies"),
    ([{k: v for k, v in r.items() if k != "theft_total"} for r in rows("A", "2017-01", "2026-09")], "theft_total"),
])
def test_check_history_rejects(tmp_path, data, msg):
    prior = [f"{y:04d}-{m:02d}" for y, m in months("2017-01", "2026-08")]
    with pytest.raises(ValueError, match=msg):
        B.check_history(write_crime(tmp_path / "bad.csv", data), prior)


# ---------------------------------------------------------------------------
# Agency names


@pytest.mark.parametrize("fbi,name,typ", [
    ("Hoover Police Department", "Hoover", "City"),
    ("Dunwoody  Police Department", "Dunwoody", "City"),
    ("St. Louis County Police Department", "St. Louis", "County"),
    ("Kent County Sheriff's Office", "Kent", "County"),
    ("Jacksonville Sheriff's Office", "Jacksonville Sheriff's Office", "Other"),
])
def test_clean_cde_name(fbi, name, typ):
    assert B.clean_cde_name(fbi) == (name, typ)


# ---------------------------------------------------------------------------
# Building the site data when agencies join or leave the sample


def test_build_when_agencies_change(tmp_path):
    data = (rows("TX0000001", "2017-01", "2026-09", pop=200_000)        # in the sample throughout
            + rows("TX0000002", "2017-01", "2026-08", sample=0)          # left: latest month is late
            + rows("TX0000003", "2025-01", "2026-09")                    # joined; RTCI file lacks it
            + rows("TX0000004", "2024-06", "2026-09")                    # joined; no name anywhere
            + rows("TX0000001", "2017-01", "2026-09", size="lg")         # size groups are not agencies
            + rows("tx", "2017-01", "2026-09", sample=float("nan")))     # nor are state totals
    write_crime(tmp_path / B.FILES["crime_data"], data)
    pd.DataFrame({"ori.x": ["TX0000001", "TX0000002"], "Agency Name": ["Oldtown", "Gone"],
                  "state_abbr": ["TX", "TX"], "Agency_Type": ["City", "City"]}).to_csv(
        tmp_path / B.FILES["agency_source"], index=False)
    pd.DataFrame({"ori": ["TX0000001", "TX0000003"], "agency_name": ["Oldtown City Police", "Newtown Police Department"],
                  "state": ["TX", "TX"], "population": [1, 1]}).to_csv(tmp_path / B.FILES["agency_names"], index=False)
    out = tmp_path / "rtci.json"
    B.build(raw=tmp_path, meta_path=tmp_path / "none.json", out=out)

    d = json.loads(out.read_text())
    assert d["dates"][0] == "2017-01" and d["dates"][-1] == "2026-09"   # the new month is appended
    assert len(d["dates"]) == len(months("2017-01", "2026-09"))
    labels = {c["id"]: c["label"] for c in d["cities"]}
    assert labels == {"TX0000001": "Oldtown, TX",                       # RTCI's name wins over the FBI's
                      "TX0000003": "Newtown, TX",
                      "TX0000004": "Unknown agency (TX0000004)"}
    i = [c["id"] for c in d["cities"]].index("TX0000003")
    joined = d["counts"]["theft"][i]
    assert joined[d["dates"].index("2024-12")] is None and joined[d["dates"].index("2025-01")] is not None

    # The page's analysis code reads it. New agencies are donors only for
    # windows they have data for.
    panel = A.Panel(out.read_text())
    cfg = {"city": "TX0000001", "crime": "theft", "start": "2026-01-01", "estimator": "synth", "interval": "jackknife"}
    with pytest.raises(A.AnalysisError, match="Fewer than two donor"):
        A.run(panel, cfg)
    res = A.run(panel, {**cfg, "first": "2025-01"})
    assert sorted(res["donors"]["ids"]) == ["TX0000003", "TX0000004"]
    assert res["dates"][-1] == "2026-09"
