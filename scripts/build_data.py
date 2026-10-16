"""Refresh the Real-Time Crime Index snapshot and build the website data file.

Follows the update approach in https://github.com/apwheele/CrimeDecomp
(src/sync_latest_data.R): find the newest monthly crime sample linked from
https://crimeindex.org/download (the RTCI moved there from GitHub after the
June 2026 sample), and download it only when its S3 ETag has changed. Each
sample carries the full history, revised, so a new one replaces the old; it
is only used if it starts no later and ends no earlier than the current data.

City names come from the RTCI agency file on the main branch of
https://github.com/AH-Datalytics/rtci (no longer updated), with FBI agency
names from https://github.com/AH-Datalytics/cde-population for agencies that
joined the sample since. Both are checked by their Git blob revisions.

    uv run python scripts/build_data.py            # check upstream, rebuild
    uv run python scripts/build_data.py --offline  # rebuild from data/raw only

Writes data/source_metadata.json and docs/data/rtci.json. Raw downloads go
to data/raw/ (not committed, about 50 MB). The output only changes when the
upstream files do, so the GitHub Actions workflow
(.github/workflows/pages.yml) can run this on a schedule and commit and
deploy only real updates. Set GITHUB_TOKEN to avoid GitHub API rate limits.
"""

import os

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "source_metadata.json"
OUT = ROOT / "docs" / "data" / "rtci.json"

CRIME_INDEX_PAGE = "https://crimeindex.org/download"
CRIME_SAMPLE_LINK = re.compile(
    r"https://crimeindex-data\.s3[.a-z0-9-]*amazonaws\.com/downloads/archive/crime_(\d{6})(?:_\d+)?\.csv")
REPO = "AH-Datalytics/rtci"
REF = "main"
NAMES_REPO = "AH-Datalytics/cde-population"
NAMES_REF = "master"
USER_AGENT = "SynthRTCI data updater"
BASE_CRIMES = ["murder", "rape", "robbery", "assault", "burglary", "theft", "motor"]
REQUIRED_COLUMNS = ["id", "size", "year", "month", "population", "sample"] + [f"{c}_total" for c in BASE_CRIMES]
FILES = {
    "crime_data": "rtci_crime_trends.csv",
    "agency_source": "rtci_pre_processed.csv",
    "agency_names": "cde_agency_names.csv",
}

# Two agencies in the crime file have no row in pre_processed.csv; same
# overrides as CrimeDecomp's src/data/raw/agency_coordinate_overrides.csv
NAME_OVERRIDES = {
    "AL0011200": ("Hoover", "AL", "City"),
    "MI6367300": ("Pontiac", "MI", "City"),
}
# FBI agency names to RTCI-style city names, e.g. "Hoover Police Department"
CDE_COUNTY = re.compile(r"^(.*?) County (?:Sheriff'?s? (?:Office|Department)|Police Department|Police)$", re.I)
CDE_CITY = re.compile(r"^(.*?) (?:Police Department|Police Dept\.?|Department of Public Safety|"
                      r"Public Safety Department|Police)$", re.I)


def github_contents(path, repo=REPO, ref=REF):
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    req = Request(url, headers=headers)
    with urlopen(req) as r:
        f = json.load(r)
    return {"name": f["name"], "path": f["path"], "revision": f["sha"], "download_url": f["download_url"]}


def latest_crime_sample(html):
    """URL of the newest monthly crime sample linked in the download page."""
    links = {m.group(0): m.group(1) for m in CRIME_SAMPLE_LINK.finditer(html)}
    if not links:
        raise ValueError(f"No monthly crime sample links found on {CRIME_INDEX_PAGE}")
    month = max(links.values())
    latest = [u for u, m in links.items() if m == month]
    if len(latest) != 1:
        raise ValueError(f"Expected one crime sample for {month} on {CRIME_INDEX_PAGE}, found {len(latest)}")
    return latest[0]


def crime_index_latest():
    """Newest monthly crime sample; its revision is the S3 ETag, which changes
    whenever the file is replaced."""
    with urlopen(Request(CRIME_INDEX_PAGE, headers={"User-Agent": USER_AGENT})) as r:
        html = r.read().decode("utf-8", errors="replace")
    try:
        url = latest_crime_sample(html)
    except ValueError as e:
        sys.exit(str(e))
    with urlopen(Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})) as r:
        etag = r.headers.get("ETag")
    if not etag:
        sys.exit(f"No ETag returned for {url}")
    return {"name": url.rsplit("/", 1)[1], "path": url.split("/", 3)[3],
            "revision": etag.strip('"'), "download_url": url}


def check_history(path, prior_dates=None):
    """First and last month of the sample agencies in a crime file.

    Raises ValueError if the file lacks a column the site needs, has no
    sample agencies, or would drop months: it must start no later and end no
    earlier than ``prior_dates`` (the months of the current site data).
    """
    cols = pd.read_csv(path, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"missing columns {', '.join(missing)}")
    d = pd.read_csv(path, usecols=["size", "sample", "year", "month"], low_memory=False)
    d = d[(d["size"] == "all") & (d["sample"] == 1)]
    if d.empty:
        raise ValueError("no sample agencies (size 'all', sample 1)")
    ym = sorted(set(zip(d["year"].astype(int), d["month"].astype(int))))
    first, last = (f"{y:04d}-{m:02d}" for y, m in (ym[0], ym[-1]))
    if prior_dates:
        if first > prior_dates[0]:
            raise ValueError(f"starts in {first}, after the current data's first month {prior_dates[0]}")
        if last < prior_dates[-1]:
            raise ValueError(f"ends in {last}, before the current data's last month {prior_dates[-1]}")
    return first, last


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(remote, dest, prior, check=None):
    """Download ``remote`` to ``dest`` unless the upstream revision is unchanged.

    ``check(path)`` vets a download before it replaces ``dest``; a ValueError
    stops the update and keeps the current file. Changed means different
    content from the file recorded in ``prior``, so a fresh checkout (no
    data/raw) that downloads the same file is unchanged.
    """
    # blob_sha is the field name used before the crime file moved off GitHub
    prior_rev = prior and prior.get("revision", prior.get("blob_sha"))
    if not (prior_rev == remote["revision"] and dest.exists()):
        print(f"Downloading {remote['download_url']}")
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        req = Request(remote["download_url"], headers={"User-Agent": USER_AGENT})
        with urlopen(req) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if check:
            try:
                check(tmp)
            except ValueError as e:
                tmp.unlink()
                sys.exit(f"Not using {remote['download_url']}: {e}")
        tmp.replace(dest)
    new = md5(dest)
    changed = not prior or prior.get("md5") != new or prior.get("name") != remote["name"]
    print(f"{'Changed' if changed else 'Unchanged'}: {remote['path']}")
    return {"name": remote["name"], "path": remote["path"], "revision": remote["revision"],
            "source_url": remote["download_url"], "md5": new}, changed


def refresh():
    RAW.mkdir(parents=True, exist_ok=True)
    prior = json.loads(META.read_text()) if META.exists() else {}
    files = prior.get("files", {})
    prior_dates = json.loads(OUT.read_text())["dates"] if OUT.exists() else None
    remotes = {"crime_data": crime_index_latest(),
               "agency_source": github_contents("data/deprecated/pre_processed.csv"),
               "agency_names": github_contents("cde_populations.csv", NAMES_REPO, NAMES_REF)}
    checks = {"crime_data": lambda p: check_history(p, prior_dates)}
    out, changed = {}, {}
    for k in remotes:
        out[k], changed[k] = sync(remotes[k], RAW / FILES[k], files.get(k), checks.get(k))
    any_changed = any(changed.values())
    if any_changed or not prior:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        meta = {
            "sources": {"crime_data": CRIME_INDEX_PAGE,
                        "agency_source": f"https://github.com/{REPO}/tree/{REF}",
                        "agency_names": f"https://github.com/{NAMES_REPO}/tree/{NAMES_REF}"},
            # When the crime data (shown on the page) were downloaded
            "downloaded_at_utc": now if changed["crime_data"] else prior.get("downloaded_at_utc", now),
            "files": out,
        }
        META.write_text(json.dumps(meta, indent=2) + "\n")
    print("RTCI data changed." if any_changed else "RTCI data unchanged.")


def clean_cde_name(name):
    """City name and agency type from an FBI agency name."""
    name = " ".join(str(name).split())
    m = CDE_COUNTY.match(name)
    if m:
        return m.group(1), "County"
    m = CDE_CITY.match(name)
    return (m.group(1), "City") if m else (name, "Other")


def agency_names(raw=RAW):
    """Agency ID -> (name, state, type): RTCI's names, then manual overrides,
    falling back to FBI agency names for agencies RTCI's file lacks."""
    names = {}
    cde_path = raw / FILES["agency_names"]
    if cde_path.exists():
        cde = pd.read_csv(cde_path, usecols=["ori", "agency_name", "state"]).dropna()
        for r in cde.itertuples():
            name, typ = clean_cde_name(r.agency_name)
            names[r.ori] = (name, r.state, typ)
    pp = pd.read_csv(raw / FILES["agency_source"], low_memory=False,
                     usecols=["ori.x", "Agency Name", "state_abbr", "Agency_Type"])
    pp = pp.dropna(subset=["ori.x"])
    # Prefer the City row when an ORI appears as both, as CrimeDecomp does
    pp["priority"] = (pp["Agency_Type"] != "City").astype(int)
    pp = pp.sort_values(["ori.x", "priority"]).drop_duplicates("ori.x")
    names.update({r["ori.x"]: (r["Agency Name"], r["state_abbr"], r["Agency_Type"]) for _, r in pp.iterrows()})
    names.update(NAME_OVERRIDES)
    return names


def build(raw=RAW, meta_path=META, out=OUT):
    cr = pd.read_csv(raw / FILES["crime_data"], low_memory=False)
    cr = cr[(cr["size"] == "all") & (cr["sample"] == 1) & (cr["population"] > 0)].copy()
    cr["date"] = cr["year"].astype(int).astype(str) + "-" + cr["month"].astype(int).map("{:02d}".format)
    dates = pd.date_range(f"{cr['date'].min()}-01", f"{cr['date'].max()}-01", freq="MS").strftime("%Y-%m").tolist()

    names = agency_names(raw)
    pops = cr.groupby("id")["population"].agg(["min", "max"])
    if (pops["min"] != pops["max"]).any():
        print("Note: population varies within some agencies; using the latest value.")
    pop = cr.sort_values("date").groupby("id")["population"].last()
    unnamed = [cid for cid in pop.index if cid not in names]
    if unnamed:
        print(f"Note: no name for {len(unnamed)} agencies, shown by ID: {', '.join(unnamed)}")

    cities = []
    for cid, p in pop.items():
        name, state, typ = names.get(cid, (f"Unknown agency ({cid})", "", "City"))
        label = f"{name}{' County' if typ == 'County' else ''}{', ' + state if state else ''}"
        cities.append({"id": cid, "label": label, "name": name, "state": state, "type": typ, "pop": int(p)})
    cities.sort(key=lambda c: (c["label"], c["id"]))
    labels = pd.Series([c["label"] for c in cities])
    for i in np.where(labels.duplicated(keep=False))[0]:
        cities[i]["label"] += f" ({cities[i]['id']})"
    ids = [c["id"] for c in cities]

    counts = {}
    for crime in BASE_CRIMES:
        w = cr.pivot(index="id", columns="date", values=f"{crime}_total").reindex(index=ids, columns=dates)
        popv = pop.reindex(ids).to_numpy()[:, None]
        v = w.to_numpy(dtype=float, copy=True)
        bad = (v < 0) | (v > popv)  # same validity rule as CrimeDecomp
        v[bad] = np.nan
        counts[crime] = [[None if np.isnan(x) else int(x) for x in row] for row in v]

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    crime_file = meta.get("files", {}).get("crime_data", {})
    data = {
        "source": {
            "url": meta.get("sources", {}).get("crime_data", CRIME_INDEX_PAGE),
            "file": crime_file.get("name", ""),
            "revision": crime_file.get("revision", ""),
            "downloaded_at_utc": meta.get("downloaded_at_utc", ""),
        },
        "dates": dates,
        "cities": cities,
        "counts": counts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: {len(cities)} agencies, "
          f"{dates[0]} to {dates[-1]}, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true", help="skip the upstream check, rebuild from data/raw")
    args = ap.parse_args()
    if not args.offline:
        refresh()
    build()
