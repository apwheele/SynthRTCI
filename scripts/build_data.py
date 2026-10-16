"""Refresh the Real-Time Crime Index snapshot and build the website data file.

Follows the update approach in https://github.com/apwheele/CrimeDecomp
(src/sync_latest_data.R): find the newest monthly crime sample linked from
https://crimeindex.org/download (the RTCI moved there from GitHub after the
June 2026 sample), and download it only when its S3 ETag has changed. The
agency file used for city names still comes from the main branch of
https://github.com/AH-Datalytics/rtci, checked by its Git blob revision.

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
USER_AGENT = "SynthRTCI data updater"
BASE_CRIMES = ["murder", "rape", "robbery", "assault", "burglary", "theft", "motor"]

# Two agencies in the crime file have no row in pre_processed.csv; same
# overrides as CrimeDecomp's src/data/raw/agency_coordinate_overrides.csv
NAME_OVERRIDES = {
    "AL0011200": ("Hoover", "AL", "City"),
    "MI6367300": ("Pontiac", "MI", "City"),
}


def github_contents(path):
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={REF}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    req = Request(url, headers=headers)
    with urlopen(req) as r:
        f = json.load(r)
    return {"name": f["name"], "path": f["path"], "revision": f["sha"], "download_url": f["download_url"]}


def crime_index_latest():
    """Newest monthly crime sample linked from the Crime Index download page.

    Its revision is the S3 ETag, which changes whenever the file is replaced.
    """
    with urlopen(Request(CRIME_INDEX_PAGE, headers={"User-Agent": USER_AGENT})) as r:
        html = r.read().decode("utf-8", errors="replace")
    links = {m.group(0): m.group(1) for m in CRIME_SAMPLE_LINK.finditer(html)}
    if not links:
        sys.exit(f"No monthly crime sample links found on {CRIME_INDEX_PAGE}")
    month = max(links.values())
    latest = [u for u, m in links.items() if m == month]
    if len(latest) != 1:
        sys.exit(f"Expected one crime sample for {month} on {CRIME_INDEX_PAGE}, found {len(latest)}")
    url = latest[0]
    with urlopen(Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})) as r:
        etag = r.headers.get("ETag")
    if not etag:
        sys.exit(f"No ETag returned for {url}")
    return {"name": url.rsplit("/", 1)[1], "path": url.split("/", 3)[3],
            "revision": etag.strip('"'), "download_url": url}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(remote, dest, prior):
    """Download ``remote`` to ``dest`` unless the upstream revision is unchanged.

    Changed means different content from the file recorded in ``prior``, so a
    fresh checkout (no data/raw) that downloads the same file is unchanged.
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
    remotes = {"crime_data": crime_index_latest(),
               "agency_source": github_contents("data/deprecated/pre_processed.csv")}
    dests = {"crime_data": RAW / "rtci_crime_trends.csv", "agency_source": RAW / "rtci_pre_processed.csv"}
    out, any_changed = {}, False
    for k in remotes:
        out[k], changed = sync(remotes[k], dests[k], files.get(k))
        any_changed |= changed
    if any_changed or not prior:
        meta = {
            "sources": {"crime_data": CRIME_INDEX_PAGE, "agency_source": f"https://github.com/{REPO}"},
            "source_ref": REF,
            "downloaded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "files": out,
        }
        META.write_text(json.dumps(meta, indent=2) + "\n")
    print("RTCI data changed." if any_changed else "RTCI data unchanged.")


def agency_names():
    pp = pd.read_csv(RAW / "rtci_pre_processed.csv", low_memory=False,
                     usecols=["ori.x", "Agency Name", "state_abbr", "Agency_Type"])
    pp = pp.dropna(subset=["ori.x"])
    # Prefer the City row when an ORI appears as both, as CrimeDecomp does
    pp["priority"] = (pp["Agency_Type"] != "City").astype(int)
    pp = pp.sort_values(["ori.x", "priority"]).drop_duplicates("ori.x")
    names = {r["ori.x"]: (r["Agency Name"], r["state_abbr"], r["Agency_Type"]) for _, r in pp.iterrows()}
    names.update(NAME_OVERRIDES)
    return names


def build():
    cr = pd.read_csv(RAW / "rtci_crime_trends.csv", low_memory=False)
    cr = cr[(cr["size"] == "all") & (cr["sample"] == 1) & (cr["population"] > 0)].copy()
    cr["date"] = cr["year"].astype(int).astype(str) + "-" + cr["month"].astype(int).map("{:02d}".format)
    dates = pd.date_range(f"{cr['date'].min()}-01", f"{cr['date'].max()}-01", freq="MS").strftime("%Y-%m").tolist()

    names = agency_names()
    pops = cr.groupby("id")["population"].agg(["min", "max"])
    if (pops["min"] != pops["max"]).any():
        print("Note: population varies within some agencies; using the latest value.")
    pop = cr.sort_values("date").groupby("id")["population"].last()

    cities = []
    for cid, p in pop.items():
        name, state, typ = names.get(cid, (f"Unknown agency ({cid})", "", "City"))
        label = f"{name}{' County' if typ == 'County' else ''}, {state}"
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

    meta = json.loads(META.read_text()) if META.exists() else {}
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(cities)} agencies, {dates[0]} to {dates[-1]}, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true", help="skip the upstream check, rebuild from data/raw")
    args = ap.parse_args()
    if not args.offline:
        refresh()
    build()
