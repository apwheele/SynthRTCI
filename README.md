# SynthRTCI

A website that runs a synthetic control analysis for a city in the
[Real-Time Crime Index](https://realtimecrimeindex.com/) (RTCI): pick a city,
a crime, and the date an intervention started, and it estimates how the city's
monthly crime rate changed compared to a synthetic control built from the
other cities. Everything runs in the browser, in Python via
[Pyodide](https://pyodide.org/), so the site is static files and can be hosted
on GitHub Pages.

The estimators, conformal intervals and placebo tests are from
[SynthPower](https://github.com/apwheele/SynthPower), which tests how well
these intervals work; see that repository for the methods and evidence.
`docs/py/synth_methods.py` is a copy of SynthPower's `methods.py` (without the
block permutation test, plus an option to drop the lasso intercept), and the
tests check that the site gives the same numbers as SynthPower's `LassoSynth`
class, including its placebo tests.

## What the site does

- **Treated city and outcome**: any of the 590 agencies in the RTCI national
  sample; violent crime, murder, rape, robbery, aggravated assault, property
  crime, burglary, theft, or motor vehicle theft. Models are fit to monthly
  rates per 100,000 residents.
- **Intervention**: a start date, and whether the month it starts in counts
  as pre-period, post-period, or is left out. The analysis window can be
  shortened.
- **Estimator**: lasso with an intercept and non-negative coefficients
  (default), lasso without an intercept, or classic synth (weights that sum
  to one). The lasso penalty (alpha) is chosen by 5-fold cross-validation
  unless you set it by hand.
- **Intervals**: rolling-origin forecast errors within the pre-period
  (default; the conformal approach that kept its coverage in SynthPower), the
  jackknife with independent cumulative draws (Wheeler 2023), the jackknife
  with block sums, or placebo tests. 80% to 99% levels. Placebo tests fit
  every donor city above a population cutoff (default 250,000) as if it were
  treated; with the cross-validated lasso each placebo gets its own penalty,
  so the page spreads them over several Python workers (about a minute for
  the Memphis example's 78 placebos on a 4-core machine). The page says how
  many post-period months get a band before you run: rolling-origin and
  block-sum bands need enough pre-period months for the post-period length,
  and placebo bands need at least 19 placebos for 95%.
- **Donor pool**: every other city with complete data in the window, less a
  population cutoff and any cities you leave out.
- **Output**: observed vs. synthetic, monthly and cumulative effects with
  intervals, trajectories of every donor city, the coefficient table, and a
  monthly table, with CSV downloads. **Print report** prints all of it with
  the settings, the RTCI data version (file, Git revision, download date) and
  software versions, since a later data update can change the results.
  **Copy link** makes a link that fills in the same settings.

Python starts loading when the page opens, but nothing runs until you press
Run analysis.

The page starts empty. Two buttons fill in every setting for an example
(the settings are in `docs/examples.json`):

- **Memphis task force**: Memphis violent crime, intervention starting
  September 29, 2025 (so October 2025 is the first post-period month), with
  the places that had 2025 National Guard deployments (Washington, DC; Los
  Angeles and the LA County Sheriff; Chicago; Portland, OR; New Orleans) left
  out of the donor pool.
- **Los Angeles, Gascon**: Los Angeles thefts after George Gascon became
  district attorney (December 2020 is the first post-period month) through
  November 2024, leaving out San Francisco, Chicago, Philadelphia, New York
  City and the LA County Sheriff, as in SynthPower. It uses the jackknife
  intervals, since 47 pre-period months are too few for rolling-origin bands
  over 48 post-period months. The estimate, +895 thefts per 100,000 (+16.4%),
  is close to SynthPower's +911 (+16.7%) but not identical: the site uses the
  population in the RTCI crime file (as CrimeDecomp does), and keeps two
  agencies (Hoover, AL and Pontiac, MI) that SynthPower dropped for lacking
  names in the RTCI agency file.

## Running it locally

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/serve.py      # serves docs/ at http://localhost:8000/ and opens it
```

The first load downloads Pyodide with NumPy, SciPy and scikit-learn (about
30 MB, then cached by the browser). The site needs to be served over HTTP;
opening `docs/index.html` as a file will not work.

## Updating the data

```bash
uv run python scripts/build_data.py
```

This follows the update approach in
[CrimeDecomp](https://github.com/apwheele/CrimeDecomp)
(`src/sync_latest_data.R`): it asks the GitHub API for the current
`Crime_Index_Reported_Crime_Trends_*.csv` in
[AH-Datalytics/rtci](https://github.com/AH-Datalytics/rtci), downloads it and
the agency file used for city names only when their Git revisions have
changed, and rebuilds `docs/data/rtci.json`. The revisions are recorded in
`data/source_metadata.json`. Raw downloads go to `data/raw/` (not committed).
Use `--offline` to rebuild from the raw files without checking upstream.

## Tests

```bash
uv run pytest
```

The tests that compare against SynthPower need its source, found next to this
repository (`../SynthPower/src`) or through the `SYNTHPOWER_SRC` environment
variable; they are skipped otherwise. Python package versions are pinned to
the ones Pyodide 314.0.7 ships, so native and browser results match.

## Deploying to GitHub Pages

The site is the static `docs/` folder. In the repository settings, under
Pages, deploy from the `main` branch, `/docs` folder. (Pages on a private
repository needs a paid GitHub plan.)

## Layout

- `docs/` -- the website: `index.html`, `styles.css`, `app.js` (controls,
  charts, tables), `examples.json` (the example settings), `worker.js` (runs
  Python in a web worker),
  `py/analysis.py` (one analysis, the code the site runs),
  `py/synth_methods.py` (from SynthPower), and `data/rtci.json`.
- `scripts/build_data.py` -- refresh the RTCI snapshot and build the data file.
- `scripts/serve.py` -- local web server.
- `tests/` -- pytest tests of `docs/py/`, run natively.

## License

MIT, see `LICENSE`. The crime data are from the Real-Time Crime Index by
AH Datalytics.

The code was written with Claude Code (Anthropic) under the direction of
Andrew P. Wheeler.
