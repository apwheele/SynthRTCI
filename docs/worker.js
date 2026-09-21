// Runs the Python analysis (docs/py/) in Pyodide, off the page's main thread.
// A module worker: cross-origin importScripts() is blocked in some browsers.
// The page may start several of these to compute placebos in parallel.
//
// Messages in:  {type: "init", data: <rtci.json text>, version}
//               {type: "run", rid, config}
//               {type: "placebos", rid, config, units}
// Messages out: {type: "status", msg}, {type: "ready", runtime}, {type: "fatal", msg}
//               {type: "progress", rid, msg}, {type: "reply", rid, json}, {type: "error", rid, msg}

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const PY_DIR = "/synthrtci";
const PY_FILES = ["synth_methods.py", "analysis.py"];

let py = null;

async function init(dataText, version) {
  postMessage({ type: "status", msg: "Starting Python (Pyodide, about 30 MB the first time, then cached)" });
  py = await loadPyodide();
  postMessage({ type: "status", msg: "Loading NumPy, SciPy and scikit-learn" });
  await py.loadPackage(["numpy", "scipy", "scikit-learn"]);
  postMessage({ type: "status", msg: "Loading the analysis code and data" });
  py.FS.mkdirTree(PY_DIR);
  for (const f of PY_FILES) {
    const r = await fetch(`py/${f}?v=${encodeURIComponent(version)}`);
    if (!r.ok) throw new Error(`Could not load py/${f} (${r.status})`);
    py.FS.writeFile(`${PY_DIR}/${f}`, await r.text());
  }
  py.globals.set("DATA_TEXT", dataText);
  const runtime = py.runPython(`
import sys
sys.path.insert(0, "${PY_DIR}")
import numpy, scipy, sklearn
import analysis
PANEL = analysis.Panel(DATA_TEXT)
del DATA_TEXT
f"Python {sys.version.split()[0]}, NumPy {numpy.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}"
`);
  postMessage({ type: "ready", runtime });
}

function call(rid, code, vars) {
  py.globals.set("PROGRESS", (msg) => postMessage({ type: "progress", rid, msg: String(msg) }));
  for (const [k, v] of Object.entries(vars)) py.globals.set(k, v);
  try {
    postMessage({ type: "reply", rid, json: py.runPython(code) });
  } catch (err) {
    // Unexpected Python errors: send the last line of the traceback
    const lines = String(err.message || err).trim().split("\n");
    postMessage({ type: "error", rid, msg: lines[lines.length - 1] });
  }
}

onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    try {
      await init(m.data, m.version);
    } catch (err) {
      postMessage({ type: "fatal", msg: String(err.message || err) });
    }
  } else if (m.type === "run") {
    call(m.rid, "analysis.run_json(PANEL, CFG, PROGRESS)", { CFG: JSON.stringify(m.config) });
  } else if (m.type === "placebos") {
    call(m.rid, "analysis.placebo_gaps_json(PANEL, CFG, UNITS, PROGRESS)",
      { CFG: JSON.stringify(m.config), UNITS: JSON.stringify(m.units) });
  }
};
