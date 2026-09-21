// Runs the Python analysis (docs/py/) in Pyodide, off the page's main thread.
// A module worker: cross-origin importScripts() is blocked in some browsers.
//
// Messages in:  {type: "init", data: <rtci.json text>, version}
//               {type: "run", id, config}
// Messages out: {type: "status", msg}, {type: "ready", runtime}, {type: "fatal", msg}
//               {type: "progress", id, msg}, {type: "result", id, json}, {type: "error", id, msg}

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

function run(id, config) {
  const progress = (msg) => postMessage({ type: "progress", id, msg: String(msg) });
  py.globals.set("CFG", JSON.stringify(config));
  py.globals.set("PROGRESS", progress);
  try {
    const json = py.runPython("analysis.run_json(PANEL, CFG, PROGRESS)");
    postMessage({ type: "result", id, json });
  } catch (err) {
    // Unexpected Python errors: send the last line of the traceback
    const lines = String(err.message || err).trim().split("\n");
    postMessage({ type: "error", id, msg: lines[lines.length - 1] });
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
    run(m.id, m.config);
  }
};
