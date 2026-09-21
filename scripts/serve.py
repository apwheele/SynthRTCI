"""Serve the website in docs/ on localhost and open it in the browser.

    uv run python scripts/serve.py            # http://localhost:8000/
    uv run python scripts/serve.py --port 8080 --no-browser

Uses explicit MIME types (the Windows registry can map .js to text/plain,
which browsers refuse for web workers) and turns off caching so edits show
up on reload. Press Ctrl+C to stop.
"""

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".py": "text/plain; charset=utf-8",
        ".css": "text/css",
        ".html": "text/html",
        ".svg": "image/svg+xml",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    ap = argparse.ArgumentParser(description="Serve docs/ on localhost.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    handler = functools.partial(Handler, directory=str(DOCS))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}/"
        print(f"Serving {DOCS} at {url} (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
