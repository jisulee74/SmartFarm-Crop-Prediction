"""
server.py - Local Web Server & API for Exhaustive Flower & Fruit Set Prediction Dashboard
Serves static dashboard files and provides REST API endpoints for 127 combinations experiment results.
"""

import http.server
import socketserver
import os
import sys
import json
import urllib.parse
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Import sync_exhaustive_experiments helpers
try:
    import sync_exhaustive_experiments as sync_exp
except ImportError:
    try:
        from . import sync_exhaustive_experiments as sync_exp
    except Exception:
        sync_exp = None


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Connection', 'close')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path.startswith("/api/"):
            return self.handle_api(path, params)

        # Fallback to standard static file serving
        return super().do_GET()

    def send_json(self, data, status_code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_api(self, path, params):
        try:
            # Endpoint: /api/experiments/catalog
            if path == "/api/experiments/catalog":
                catalog_path = os.path.join(DIRECTORY, "cache", "combination_catalog.json")
                if os.path.exists(catalog_path):
                    with open(catalog_path, "r", encoding="utf-8") as f:
                        return self.send_json(json.load(f))
                if sync_exp:
                    return self.send_json(sync_exp.load_catalog())
                return self.send_json({"error": "Catalog not found"}, 404)

            # Endpoint: /api/experiments/metadata
            if path == "/api/experiments/metadata":
                if not sync_exp:
                    return self.send_json({"error": "sync_exhaustive_experiments module not available"}, 500)
                index = sync_exp.load_index()
                targets_meta = {}
                for t_id, t_data in index.get("targets", {}).items():
                    targets_meta[t_id] = {
                        "target_id": t_id,
                        "crop": t_data.get("crop"),
                        "crop_kr": t_data.get("crop_kr"),
                        "target_kr": t_data.get("target_kr"),
                        "part": t_data.get("part"),
                        "status": t_data.get("status"),
                        "group_count": len(t_data.get("groups", {})),
                        "frozen_winner": t_data.get("frozen", {}).get("overall"),
                        "frozen": t_data.get("frozen", {}),
                        "groups": t_data.get("groups", {})
                    }
                return self.send_json({
                    "campaign": index.get("campaign", {}),
                    "targets": targets_meta,
                    "models": sync_exp.MODELS,
                    "seeds": sync_exp.SEEDS
                })

            # Endpoint: /api/experiments/comparison?target=...&model=...&split=...
            elif path == "/api/experiments/comparison":
                if not sync_exp:
                    return self.send_json({"error": "sync_exhaustive_experiments module not available"}, 500)
                target = params.get("target", ["tomato_first"])[0]
                model = params.get("model", ["tabm"])[0]
                split = params.get("split", ["validation"])[0]

                res = sync_exp.get_comparison_data(target, model, split)
                return self.send_json(res)

            # Endpoint: /api/experiments/predictions?target=...&group=...&model=...&split=...&seed=...
            elif path == "/api/experiments/predictions":
                if not sync_exp:
                    return self.send_json({"error": "sync_exhaustive_experiments module not available"}, 500)
                target = params.get("target", ["tomato_first"])[0]
                group = params.get("group", [""])[0]
                model = params.get("model", ["tabm"])[0]
                split = params.get("split", ["validation"])[0]
                seed = params.get("seed", ["all"])[0]

                res = sync_exp.get_predictions_data(target, group, model, split, seed)
                return self.send_json(res)

            # Endpoint: /api/experiments/refresh
            elif path == "/api/experiments/refresh":
                if not sync_exp:
                    return self.send_json({"error": "sync_exhaustive_experiments module not available"}, 500)
                sync_exp.build_index()
                return self.send_json({"status": "refreshed"})

            # Endpoint: /api/analysis/bundle
            elif path == "/api/analysis/bundle":
                bundle_path = os.path.join(DIRECTORY, "ax_direction", "dashboard_bundle.json")
                if not os.path.exists(bundle_path):
                    return self.send_json({"error": "dashboard_bundle.json not found"}, 404)
                with open(bundle_path, "r", encoding="utf-8") as bf:
                    data = json.load(bf)
                return self.send_json(data)

            else:
                return self.send_json({"error": "Unknown API endpoint"}, 404)

        except Exception as e:
            return self.send_json({"error": str(e)}, 500)


class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(port=PORT):
    for p in range(port, port + 50):
        try:
            with ThreadingServer(("", p), DashboardRequestHandler) as httpd:
                url = f"http://localhost:{p}"
                print("=======================================================")
                print(f"[Exhaustive Dashboard Server] Started successfully")
                print(f"URL: {url}")
                print(f"Directory: {DIRECTORY}")
                print("Endpoints:")
                print(f"  - Catalog:     {url}/api/experiments/catalog")
                print(f"  - Metadata:    {url}/api/experiments/metadata")
                print(f"  - Comparison:  {url}/api/experiments/comparison?target=tomato_first&model=tabm")
                print(f"  - Predictions: {url}/api/experiments/predictions?target=tomato_first&group=COMBO_b5b76a4693c1&model=tabm&split=validation&seed=all")
                print("Press Ctrl+C to terminate.")
                print("=======================================================", flush=True)
                httpd.serve_forever()
                break
        except OSError:
            continue


if __name__ == "__main__":
    run_server()
