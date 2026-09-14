"""
server.py - Local Web Server & API for Flower & Fruit Set Prediction Dashboard
Serves static dashboard files and provides REST API endpoints for experiment results.
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

# Import sync_experiments helpers
try:
    import sync_experiments
except ImportError:
    sync_experiments = None

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
            if not sync_experiments:
                return self.send_json({"error": "sync_experiments module not available"}, 500)

            # Endpoint: /api/experiments/metadata
            if path == "/api/experiments/metadata":
                index = sync_experiments.load_index()
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
                    "models": sync_experiments.MODELS,
                    "seeds": sync_experiments.SEEDS
                })

            # Endpoint: /api/experiments/comparison?target=...&model=...&split=...
            elif path == "/api/experiments/comparison":
                target = params.get("target", ["tomato_first"])[0]
                model = params.get("model", ["tabm"])[0]
                split = params.get("split", ["validation"])[0]

                index = sync_experiments.load_index()
                t_data = index.get("targets", {}).get(target)
                if not t_data:
                    return self.send_json({"error": f"Target not found: {target}"}, 404)

                groups_dict = t_data.get("groups", {})
                results_dict = t_data.get("results", {})
                frozen_winners = t_data.get("frozen", {}).get("winners", {})
                winner_group = frozen_winners.get(model, {}).get("group")

                rows = []
                for gid, ginfo in groups_dict.items():
                    key = f"{gid}::{model}::{split}"
                    res_item = results_dict.get(key)
                    if not res_item:
                        rows.append({
                            "group_id": gid,
                            "old_group_id": ginfo.get("old_group_id", ""),
                            "readable_name": ginfo.get("readable_name", ""),
                            "feature_count": ginfo.get("feature_count", 0),
                            "category_names": ginfo.get("category_names", []),
                            "category_ids": ginfo.get("category_ids", []),
                            "completed_seeds": 0,
                            "status": "unstarted",
                            "is_winner": (gid == winner_group),
                            "params": {},
                            "bounded": {},
                            "raw": {}
                        })
                    else:
                        rows.append({
                            "group_id": gid,
                            "old_group_id": ginfo.get("old_group_id", ""),
                            "readable_name": ginfo.get("readable_name", ""),
                            "feature_count": ginfo.get("feature_count", 0),
                            "category_names": ginfo.get("category_names", []),
                            "category_ids": ginfo.get("category_ids", []),
                            "completed_seeds": res_item.get("completed_seeds", 0),
                            "status": res_item.get("status", "unstarted"),
                            "is_winner": (gid == winner_group),
                            "params": res_item.get("params", {}),
                            "upper_bound": res_item.get("upper_bound"),
                            "output_clip_rate": res_item.get("output_clip_rate"),
                            "best_epoch": res_item.get("best_epoch"),
                            "n_eval": res_item.get("n_eval"),
                            "bounded": res_item.get("bounded", {}),
                            "raw": res_item.get("raw", {})
                        })

                # Sort by validation bounded RMSE mean (asc), MAE (asc), feature_count (asc), group_id (asc)
                def sort_key(item):
                    b_rmse = item.get("bounded", {}).get("rmse_mean")
                    b_mae = item.get("bounded", {}).get("mae_mean")
                    val_rmse = b_rmse if b_rmse is not None else 999999.0
                    val_mae = b_mae if b_mae is not None else 999999.0
                    return (val_rmse, val_mae, item.get("feature_count", 999), item.get("group_id", ""))

                rows.sort(key=sort_key)
                for idx, r in enumerate(rows):
                    r["rank"] = idx + 1

                provisional_winner = rows[0]["group_id"] if rows and rows[0].get("bounded", {}).get("rmse_mean") is not None else None

                return self.send_json({
                    "target": target,
                    "model": model,
                    "split": split,
                    "target_status": t_data.get("status"),
                    "frozen_winner": winner_group,
                    "provisional_winner": provisional_winner,
                    "total_candidates": len(rows),
                    "rows": rows
                })

            # Endpoint: /api/experiments/predictions?target=...&group=...&model=...&split=...&seed=...
            elif path == "/api/experiments/predictions":
                target = params.get("target", ["tomato_first"])[0]
                group = params.get("group", [""])[0]
                model = params.get("model", ["tabm"])[0]
                split = params.get("split", ["validation"])[0]
                seed = params.get("seed", ["all"])[0]

                res = sync_experiments.get_prediction_data(target, group, model, split, seed)
                return self.send_json(res)

            # Endpoint: /api/experiments/refresh
            elif path == "/api/experiments/refresh":
                sync_experiments.sync_index()
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
                print(f"[Dashboard Server] Started successfully with API support")
                print(f"URL: {url}")
                print(f"Directory: {DIRECTORY}")
                print("Endpoints:")
                print(f"  - Metadata:    {url}/api/experiments/metadata")
                print(f"  - Comparison:  {url}/api/experiments/comparison?target=tomato_first&model=tabm")
                print(f"  - Predictions: {url}/api/experiments/predictions?target=tomato_first&group=VG_Flower_first_d0edfb3a_V2&model=tabm&split=validation&seed=all")
                print("Press Ctrl+C to terminate.")
                print("=======================================================", flush=True)
                httpd.serve_forever()
                break
        except OSError:
            continue

if __name__ == "__main__":
    run_server()
