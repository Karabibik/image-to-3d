"""JSON-over-HTTP server shared by every model worker.

Standard library only, so it runs unchanged inside each model's own venv.

Protocol:
    GET  /health -> {"ok": true, "loaded": [...]}
    POST /run    -> body {"task": "generate"|"refine"|"paint", "model": str, "image": path,
                          "mesh": path (refine, paint), "out": path, "seed": int, "params": {...}}
                    reply {"ok": true, "mesh": path, "info": {...}}
                       or {"ok": false, "error": str, "traceback": str}
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

Handler = Callable[[dict], dict]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def serve(tasks: dict[str, Handler], loaded: Callable[[], list[str]] = lambda: []) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    gpu_lock = threading.Lock()

    class RequestHandler(BaseHTTPRequestHandler):
        def _reply(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._reply(200, {"ok": True, "loaded": loaded()})
            else:
                self._reply(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/run":
                self._reply(404, {"ok": False, "error": "not found"})
                return
            job = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            fn = tasks.get(job.get("task", ""))
            if fn is None:
                self._reply(400, {"ok": False, "error": f"unknown task {job.get('task')!r}"})
                return
            with gpu_lock:
                started = time.time()
                log(f"{job['task']} {job.get('model', '')} started")
                try:
                    result = fn(job)
                except Exception as exc:  # report every failure back to the UI
                    tb = traceback.format_exc()
                    log(tb)
                    self._reply(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": tb})
                    return
                info = result.setdefault("info", {})
                info["seconds"] = round(time.time() - started, 1)
                log(f"{job['task']} finished in {info['seconds']}s")
                self._reply(200, {"ok": True, **result})

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    log(f"worker listening on {args.host}:{args.port} (python {sys.version.split()[0]})")
    server.serve_forever()


def cuda_memory_info() -> dict:
    """Peak VRAM of the last job, for the run report; empty when torch/CUDA is absent."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return {"peak_vram_gb": round(peak, 2), "gpu": torch.cuda.get_device_name()}
