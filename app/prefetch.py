"""Download model weights ahead of first use.

Usage: python -m app.prefetch hunyuan ultrashape   (or "all")
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

from .orchestrator import load_config


def prefetch(names: list[str]) -> None:
    workers = load_config()["workers"]
    if names == ["all"]:
        names = list(workers)
    for name in names:
        if name not in workers:
            print(f"[prefetch] unknown worker {name!r}; choose from {', '.join(workers)}", flush=True)
            continue
        for item in workers[name].get("prefetch", []):
            print(f"[prefetch] {name}: {item['repo']}", flush=True)
            # local_dir / cache_dir match where that model's own loader looks (env vars expanded).
            where = {k: os.path.expandvars(item[k]) for k in ("local_dir", "cache_dir") if k in item}
            try:
                snapshot_download(item["repo"], allow_patterns=item.get("allow"), ignore_patterns=item.get("ignore"), **where)
            except Exception as exc:  # keep going; the worker retries on first use
                print(f"[prefetch] {item['repo']} failed: {exc}", flush=True)
    print("[prefetch] done", flush=True)


if __name__ == "__main__":
    prefetch(sys.argv[1:] or ["all"])
