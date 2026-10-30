"""On-pod smoke test: run every pipeline on one image and record time, VRAM and mesh health.

Usage (inside the pod):
    /opt/venvs/ui/bin/python scripts/smoke_test.py /workspace/test.png [--only hunyuan trellis ...]
        [--views /workspace/views]   # front.png, left.png, back.png, right.png (any, front required)

Multi-view pipelines run only with --views.

Writes a Markdown table to /workspace/outputs/smoke_<timestamp>.md and prints it.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.multiview import SLOTS  # noqa: E402
from app.orchestrator import Job, Runner, WorkerManager, load_config  # noqa: E402
from app.printprep import PrintSettings  # noqa: E402

# (generator, refiner, generator params, refiner params): defaults plus the heavy settings.
PIPELINES = [
    ("hunyuan", None, {}, {}),
    ("hunyuan", "ultrashape", {}, {}),
    ("trellis", None, {"resolution": "1024"}, {}),
    ("trellis", None, {"resolution": "1536"}, {}),
    ("pixal3d", None, {"resolution": "1536"}, {}),
    ("hunyuan", "hi3d", {}, {"resolution": "1024"}),
    ("hunyuan", "hi3d", {}, {"resolution": "1536"}),
    ("hunyuan2mv", None, {}, {}),
    ("hunyuan2mv", None, {"variant": "turbo", "steps": 5}, {}),
    ("hunyuan2mv", "ultrashape", {}, {}),
    ("pixal3d_mv", None, {"resolution": "1536"}, {}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--only", nargs="*", help="limit to pipelines that use these model keys")
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace"))
    parser.add_argument("--views", type=Path, help="folder with <slot>.png turnaround views")
    args = parser.parse_args()
    views = {p.stem: p for p in sorted(args.views.glob("*.png")) if p.stem in SLOTS} if args.views else {}

    config = load_config()
    workers = WorkerManager(config["workers"], args.data_dir / "logs")
    runner = Runner(config, args.data_dir / "outputs" / "smoke", workers)
    rows, gpu = [], ""
    try:
        for gen, ref, gen_params, ref_params in PIPELINES:
            if args.only and not ({gen, ref} & set(args.only)):
                continue
            multiview = config["generators"][gen].get("input") == "multiview"
            if multiview and "front" not in views:
                continue
            label = gen + (f" + {ref}" if ref else "") + f" {gen_params or ''}{ref_params or ''}"
            print(f"=== {label}", flush=True)
            started = time.time()
            try:
                result = runner.run(Job(views["front"] if multiview else args.image, gen, ref, 1234,
                                        gen_params, ref_params, PrintSettings(target_faces=None),
                                        "birefnet-general", views=views if multiview else None))
            except Exception as exc:
                rows.append(f"| {label} | failed | | | | `{str(exc).splitlines()[0][:120]}` |")
                continue
            infos = [result.record["generator"].get("info", {})]
            if result.record.get("refiner"):
                infos.append(result.record["refiner"].get("info", {}))
            vram = max((i.get("peak_vram_gb", 0) for i in infos), default=0)
            gpu = gpu or next((i["gpu"] for i in infos if "gpu" in i), "")
            report = result.report
            rows.append(
                f"| {label} | {time.time() - started:.0f}s | {vram:.1f} GB | {report.faces:,} | "
                f"{'yes' if report.watertight else 'no'} | {result.run_dir.name} |"
            )
    finally:
        workers.stop()

    table = "\n".join([
        f"# Smoke test {datetime.now():%Y-%m-%d %H:%M} on {gpu or 'unknown GPU'}",
        f"Image: {args.image}",
        "",
        "| Pipeline | Time (incl. load/download) | Peak VRAM | Faces | Watertight | Run / error |",
        "|---|---|---|---|---|---|",
        *rows,
    ])
    out = args.data_dir / "outputs" / f"smoke_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
