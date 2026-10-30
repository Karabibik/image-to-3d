"""Model worker lifecycle and the generate -> refine -> print-prep job chain."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx
import yaml
from PIL import Image

from .imageprep import prepare_input
from .multiview import align_views
from .printprep import PrintReport, PrintSettings, prepare

REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("models.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class WorkerError(RuntimeError):
    pass


class WorkerManager:
    """Runs at most one model worker; switching models stops the previous one so its VRAM is freed."""

    def __init__(self, workers: dict, log_dir: Path, mock: bool = False, startup_timeout: float = 900):
        self.workers = workers
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mock = mock
        self.startup_timeout = startup_timeout
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self.current: str | None = None

    def available(self, name: str) -> bool:
        return self.mock or Path(self.workers[name]["python"]).exists()

    def log_path(self, name: str) -> Path:
        return self.log_dir / f"{name}.log"

    def tail(self, name: str | None, max_bytes: int = 6000) -> str:
        if not name or not self.log_path(name).exists():
            return ""
        with open(self.log_path(name), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode("utf-8", errors="replace")

    def _url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.workers[name]['port']}"

    def _command(self, name: str) -> list[str]:
        spec = self.workers[name]
        port = ["--port", str(spec["port"])]
        if self.mock:
            return [sys.executable, str(REPO_DIR / "workers" / "mock_worker.py"), *port]
        return [spec["python"], str(REPO_DIR / spec["script"]), *port]

    def _start(self, name: str) -> None:
        self.stop()
        spec = self.workers[name]
        if not self.available(name):
            raise WorkerError(f"{name} is not installed in this image ({spec['python']} not found)")
        env = os.environ.copy()
        env["MODEL_ROOT"] = spec.get("root", "")
        env.update({k: str(v) for k, v in spec.get("env", {}).items()})
        with open(self.log_path(name), "a", encoding="utf-8") as log_file:
            log_file.write(f"\n===== starting {name} at {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
            log_file.flush()
            self._proc = subprocess.Popen(
                self._command(name), stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=REPO_DIR
            )
        self.current = name
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                code = self._proc.returncode
                self._proc, self.current = None, None
                raise WorkerError(f"{name} worker exited with code {code} during startup:\n{self.tail(name)}")
            try:
                if httpx.get(self._url(name) + "/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        self.stop()
        raise WorkerError(f"{name} worker was not ready after {self.startup_timeout:.0f}s:\n{self.tail(name)}")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc, self.current = None, None

    def run(self, name: str, job: dict) -> dict:
        with self._lock:
            if self.current != name or self._proc is None or self._proc.poll() is not None:
                self._start(name)
            try:
                response = httpx.post(self._url(name) + "/run", json=job, timeout=None)
                data = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                raise WorkerError(f"{name} worker failed: {exc}\n{self.tail(name)}") from exc
            if not data.get("ok"):
                hint = ""
                if "out of memory" in str(data.get("error", "")).lower():
                    hint = "\nOut of GPU memory: lower the resolution, enable a low-VRAM option, or use a bigger GPU."
                raise WorkerError(f"{name}: {data.get('error')}{hint}")
            return data


@dataclass
class Job:
    image: Path
    generator: str
    refiner: str | None
    seed: int
    gen_params: dict
    ref_params: dict
    print_settings: PrintSettings
    rembg_model: str
    name: str = ""
    # Turnaround views for multi-view generators ({slot: image}, slots from app.multiview).
    # `image` is then the front view, which refiners and the history use.
    views: dict[str, Path] | None = None


@dataclass
class JobResult:
    run_dir: Path
    report: PrintReport
    record: dict


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:60] or "model"


class Runner:
    def __init__(self, config: dict, outputs_dir: Path, workers: WorkerManager):
        self.config = config
        self.outputs_dir = Path(outputs_dir)
        self.workers = workers

    def _new_run_dir(self, job: Job) -> Path:
        now = datetime.now()
        stages = job.generator + (f"+{job.refiner}" if job.refiner else "")
        base = self.outputs_dir / f"{now:%Y-%m-%d}" / f"{now:%H%M%S}_{job.name}_{stages}_s{job.seed}"
        run_dir, n = base, 1
        while run_dir.exists():  # same image/model/seed twice within a second
            n += 1
            run_dir = base.with_name(f"{base.name}_{n}")
        run_dir.mkdir(parents=True)
        return run_dir

    def run(self, job: Job, status: Callable[[str], None] = lambda _msg: None) -> JobResult:
        job.name = job.name or safe_name(Path(job.image).stem)
        generator = self.config["generators"][job.generator]
        refiner = self.config["refiners"][job.refiner] if job.refiner else None
        multiview = generator.get("input") == "multiview"
        if multiview and not (job.views and "front" in job.views):
            raise ValueError(f"{generator['label']} needs turnaround views including the front")
        if job.views and not multiview:
            raise ValueError(f"{generator['label']} takes a single image, not turnaround views")
        run_dir = self._new_run_dir(job)

        status("Preparing image" + ("s" if job.views else ""))
        image_path, removed = prepare_input(job.image, run_dir / "input.png", job.rembg_model)
        record: dict = {
            "name": job.name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "source_image": str(job.image),
            "background_removed": removed,
            "generator": {"key": job.generator, "params": job.gen_params},
            "refiner": {"key": job.refiner, "params": job.ref_params} if job.refiner else None,
            "seed": job.seed,
        }
        request = {
            "task": "generate", "model": job.generator, "image": str(image_path),
            "out": str(run_dir / "raw.glb"), "seed": job.seed, "params": job.gen_params,
        }
        if job.views:
            request["views"] = record["views"] = self._prepare_views(job, run_dir, image_path)

        status(f"Generating with {generator['label']}")
        result = self.workers.run(generator["worker"], request)
        mesh_path, up = result["mesh"], generator.get("up", "y")
        record["generator"]["info"] = result.get("info", {})

        if refiner:
            status(f"Refining with {refiner['label']}")
            result = self.workers.run(refiner["worker"], {
                "task": "refine", "model": job.refiner, "image": str(image_path), "mesh": mesh_path,
                "out": str(run_dir / "refined.glb"), "seed": job.seed, "params": job.ref_params,
            })
            mesh_path, up = result["mesh"], refiner.get("up", up)
            record["refiner"]["info"] = result.get("info", {})

        status("Preparing for print")
        settings = self._model_print_settings(job.print_settings, job.generator, up, job.gen_params)
        report = prepare(mesh_path, run_dir / job.name, settings)
        record.update(final_mesh=mesh_path, print_settings=asdict(settings), report=report.as_dict())
        (run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return JobResult(run_dir=run_dir, report=report, record=record)

    @staticmethod
    def _prepare_views(job: Job, run_dir: Path, front_cutout: Path) -> dict[str, str]:
        """Cut out every view and frame them alike; returns {slot: path of view_<slot>.png}."""
        cutouts = {}
        for slot, src in job.views.items():
            if slot == "front":  # job.image is the front view, already cut out as input.png
                path = front_cutout
            else:
                path, _ = prepare_input(src, run_dir / f"cutout_{slot}.png", job.rembg_model)
            cutouts[slot] = Image.open(path)
        paths = {}
        for slot, image in align_views(cutouts).items():
            paths[slot] = str(run_dir / f"view_{slot}.png")
            image.save(paths[slot])
        return paths

    def _model_print_settings(self, settings: PrintSettings, generator: str, up: str, params: dict) -> PrintSettings:
        """Add the generator's facing and levelling corrections (refiners keep their input's pose)."""
        spec = self.config["generators"].get(generator, {})
        turn = spec.get("rotate_z", 0)
        # rotate_x runs before rotate_z in print prep; for the generators that set level_param
        # (Pixal3D's frame, turned 180 degrees) +elevation stands the figure back up.
        tilt = float(params.get(spec["level_param"], 0)) if "level_param" in spec else 0.0
        return replace(settings, source_up=up, rotate_x_deg=settings.rotate_x_deg + tilt,
                       rotate_z_deg=(settings.rotate_z_deg + turn) % 360)

    def reprint(self, run_dir: Path, settings: PrintSettings) -> PrintReport:
        """Repeat print prep on a finished run's mesh, e.g. at another height or face budget."""
        record = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
        up = record.get("print_settings", {}).get("source_up", "y")
        settings = self._model_print_settings(settings, record["generator"]["key"], up,
                                              record["generator"].get("params", {}))
        faces = f"{settings.target_faces // 1000}k" if settings.target_faces else "full"
        height = f"{settings.target_height_mm:g}mm" if settings.target_height_mm else "native"
        stem = Path(run_dir) / f"{record['name']}_{height}_{faces}"
        return prepare(record["final_mesh"], stem, settings)

    def history(self, limit: int = 100) -> list[Path]:
        runs = sorted(
            self.outputs_dir.glob("*/*/run.json"), key=lambda p: (p.parent.parent.name, p.parent.name), reverse=True
        )
        return [p.parent for p in runs[:limit]]
