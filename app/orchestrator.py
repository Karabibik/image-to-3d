"""Model worker lifecycle and the steps of a run: image, shape, refine, color, print."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx
import yaml
from PIL import Image

from .imageprep import prepare_input
from .multiview import align_views
from .printprep import PrintReport, PrintSettings, make_preview, prepare

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
    """Everything one unattended run needs; see Runner.run."""

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
    # `image` is then the front view, which refiners and the painter use.
    views: dict[str, Path] | None = None
    painter: str | None = None
    paint_params: dict = field(default_factory=dict)


@dataclass
class Prepared:
    """Cut-outs from the Image step, ready for any number of Shape runs."""

    image: Path
    name: str
    source: str
    background_removed: bool
    views: dict[str, Path] | None = None


@dataclass
class JobResult:
    run_dir: Path
    report: PrintReport
    record: dict


Status = Callable[[str], None]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:60] or "model"


def _new_dir(base: Path) -> Path:
    folder, n = base, 1
    while folder.exists():  # same image/model/seed twice within a second
        n += 1
        folder = base.with_name(f"{base.name}_{n}")
    folder.mkdir(parents=True)
    return folder


def _no_status(_msg: str) -> None:
    pass


class Runner:
    """One method per step. Each Shape run gets a folder whose run.json is rewritten after every step."""

    def __init__(self, config: dict, outputs_dir: Path, workers: WorkerManager):
        self.config = config
        self.outputs_dir = Path(outputs_dir)
        self.workers = workers

    def load(self, run_dir: str | Path) -> dict:
        return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))

    def _save(self, run_dir: str | Path, record: dict) -> None:
        (Path(run_dir) / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    def prepare_image(self, image: Path, rembg_model: str, views: dict[str, Path] | None = None, name: str = "",
                      status: Status = _no_status) -> Prepared:
        name = name or safe_name(Path(image).stem)
        folder = _new_dir(self.outputs_dir / "prepared" / f"{datetime.now():%Y%m%d_%H%M%S}_{name}")
        status("Preparing image" + ("s" if views else ""))
        cutout, removed = prepare_input(image, folder / "input.png", rembg_model)
        framed = self._prepare_views(views, rembg_model, folder, cutout) if views else None
        return Prepared(cutout, name, str(image), removed, framed)

    @staticmethod
    def _prepare_views(views: dict[str, Path], rembg_model: str, folder: Path, front_cutout: Path) -> dict[str, Path]:
        """Cut out every view and frame them alike; returns {slot: path of view_<slot>.png}."""
        cutouts = {}
        for slot, src in views.items():
            if slot == "front":  # the main image is the front view, already cut out
                path = front_cutout
            else:
                path, _ = prepare_input(src, folder / f"cutout_{slot}.png", rembg_model)
            cutouts[slot] = Image.open(path)
        paths = {}
        for slot, image in align_views(cutouts).items():
            paths[slot] = folder / f"view_{slot}.png"
            image.save(paths[slot])
        return paths

    def prepared_from(self, run_dir: str | Path) -> Prepared:
        """The inputs of an earlier run, to generate from them again."""
        record = self.load(run_dir)
        views = {slot: Path(path) for slot, path in record["views"].items()} if record.get("views") else None
        return Prepared(Path(run_dir) / "input.png", record["name"], record.get("source_image", ""),
                        record.get("background_removed", False), views)

    def generate(self, prepared: Prepared, generator: str, seed: int, params: dict,
                 status: Status = _no_status) -> Path:
        """Shape step. Always a new run, so another seed never replaces an earlier result."""
        spec = self.config["generators"][generator]
        multiview = spec.get("input") == "multiview"
        if multiview and not (prepared.views and "front" in prepared.views):
            raise ValueError(f"{spec['label']} needs turnaround views including the front")
        if prepared.views and not multiview:
            raise ValueError(f"{spec['label']} takes a single image, not turnaround views")

        now = datetime.now()
        run_dir = _new_dir(self.outputs_dir / f"{now:%Y-%m-%d}" / f"{now:%H%M%S}_{prepared.name}_{generator}_s{seed}")
        image = shutil.copy(prepared.image, run_dir / "input.png")
        record: dict = {
            "name": prepared.name,
            "created": now.isoformat(timespec="seconds"),
            "source_image": prepared.source,
            "background_removed": prepared.background_removed,
            "generator": {"key": generator, "params": params},
            "refiner": None,
            "seed": seed,
        }
        request = {
            "task": "generate", "model": generator, "image": str(image),
            "out": str(run_dir / "raw.glb"), "seed": seed, "params": params,
        }
        if prepared.views:
            request["views"] = record["views"] = {
                slot: str(shutil.copy(path, run_dir / f"view_{slot}.png")) for slot, path in prepared.views.items()
            }

        status(f"Generating with {spec['label']}")
        result = self.workers.run(spec["worker"], request)
        record["generator"]["info"] = result.get("info", {})
        record.update(raw_mesh=result["mesh"], final_mesh=result["mesh"], up=spec.get("up", "y"))
        self._save(run_dir, record)
        self._add_preview(run_dir, record, "Shape", result["mesh"], status)
        return run_dir

    def refine(self, run_dir: str | Path, refiner: str, params: dict, status: Status = _no_status) -> dict:
        """Refine step. Always works from the Shape result, so it can be repeated with other settings."""
        run_dir = Path(run_dir)
        record = self.load(run_dir)
        spec = self.config["refiners"][refiner]
        self._clear_results(run_dir, record)
        status(f"Refining with {spec['label']}")
        result = self.workers.run(spec["worker"], {
            "task": "refine", "model": refiner, "image": str(run_dir / "input.png"),
            "mesh": record.get("raw_mesh", str(run_dir / "raw.glb")),
            "out": str(run_dir / "refined.glb"), "seed": record["seed"], "params": params,
        })
        record["refiner"] = {"key": refiner, "params": params, "info": result.get("info", {})}
        record.update(final_mesh=result["mesh"], up=spec.get("up", self._up(record)))
        self._save(run_dir, record)
        self._add_preview(run_dir, record, "Refined", result["mesh"], status)
        return record

    def paint(self, run_dir: str | Path, painter: str, params: dict, status: Status = _no_status) -> dict:
        """Color step. Writes a separate, lighter textured GLB; the print mesh is not touched."""
        run_dir = Path(run_dir)
        record = self.load(run_dir)
        spec = self.config["painters"][painter]
        status("Reducing the mesh for painting")
        mesh_path = run_dir / "paint_input.obj"
        # Normalised like every other preview, so the viewer frames the painted model the same way.
        make_preview(record["final_mesh"], mesh_path, self._up(record), int(params.get("faces", 40000)))
        status(f"Painting with {spec['label']}")
        result = self.workers.run(spec["worker"], {
            "task": "paint", "model": painter, "image": str(run_dir / "input.png"), "mesh": str(mesh_path),
            "out": str(run_dir / "textured.glb"), "seed": record["seed"], "params": params,
        })
        record["paint"] = {"key": painter, "params": params, "info": result.get("info", {}), "mesh": result["mesh"]}
        record.setdefault("previews", {})["Color"] = result["mesh"]
        self._save(run_dir, record)
        return record

    def print_ready(self, run_dir: str | Path, settings: PrintSettings, status: Status = _no_status) -> PrintReport:
        """Print step. Repeat it at any time for another height or face budget."""
        run_dir = Path(run_dir)
        record = self.load(run_dir)
        settings = replace(settings, source_up=self._up(record))
        status("Preparing for print")
        faces = f"{settings.target_faces // 1000}k" if settings.target_faces else "full"
        # A dot would read as a file extension further down.
        height = f"{settings.target_height_mm:g}mm".replace(".", "p") if settings.target_height_mm else "native"
        report = prepare(record["final_mesh"], run_dir / f"{record['name']}_{height}_{faces}", settings)
        record.update(print_settings=asdict(settings), report=report.as_dict())
        self._save(run_dir, record)
        return report

    def run(self, job: Job, status: Status = _no_status) -> JobResult:
        """Every step in order, unattended."""
        prepared = self.prepare_image(job.image, job.rembg_model, job.views, job.name, status)
        run_dir = self.generate(prepared, job.generator, job.seed, job.gen_params, status)
        if job.refiner:
            self.refine(run_dir, job.refiner, job.ref_params, status)
        if job.painter:
            self.paint(run_dir, job.painter, job.paint_params, status)
        report = self.print_ready(run_dir, job.print_settings, status)
        return JobResult(run_dir=run_dir, report=report, record=self.load(run_dir))

    def _add_preview(self, run_dir: Path, record: dict, stage: str, mesh: str, status: Status) -> None:
        status("Making a preview")
        path = make_preview(mesh, run_dir / f"preview_{stage.lower()}.glb", self._up(record))
        record.setdefault("previews", {})[stage] = str(path)
        self._save(run_dir, record)

    @staticmethod
    def _up(record: dict) -> str:
        # Runs from before the split into steps only have it in their print settings.
        return record.get("up") or record.get("print_settings", {}).get("source_up", "y")

    def _clear_results(self, run_dir: Path, record: dict) -> None:
        """Drop what was built on the current mesh, so nothing stale stays next to the new one."""
        record["up"] = self._up(record)
        for pattern in ("textured*", "paint_input*", "*.stl", "*.3mf", "*_preview.glb"):
            for path in run_dir.glob(pattern):
                path.unlink()
        for key in ("paint", "report", "print_settings"):
            record.pop(key, None)
        for stage in ("Refined", "Color"):
            record.get("previews", {}).pop(stage, None)

    @staticmethod
    def previews(record: dict) -> dict[str, str]:
        """Viewable meshes of a run by stage, in step order; old runs only have the print one."""
        stages = dict(record.get("previews", {}))
        if "report" in record:
            stages["Print"] = record["report"]["files"]["preview"]
        return stages

    def history(self, limit: int = 100) -> list[Path]:
        runs = sorted(
            self.outputs_dir.glob("*/*/run.json"), key=lambda p: (p.parent.parent.name, p.parent.name), reverse=True
        )
        return [p.parent for p in runs[:limit]]
