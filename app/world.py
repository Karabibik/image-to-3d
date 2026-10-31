"""The steps of a world: panorama, layered meshes, terrain tiles."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from .orchestrator import Status, WorkerManager, _new_dir, _no_status, safe_name
from .printprep import PrintReport
from .terrain import TileSettings, cut_tile, export_world

# What HunyuanWorld's layer and depth steps are built around.
PANORAMA_SIZE = (1920, 960)
# Width over height an uploaded panorama may have; beyond this it is not a full 360-degree view.
PANORAMA_RATIO = (1.9, 2.1)


class WorldRunner:
    """One method per step. Each panorama starts a folder whose world.json is rewritten after every step."""

    def __init__(self, config: dict, outputs_dir: Path, workers: WorkerManager):
        self.config = config
        self.outputs_dir = Path(outputs_dir)
        self.workers = workers

    def load(self, world_dir: str | Path) -> dict:
        return json.loads((Path(world_dir) / "world.json").read_text(encoding="utf-8"))

    def _save(self, world_dir: Path, record: dict) -> None:
        (world_dir / "world.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _new_world(self, name: str, seed: int) -> Path:
        now = datetime.now()
        return _new_dir(self.outputs_dir / "world" / f"{now:%Y-%m-%d}" / f"{now:%H%M%S}_{name}_s{seed}")

    def panorama(self, source: str, seed: int, params: dict, image: Path | None = None, prompt: str = "",
                 status: Status = _no_status) -> Path:
        """Panorama step. Always a new world, so another seed never replaces an earlier one."""
        spec = self.config["panoramas"][source]
        if spec["input"] == "image" and not image:
            raise ValueError("Upload an image first.")
        if spec["input"] == "text" and not prompt.strip():
            raise ValueError("Describe the scene first.")
        name = safe_name(Path(image).stem if image else prompt[:30])
        world_dir = self._new_world(name, seed)
        request = {
            "task": spec["task"], "model": source, "prompt": prompt, "out": str(world_dir / "panorama.png"),
            "seed": seed, "params": params,
        }
        if image:
            request["image"] = str(shutil.copy(image, world_dir / f"input{Path(image).suffix}"))
        status(f"Making a panorama with {spec['label']}")
        result = self.workers.run(spec["worker"], request)
        self._start_record(world_dir, name, seed, source, prompt, result["image"], params, result.get("info", {}))
        return world_dir

    def use_panorama(self, image: Path, seed: int, status: Status = _no_status) -> Path:
        """Panorama step for a 360-degree image made elsewhere."""
        name = safe_name(Path(image).stem)
        with Image.open(image) as picture:
            ratio = picture.width / picture.height
            if not PANORAMA_RATIO[0] <= ratio <= PANORAMA_RATIO[1]:
                raise ValueError(f"A panorama is twice as wide as it is tall; this one is {ratio:.2f} to 1.")
            status("Resizing the panorama")
            world_dir = self._new_world(name, seed)
            path = world_dir / "panorama.png"
            picture.convert("RGB").resize(PANORAMA_SIZE, Image.LANCZOS).save(path)
        self._start_record(world_dir, name, seed, "upload", "", str(path), {}, {})
        return world_dir

    def _start_record(self, world_dir: Path, name: str, seed: int, source: str, prompt: str, panorama: str,
                      params: dict, info: dict) -> None:
        self._save(world_dir, {
            "name": name, "created": datetime.now().isoformat(timespec="seconds"), "seed": seed,
            "source": {"key": source, "prompt": prompt, "params": params, "info": info}, "panorama": panorama,
        })

    def scene(self, world_dir: str | Path, labels_fg1: list[str], labels_fg2: list[str], classes: str, params: dict,
              status: Status = _no_status) -> dict:
        """World step. Can be repeated with other labels, which drops the tiles of the previous build."""
        world_dir = Path(world_dir)
        record = self.load(world_dir)
        scratch = world_dir / "scene_new"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir()
        status("Building the world")
        result = self.workers.run(self.config["scene"]["worker"], {
            "task": "scene", "model": "scene", "image": record["panorama"], "out": str(scratch),
            "labels_fg1": labels_fg1, "labels_fg2": labels_fg2, "classes": classes, "seed": record["seed"],
            "params": params,
        })
        # Only now: a failed build keeps what the world had.
        folder = world_dir / "scene"
        shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(world_dir / "tiles", ignore_errors=True)
        record.pop("scene", None)
        record.pop("tiles", None)
        self._save(world_dir, record)
        scratch.rename(folder)

        layers = [{"type": layer["type"], "path": str(folder / layer["file"]), "faces": layer["faces"]}
                  for layer in result["layers"]]
        status("Merging the layers")
        record["scene"] = {
            "labels_fg1": labels_fg1, "labels_fg2": labels_fg2, "classes": classes, "params": params,
            "info": result.get("info", {}), "layers": layers, **export_world(layers, world_dir),
        }
        self._save(world_dir, record)
        return record

    def tile(self, world_dir: str | Path, settings: TileSettings, target_faces: int | None, formats: tuple[str, ...],
             status: Status = _no_status) -> PrintReport:
        """Tile step. Repeat it for as many tiles as the world is worth; each keeps its own files."""
        world_dir = Path(world_dir)
        record = self.load(world_dir)
        scene = record.get("scene")
        if not scene:
            raise ValueError("Build the world first.")
        status("Cutting the tile")
        tiles = record.setdefault("tiles", [])
        folder = world_dir / "tiles"
        folder.mkdir(exist_ok=True)
        # A dot would read as a file extension further down.
        name = f"{record['name']}_tile{len(tiles) + 1}_{settings.width_mm:g}mm".replace(".", "p")
        report = cut_tile(scene["layers"], scene["unit"], settings, folder / name, target_faces, formats)
        tiles.append({"settings": asdict(settings), "report": report.as_dict()})
        self._save(world_dir, record)
        return report

    @staticmethod
    def previews(record: dict) -> dict[str, str]:
        """Viewable meshes of a world by stage, oldest first."""
        stages = {}
        if record.get("scene"):
            stages["World"] = record["scene"]["preview"]
        for number, tile in enumerate(record.get("tiles", []), start=1):
            stages[f"Tile {number}"] = tile["report"]["files"]["preview"]
        return stages

    @staticmethod
    def files(record: dict) -> list[str]:
        """Downloads, the print files first."""
        files = [path for tile in reversed(record.get("tiles", []))
                 for kind, path in tile["report"]["files"].items() if kind != "preview"]
        files.append(record["panorama"])
        scene = record.get("scene")
        if scene:
            files.append(scene["world"])
            files.extend(layer["path"] for layer in scene["layers"])
        return files

    def history(self, limit: int = 100) -> list[Path]:
        worlds = sorted(self.outputs_dir.glob("world/*/*/world.json"),
                        key=lambda p: (p.parent.parent.name, p.parent.name), reverse=True)
        return [p.parent for p in worlds[:limit]]
