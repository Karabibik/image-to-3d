from pathlib import Path

import gradio as gr
import pytest
import trimesh
from PIL import Image

from app import world_ui
from app.orchestrator import WorkerError, WorkerManager
from app.printprep import preview_copy
from app.terrain import MAP_PIXELS, TileSettings
from app.world import WorldRunner
from app.world_ui import CONFIG


@pytest.fixture
def runner(tmp_path):
    workers = WorkerManager(CONFIG["workers"], tmp_path / "logs", mock=True, startup_timeout=60)
    yield WorldRunner(CONFIG, tmp_path / "outputs", workers)
    workers.stop()


@pytest.fixture
def art(tmp_path) -> Path:
    path = tmp_path / "harbour.jpg"
    Image.new("RGB", (640, 480), (90, 120, 150)).save(path)
    return path


@pytest.fixture
def world(runner, art) -> Path:
    world_dir = runner.panorama("hypano", 7, {"steps": 40}, image=art)
    runner.scene(world_dir, ["trees"], [], "outdoor", {"cache": False})
    return world_dir


def test_a_panorama_from_an_image_starts_a_world(runner, art):
    world_dir = runner.panorama("hypano", 7, {"steps": 40}, image=art, prompt="dusk")
    record = runner.load(world_dir)
    assert world_dir.parents[1].name == "world"
    assert record["seed"] == 7
    assert record["source"]["key"] == "hypano"
    assert record["source"]["prompt"] == "dusk"
    assert Image.open(record["panorama"]).size == (1920, 960)
    assert (world_dir / "input.jpg").exists()


def test_a_panorama_from_text_needs_a_description(runner):
    with pytest.raises(ValueError, match="Describe"):
        runner.panorama("hyworld1_text", 7, {}, prompt="  ")
    world_dir = runner.panorama("hyworld1_text", 7, {}, prompt="a rocky island at sea")
    assert "a_rocky_island" in world_dir.name


def test_a_panorama_from_an_image_needs_one(runner):
    with pytest.raises(ValueError, match="Upload"):
        runner.panorama("hyworld1_image", 7, {})


def test_every_new_seed_is_a_new_world(runner, art):
    assert runner.panorama("hypano", 1, {}, image=art) != runner.panorama("hypano", 2, {}, image=art)
    assert len(runner.history()) == 2


def test_an_uploaded_panorama_is_resized_to_what_the_world_step_expects(runner, tmp_path):
    path = tmp_path / "pano.jpg"
    Image.new("RGB", (4096, 2048), (10, 20, 30)).save(path)
    record = runner.load(runner.use_panorama(path, 3))
    assert Image.open(record["panorama"]).size == (1920, 960)
    assert record["source"]["key"] == "upload"


def test_a_picture_that_is_no_panorama_is_refused(runner, art):
    with pytest.raises(ValueError, match="twice as wide"):
        runner.use_panorama(art, 3)
    assert runner.history() == []


def test_building_a_world(runner, world):
    record = runner.load(world)
    scene = record["scene"]
    assert [layer["type"] for layer in scene["layers"]] == ["fg1", "bg", "sky"]
    assert all(Path(layer["path"]).exists() for layer in scene["layers"])
    assert scene["labels_fg1"] == ["trees"]
    assert scene["unit"] == pytest.approx(1.7, abs=0.2)
    assert Path(scene["world"]).exists()
    assert Image.open(scene["map"]).size == (MAP_PIXELS, MAP_PIXELS)
    assert runner.previews(record) == {"World": scene["preview"]}
    assert not (world / "scene_new").exists()


def test_cutting_tiles(runner, world):
    first = runner.tile(world, TileSettings(center_x=3, center_y=2, size=6, grid=120), 40_000, ("stl",))
    second = runner.tile(world, TileSettings(center_x=-2, size=5, grid=120), 40_000, ("stl", "3mf"))
    assert first.watertight and second.watertight
    record = runner.load(world)
    assert list(runner.previews(record)) == ["World", "Tile 1", "Tile 2"]
    assert Path(first.files["stl"]).exists()  # the second tile keeps the first
    files = runner.files(record)
    assert files[0] == second.files["stl"]
    assert record["panorama"] in files
    assert record["scene"]["world"] in files
    assert not list((world / "tiles").glob("*_source.ply"))


def test_a_tile_needs_a_world(runner, art):
    world_dir = runner.panorama("hypano", 7, {}, image=art)
    with pytest.raises(ValueError, match="Build the world"):
        runner.tile(world_dir, TileSettings(), 40_000, ("stl",))


def test_rebuilding_drops_the_old_tiles(runner, world):
    runner.tile(world, TileSettings(grid=64), 40_000, ("stl",))
    record = runner.scene(world, [], ["houses"], "outdoor", {})
    assert "tiles" not in record
    assert not (world / "tiles").exists()
    assert record["scene"]["labels_fg2"] == ["houses"]


def test_a_failed_rebuild_keeps_the_world(runner, world, monkeypatch):
    before = runner.load(world)
    runner.tile(world, TileSettings(grid=64), 40_000, ("stl",))

    def refuse(_name, _job):
        raise WorkerError("hyworld1: out of memory")

    monkeypatch.setattr(runner.workers, "run", refuse)
    with pytest.raises(WorkerError):
        runner.scene(world, [], [], "outdoor", {})
    after = runner.load(world)
    assert after["scene"] == before["scene"]
    assert len(after["tiles"]) == 1
    assert all(Path(layer["path"]).exists() for layer in after["scene"]["layers"])


def test_the_ui_builds_with_every_step_in_one_gpu_queue():
    steps = [fn for fn in world_ui.build().fns.values() if fn.concurrency_id == "gpu"]
    assert len(steps) == len(world_ui.STEPS)
    assert not any(isinstance(output, gr.Model3D) for fn in steps for output in fn.outputs)  # stays blank


def test_labels_split_on_commas():
    assert world_ui._labels(" trees, stones ;; ") == ["trees", "stones"]


@pytest.mark.parametrize("step", [world_ui._step_world, world_ui._step_tile])
def test_later_steps_ask_for_a_panorama(step):
    with pytest.raises(ValueError, match="panorama"):
        step({"labels_fg1": "", "labels_fg2": "", "classes": "outdoor", "faces": "200k", "formats": ["stl"],
              "center_x": 0, "center_y": 0, "size": 4, "rotation": 0, "grid": 100, "width_mm": 50, "base_mm": 3,
              "exaggeration": 1, "smoothing": 0}, {"world": None}, lambda _msg: None)


def test_the_map_is_empty_until_there_is_a_world():
    image, hint = world_ui._map({"world": None}, 0, 0, 4, 0)
    assert image is None
    assert "Build the world" in hint


def test_clicking_the_map_places_the_tile(runner, world, monkeypatch):
    monkeypatch.setattr(world_ui, "RUNNER", runner)
    half = runner.load(world)["scene"]["map_half"]

    class Click:
        index = [MAP_PIXELS * 3 // 4, MAP_PIXELS // 4]

    x, y = world_ui._pick({"world": str(world)}, Click)
    assert x == pytest.approx(half / 2, abs=0.05)
    assert y == pytest.approx(half / 2, abs=0.05)


def test_every_world_model_is_described_and_fetched_without_its_unused_weights():
    for spec in CONFIG["panoramas"].values():
        assert spec["hint"] and spec["worker"] in CONFIG["workers"]
    ports = [spec["port"] for spec in CONFIG["workers"].values()]
    assert len(ports) == len(set(ports))
    fetched = {item["repo"]: item for spec in CONFIG["workers"].values() for item in spec["prefetch"]}
    assert fetched["tencent/HY-World-2.0"]["allow"] == ["HY-Pano-2.0/pytorch_lora_weights.safetensors"]
    for repo in ("black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-Fill-dev"):
        assert "flux1-*.safetensors" in fetched[repo]["ignore"]


def test_object_previews_still_fit_by_height():
    wide = trimesh.creation.box(extents=(10, 10, 1))
    assert preview_copy(wide).extents[1] == pytest.approx(1)
    assert preview_copy(wide, whole=True).extents.max() == pytest.approx(1)
