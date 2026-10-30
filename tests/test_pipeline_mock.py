"""End-to-end orchestration with mock workers (real subprocesses and HTTP, no GPU)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.orchestrator import Job, Runner, WorkerManager, load_config
from app.printprep import PrintSettings


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_DELAY", "0")
    config = load_config()
    workers = WorkerManager(config["workers"], tmp_path / "logs", mock=True, startup_timeout=60)
    yield Runner(config, tmp_path / "outputs", workers)
    workers.stop()


@pytest.fixture
def cutout(tmp_path):
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((64, 32, 192, 224), fill=(200, 80, 60, 255))
    path = tmp_path / "Hero Portrait.png"
    image.save(path)
    return path


def test_generate_refine_print(runner, cutout):
    job = Job(
        image=cutout, generator="hunyuan", refiner="hi3d", seed=7,
        gen_params={"steps": 5}, ref_params={"resolution": "512"},
        print_settings=PrintSettings(target_height_mm=28, target_faces=5000), rembg_model="u2net",
    )
    statuses = []
    result = runner.run(job, status=statuses.append)

    assert statuses[0] == "Preparing image" and statuses[-1] == "Preparing for print"
    assert result.record["name"] == "Hero_Portrait"
    assert result.record["background_removed"] is False
    assert result.report.watertight
    assert result.report.size_mm[2] == pytest.approx(28, rel=1e-3)
    for key in ("stl", "3mf", "preview"):
        assert Path(result.report.files[key]).parent == result.run_dir
        assert Path(result.report.files[key]).exists()
    saved = json.loads((result.run_dir / "run.json").read_text())
    assert saved["refiner"]["key"] == "hi3d" and saved["print_settings"]["source_up"] == "y"

    assert runner.history() == [result.run_dir]
    again = runner.reprint(result.run_dir, PrintSettings(target_height_mm=54, target_faces=2000))
    assert again.size_mm[2] == pytest.approx(54, rel=1e-3)
    assert again.faces <= 2100


def test_turnaround_views_are_cut_out_framed_and_recorded(runner, cutout, tmp_path):
    side = Image.new("RGBA", (128, 300), (0, 0, 0, 0))  # same figure, smaller upload
    ImageDraw.Draw(side).ellipse((40, 20, 88, 280), fill=(60, 80, 200, 255))
    side_path = tmp_path / "side.png"
    side.save(side_path)
    views = {"front": cutout, "left": side_path}
    job = Job(cutout, "hunyuan2mv", "ultrashape", 2, {}, {}, PrintSettings(target_faces=2000),
              "u2net", name="Sheet", views=views)
    result = runner.run(job)

    assert result.run_dir.name.endswith("_Sheet_hunyuan2mv+ultrashape_s2")
    assert result.record["views"] == {slot: str(result.run_dir / f"view_{slot}.png") for slot in views}
    framed = {slot: Image.open(result.run_dir / f"view_{slot}.png") for slot in views}
    heights = []
    for image in framed.values():
        assert image.size == (1024, 1024)
        rows = (np.asarray(image.getchannel("A")) > 16).any(axis=1).nonzero()[0]
        heights.append(rows.max() - rows.min())
    assert abs(heights[0] - heights[1]) <= 2  # framed to one figure height
    assert (result.run_dir / "input.png").exists() and result.report.watertight


def test_view_jobs_need_a_multiview_generator_and_a_front(runner, cutout):
    settings = PrintSettings(target_faces=2000)
    with pytest.raises(ValueError, match="single image"):
        runner.run(Job(cutout, "hunyuan", None, 1, {}, {}, settings, "u2net", views={"front": cutout}))
    with pytest.raises(ValueError, match="front"):
        runner.run(Job(cutout, "hunyuan2mv", None, 1, {}, {}, settings, "u2net", views={"back": cutout}))
    with pytest.raises(ValueError, match="front"):
        runner.run(Job(cutout, "hunyuan2mv", None, 1, {}, {}, settings, "u2net"))


def test_identical_jobs_get_separate_run_dirs(runner, cutout):
    settings = PrintSettings(target_faces=2000)
    first = runner.run(Job(cutout, "hunyuan", None, 5, {}, {}, settings, "u2net"))
    second = runner.run(Job(cutout, "hunyuan", None, 5, {}, {}, settings, "u2net"))
    assert first.run_dir != second.run_dir


def test_switching_models_restarts_worker(runner, cutout):
    settings = PrintSettings(target_faces=2000)
    runner.run(Job(cutout, "hunyuan", None, 1, {}, {}, settings, "u2net"))
    first = runner.workers._proc
    runner.run(Job(cutout, "hunyuan", None, 2, {}, {}, settings, "u2net"))
    assert runner.workers._proc is first  # same model: worker stays warm
    runner.run(Job(cutout, "hunyuan2mv", None, 3, {}, {}, settings, "u2net", views={"front": cutout}))
    assert runner.workers.current == "hunyuan2mv" and first.poll() is not None
