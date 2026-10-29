"""End-to-end orchestration with mock workers (real subprocesses and HTTP, no GPU)."""

import json
from pathlib import Path

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


def test_generator_facing_correction_applies_to_runs_and_reprints(runner, cutout):
    settings = PrintSettings(target_faces=2000, rotate_z_deg=90)
    result = runner.run(Job(cutout, "pixal3d", "hi3d", 3, {}, {}, settings, "u2net"))
    assert result.record["print_settings"]["rotate_z_deg"] == 270  # user's 90 + Pixal3D's 180
    plain = runner.run(Job(cutout, "hunyuan", None, 3, {}, {}, settings, "u2net"))
    assert plain.record["print_settings"]["rotate_z_deg"] == 90
    runner.reprint(result.run_dir, PrintSettings(target_faces=2000))  # must not raise or drop the turn


def test_identical_jobs_get_separate_run_dirs(runner, cutout):
    settings = PrintSettings(target_faces=2000)
    first = runner.run(Job(cutout, "hunyuan", None, 5, {}, {}, settings, "u2net"))
    second = runner.run(Job(cutout, "hunyuan", None, 5, {}, {}, settings, "u2net"))
    assert first.run_dir != second.run_dir


def test_switching_models_restarts_worker(runner, cutout):
    settings = PrintSettings(target_faces=2000)
    runner.run(Job(cutout, "trellis", None, 1, {}, {}, settings, "u2net"))
    first = runner.workers._proc
    runner.run(Job(cutout, "trellis", None, 2, {}, {}, settings, "u2net"))
    assert runner.workers._proc is first  # same model: worker stays warm
    runner.run(Job(cutout, "pixal3d", None, 3, {}, {}, settings, "u2net"))
    assert runner.workers.current == "pixal3d" and first.poll() is not None
