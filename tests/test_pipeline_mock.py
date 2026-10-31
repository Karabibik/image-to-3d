"""End-to-end orchestration with mock workers (real subprocesses and HTTP, no GPU)."""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
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
    again = runner.print_ready(result.run_dir, PrintSettings(target_height_mm=54.5, target_faces=2000))
    assert again.size_mm[2] == pytest.approx(54.5, rel=1e-3)
    assert again.faces <= 2100
    assert Path(again.files["stl"]).name == "Hero_Portrait_54p5mm_2k.stl"


def test_steps_run_one_at_a_time_and_save_as_they_go(runner, cutout):
    prepared = runner.prepare_image(cutout, "u2net")
    assert prepared.image.parent.parent.name == "prepared"
    assert runner.history() == []  # nothing to reopen until a shape exists

    run_dir = runner.generate(prepared, "hunyuan", 3, {"steps": 5})
    record = runner.load(run_dir)
    assert record["final_mesh"] == record["raw_mesh"] == str(run_dir / "raw.glb")
    assert list(runner.previews(record)) == ["Shape"]
    assert run_dir.name.endswith("_Hero_Portrait_hunyuan_s3")

    runner.refine(run_dir, "ultrashape", {"steps": 12})
    record = runner.load(run_dir)
    assert record["final_mesh"] == str(run_dir / "refined.glb")
    assert record["refiner"]["params"] == {"steps": 12}

    runner.paint(run_dir, "hunyuan_paint", {"faces": 500})
    record = runner.load(run_dir)
    paint_input = trimesh.load(run_dir / "paint_input.obj", force="mesh")
    assert len(paint_input.faces) <= 550
    assert paint_input.extents[1] == pytest.approx(1.0, rel=1e-3)  # framed like the other previews
    textured = trimesh.load(record["paint"]["mesh"], force="mesh")
    assert textured.visual.kind == "texture" and textured.visual.material.baseColorTexture is not None

    report = runner.print_ready(run_dir, PrintSettings(target_faces=2000))
    assert list(runner.previews(runner.load(run_dir))) == ["Shape", "Refined", "Color", "Print"]
    assert len(trimesh.load(report.files["stl"]).faces) <= 2100  # painting left the print mesh alone


def test_each_generate_is_a_new_run(runner, cutout):
    prepared = runner.prepare_image(cutout, "u2net")
    first = runner.generate(prepared, "hunyuan", 1, {})
    second = runner.generate(prepared, "hunyuan", 2, {})
    assert first != second and set(runner.history()) == {first, second}
    assert (first / "input.png").exists() and (second / "input.png").exists()


def test_refining_again_clears_what_was_built_on_the_old_mesh(runner, cutout):
    run_dir = runner.generate(runner.prepare_image(cutout, "u2net"), "hunyuan", 4, {})
    runner.refine(run_dir, "ultrashape", {})
    runner.paint(run_dir, "hunyuan_paint", {"faces": 500})
    report = runner.print_ready(run_dir, PrintSettings(target_faces=2000))

    runner.refine(run_dir, "hi3d", {"resolution": "512"})
    record = runner.load(run_dir)
    assert record["refiner"]["key"] == "hi3d"
    assert "paint" not in record and "report" not in record
    assert not Path(report.files["stl"]).exists() and not (run_dir / "textured.glb").exists()
    assert list(runner.previews(record)) == ["Shape", "Refined"]
    assert (run_dir / "raw.glb").exists()


def test_runs_from_before_the_steps_still_open(runner, cutout):
    run_dir = runner.run(Job(cutout, "hunyuan", None, 6, {}, {}, PrintSettings(target_faces=2000), "u2net")).run_dir
    old = runner.load(run_dir)
    for key in ("raw_mesh", "up", "previews"):
        del old[key]
    (run_dir / "run.json").write_text(json.dumps(old))

    assert list(runner.previews(old)) == ["Print"]
    assert runner.prepared_from(run_dir).image == run_dir / "input.png"
    assert runner.print_ready(run_dir, PrintSettings(target_height_mm=28, target_faces=2000)).watertight
    runner.refine(run_dir, "ultrashape", {})
    assert runner.load(run_dir)["up"] == "y"
    assert runner.generate(runner.prepared_from(run_dir), "hunyuan", 7, {}) != run_dir


def test_turnaround_views_are_cut_out_framed_and_recorded(runner, cutout, tmp_path):
    side = Image.new("RGBA", (128, 300), (0, 0, 0, 0))  # same figure, smaller upload
    ImageDraw.Draw(side).ellipse((40, 20, 88, 280), fill=(60, 80, 200, 255))
    side_path = tmp_path / "side.png"
    side.save(side_path)
    views = {"front": cutout, "left": side_path}
    job = Job(cutout, "hunyuan2mv", "ultrashape", 2, {}, {}, PrintSettings(target_faces=2000),
              "u2net", name="Sheet", views=views)
    result = runner.run(job)

    assert result.run_dir.name.endswith("_Sheet_hunyuan2mv_s2")
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
