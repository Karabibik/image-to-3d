from pathlib import Path

import gradio as gr
import pytest
from gradio.components.image import ImageData
from PIL import Image, ImageDraw

from app import ui, ui_common
from app.orchestrator import Runner
from app.ui import build


def test_no_generator_writes_to_a_3d_viewer():
    # Model3D stays blank when a generator feeds it: no error, just no canvas in the browser.
    for fn in build().fns.values():
        if fn.types_generator:
            assert not [block for block in fn.outputs if isinstance(block, gr.Model3D)], fn.name


def test_steps_share_one_gpu_queue():
    steps = [fn for fn in build().fns.values() if fn.api_name in ("image", "shape", "refine", "color", "print", "run_all")]
    assert len(steps) == 6
    assert {fn.concurrency_id for fn in steps} == {"gpu"}
    assert {fn.concurrency_limit for fn in steps} == {1}


@pytest.mark.parametrize("refiner, color, current, prepared, expected", [
    (ui.NO_REFINER, False, 1, None, ["Image", "Shape", "Print"]),
    (ui.NO_REFINER, False, 1, "cut-out", ["Shape", "Print"]),
    ("ultrashape", True, 1, None, ["Image", "Shape", "Refine", "Color", "Print"]),
    ("ultrashape", False, 3, "cut-out", ["Refine", "Print"]),
    (ui.NO_REFINER, True, 4, "cut-out", ["Color", "Print"]),
    (ui.NO_REFINER, False, 5, "cut-out", ["Print"]),
])
def test_run_all_goes_from_the_current_step_and_skips_what_is_off_or_done(refiner, color, current, prepared, expected):
    chosen = ui._steps_from({"refiner": refiner, "include_color": color, "current": current,
                             "session": {"prepared": prepared}})
    assert [step.title for step in chosen] == expected


@pytest.mark.parametrize("step", [ui._step_refine, ui._step_color, ui._step_print])
def test_later_steps_ask_for_a_shape_first(step):
    with pytest.raises(ValueError, match="Generate a shape first"):
        step({"refiner": "ultrashape"}, {"prepared": None, "run": None}, lambda _msg: None)


def test_a_turnaround_sheet_becomes_framed_views(tmp_path, monkeypatch):
    sheet = Image.new("RGBA", (900, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for left in (40, 340, 640):  # three figures with clear gaps
        draw.ellipse((left, 20, left + 200, 280), fill=(180, 90, 60, 255))
    path = tmp_path / "Kano sheet.png"
    sheet.save(path)
    monkeypatch.setattr(ui, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(ui, "RUNNER", Runner(ui.CONFIG, tmp_path / "outputs", ui.WORKERS))

    values = {"mode": "multiview", "sheet": str(path), "order": "front, left, back", "rembg_model": "u2net",
              **dict.fromkeys(ui.SLOTS)}
    session = {"prepared": None, "run": "an earlier run"}
    note = ui._step_image(values, session, lambda _msg: None)

    assert note == "found 3 figure(s), used 3"
    assert session["run"] is None and session["prepared"].name == "Kano_sheet"
    assert sorted(session["prepared"].views) == ["back", "front", "left"]
    assert all(Path(view).name.startswith("view_") and Path(view).exists() for view in session["prepared"].views.values())
    assert [caption for _, caption in ui._gallery(session["prepared"])] == [
        ui.SLOT_LABELS[slot] for slot in session["prepared"].views
    ]


@pytest.mark.parametrize("name, mode", [("art.jpg", "RGB"), ("art.png", "RGBA"), ("art.webp", "RGB"), ("art.png", "P")])
def test_every_upload_box_takes_common_image_formats(tmp_path, name, mode):
    path = tmp_path / name
    Image.new(mode, (32, 32)).save(path)
    uploads = [block for block in build().blocks.values() if isinstance(block, gr.Image)]
    assert len(uploads) == 6  # one image, one sheet, four views
    for upload in uploads:
        assert upload.preprocess(ImageData(path=str(path), orig_name=name)) == str(path)


def test_failures_before_a_step_starts_are_shown(monkeypatch, tmp_path):
    options = {}
    monkeypatch.setattr(gr.Blocks, "launch", lambda _self, **kwargs: options.update(kwargs))
    monkeypatch.setattr(ui_common, "OUTPUT_DIR", tmp_path / "outputs")
    ui.main()
    assert options["show_error"] is True


def test_every_model_has_a_hint():
    for section in ("generators", "refiners", "painters"):
        for spec in ui.CONFIG[section].values():
            assert spec.get("hint")
