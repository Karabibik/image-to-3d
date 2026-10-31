import gradio as gr
import pytest

from app import ui
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


@pytest.mark.parametrize("refiner, color, current, expected", [
    (ui.NO_REFINER, False, 1, ["Image", "Shape", "Print"]),
    ("ultrashape", True, 1, ["Image", "Shape", "Refine", "Color", "Print"]),
    ("ultrashape", False, 3, ["Refine", "Print"]),
    (ui.NO_REFINER, True, 4, ["Color", "Print"]),
    (ui.NO_REFINER, False, 5, ["Print"]),
])
def test_run_all_goes_from_the_current_step_and_skips_optional_steps(refiner, color, current, expected):
    chosen = ui._steps_from({"refiner": refiner, "include_color": color, "current": current})
    assert [step.title for step in chosen] == expected


@pytest.mark.parametrize("step", [ui._step_refine, ui._step_color, ui._step_print])
def test_later_steps_ask_for_a_shape_first(step):
    with pytest.raises(ValueError, match="Generate a shape first"):
        step({"refiner": "ultrashape"}, {"prepared": None, "run": None}, lambda _msg: None)


def test_every_model_has_a_hint():
    for section in ("generators", "refiners", "painters"):
        for spec in ui.CONFIG[section].values():
            assert spec.get("hint")
