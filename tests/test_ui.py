import gradio as gr

from app.ui import build


def test_no_generator_writes_to_a_3d_viewer():
    # Model3D stays blank when a generator feeds it: no error, just no canvas in the browser.
    for fn in build().fns.values():
        if fn.types_generator:
            assert not [block for block in fn.outputs if isinstance(block, gr.Model3D)], fn.name
