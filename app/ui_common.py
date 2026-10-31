"""What the object and world UIs share: paths, look, model controls, progress and launching."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import gradio as gr

from .orchestrator import Status, WorkerManager

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace" if Path("/workspace").is_dir() else Path.cwd()))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DATA_DIR / "outputs"))
LOG_DIR = Path(os.environ.get("LOG_DIR", DATA_DIR / "logs"))
CSS_PATH = Path(__file__).with_name("ui.css")
# Every preview is one unit tall, so one camera distance frames them all, with room for a wide base.
VIEWER_STYLE = {"display_mode": "solid", "clear_color": (0.15, 0.15, 0.17, 1.0), "camera_position": (None, 85, 1.8)}
# image_mode None passes the upload through untouched: converting it here fails on JPEGs, and
# app.imageprep handles every format anyway.
UPLOAD_STYLE = {"show_label": False, "type": "filepath", "image_mode": None, "format": "png", "sources": ["upload"]}
THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=(gr.themes.GoogleFont("Inter", weights=(400, 500, 600)), "system-ui", "Segoe UI", "sans-serif"),
    font_mono=("ui-monospace", "Consolas", "monospace"),
    spacing_size=gr.themes.sizes.spacing_sm,
    radius_size=gr.themes.sizes.radius_sm,
).set(
    block_title_background_fill="none",
    block_title_padding="0",
    block_title_text_color="*body_text_color_subdued",
    block_title_text_color_dark="*body_text_color_subdued",
    block_title_text_weight="500",
)


def control(spec: dict) -> gr.components.Component:
    common = {"label": spec.get("label"), "info": spec.get("info")}
    kind = spec["type"]
    if kind in ("int", "float"):
        step = spec.get("step", 1 if kind == "int" else 0.1)
        return gr.Slider(spec["min"], spec["max"], value=spec["default"], step=step, **common)
    if kind == "bool":
        return gr.Checkbox(value=spec["default"], **common)
    if kind == "choice":
        return gr.Radio(spec["choices"], value=spec["default"], **common)
    raise ValueError(f"unknown param type {kind!r}")


def model_panels(models: dict, selected: str, prefix: str, inputs: dict, workers: WorkerManager) -> dict:
    """One hint plus an Advanced panel per model; only the selected model's panel is visible."""
    panels = {}
    for key, spec in models.items():
        missing = "" if workers.available(spec["worker"]) else " **Not installed in this image.**"
        with gr.Column(visible=key == selected) as panel:
            gr.Markdown(f"{spec.get('hint', '')}{missing}", elem_classes="hint")
            with gr.Accordion("Advanced", open=False):
                for name, param in spec["params"].items():
                    inputs[f"{prefix}.{key}.{name}"] = control(param)
        panels[key] = panel
    return panels


def model_params(values: dict, prefix: str, key: str) -> dict:
    start = f"{prefix}.{key}."
    return {name[len(start):]: value for name, value in values.items() if name.startswith(start)}


def watch(title: str, work: Callable[[Status], str], status: gr.Markdown, log: gr.Textbox, workers: WorkerManager):
    """Run `work` on a thread, yielding its progress; returns its summary line and duration."""
    progress = {"status": "Starting", "result": None, "error": None}

    def target() -> None:
        try:
            progress["result"] = work(lambda msg: progress.__setitem__("status", msg))
        except Exception as exc:  # shown in the status line
            progress["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    started = time.time()
    thread.start()
    while thread.is_alive():
        # Regular yields also keep the RunPod proxy connection alive on long jobs.
        yield {status: f"**{title}** · {progress['status']} · {time.time() - started:.0f}s",
               log: workers.tail(workers.current)}
        thread.join(timeout=1.5)
    if progress["error"] is not None:
        raise progress["error"]
    return progress["result"], time.time() - started


def launch(demo: gr.Blocks) -> None:
    password = os.environ.get("UI_PASSWORD")
    auth = (os.environ.get("UI_USERNAME", "admin"), password) if password else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("UI_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("UI_PORT", "7860")),
        auth=auth,
        allowed_paths=[str(OUTPUT_DIR)],
        theme=THEME,
        css_paths=CSS_PATH,
        footer_links=[],
        # Failures before a step's code runs cannot reach the status line; without this they are silent.
        show_error=True,
    )
