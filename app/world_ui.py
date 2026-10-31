"""Gradio UI of the world image: take a picture or a description to a printable terrain tile."""

from __future__ import annotations

import atexit
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gradio as gr
from PIL import Image

from .orchestrator import Status, WorkerManager, load_config
from .terrain import MAP_PIXELS, TileSettings, grid_step, map_with_tile
from .ui_common import LOG_DIR, OUTPUT_DIR, UPLOAD_STYLE, VIEWER_STYLE, launch, model_panels, model_params, watch
from .world import WorldRunner

SOURCES = [("Image", "image"), ("Text", "text"), ("My own panorama", "upload")]
PANORAMA = "Panorama"
# A world is wide and flat, so look down on it. Previews fit by their longest side.
WORLD_VIEWER_STYLE = {**VIEWER_STYLE, "camera_position": (None, 55, 1.9)}

CONFIG = load_config(Path(__file__).with_name("models.world.yaml"))
WORKERS = WorkerManager(CONFIG["workers"], LOG_DIR, mock=os.environ.get("MOCK_WORKERS") == "1")
RUNNER = WorldRunner(CONFIG, OUTPUT_DIR, WORKERS)
atexit.register(WORKERS.stop)


# ---------------------------------------------------------------- steps

@dataclass(frozen=True)
class Step:
    tab: int
    title: str
    # Takes the control values, the session and a status callback; returns a summary line.
    work: Callable[[dict, dict, Status], str]


def _text_model() -> str:
    return next(key for key, spec in CONFIG["panoramas"].items() if spec["input"] == "text")


def _labels(text: str) -> list[str]:
    return [label.strip() for label in text.replace(";", ",").split(",") if label.strip()]


def _need_world(session: dict) -> str:
    if not session["world"]:
        raise ValueError("Make a panorama first.")
    return session["world"]


def _tile_settings(v: dict) -> TileSettings:
    return TileSettings(
        center_x=float(v["center_x"] or 0), center_y=float(v["center_y"] or 0), size=float(v["size"]),
        rotation_deg=float(v["rotation"] or 0), grid=int(v["grid"]), width_mm=float(v["width_mm"]),
        base_mm=float(v["base_mm"]), exaggeration=float(v["exaggeration"]), smoothing=float(v["smoothing"]),
    )


def _step_panorama(v: dict, session: dict, say: Status) -> str:
    seed = int(v["seed"])
    if v["source"] == "upload":
        if not v["image"]:
            raise ValueError("Upload a panorama first.")
        world = RUNNER.use_panorama(Path(v["image"]), seed, say)
    else:
        key = v["model"] if v["source"] == "image" else _text_model()
        image = Path(v["image"]) if v["source"] == "image" and v["image"] else None
        world = RUNNER.panorama(key, seed, model_params(v, "pano", key), image, v["prompt"], say)
    session["world"] = str(world)
    return f"seed {seed}"


def _step_world(v: dict, session: dict, say: Status) -> str:
    record = RUNNER.scene(_need_world(session), _labels(v["labels_fg1"]), _labels(v["labels_fg2"]), v["classes"],
                          model_params(v, "scene", "scene"), say)
    layers = record["scene"]["layers"]
    return f"{len(layers)} layers · {sum(layer['faces'] for layer in layers):,} faces"


def _step_tile(v: dict, session: dict, say: Status) -> str:
    faces = CONFIG["print"]["faces"].get(v["faces"], v["faces"])
    report = RUNNER.tile(_need_world(session), _tile_settings(v), int(faces) or None,
                         tuple(v["formats"]) or ("stl",), say)
    size = " × ".join(f"{value:.1f}" for value in report.size_mm)
    parts = [f"{report.faces:,} faces", f"{size} mm", "watertight" if report.watertight else "⚠ not watertight"]
    return " · ".join(parts + [f"⚠ {warning}" for warning in report.warnings])


STEPS = [Step(1, "Panorama", _step_panorama), Step(2, "World", _step_world), Step(3, "Tile", _step_tile)]


# ---------------------------------------------------------------- results

def _history_choices() -> list[tuple[str, str]]:
    choices = []
    for world_dir in RUNNER.history():
        try:
            record = RUNNER.load(world_dir)
        except (OSError, json.JSONDecodeError):
            continue
        created = record["created"][5:16].replace("T", " ")
        built = " · world built" if record.get("scene") else ""
        choices.append((f"{created} · {record['name']} · s{record['seed']}{built}", str(world_dir)))
    return choices


def _map(state: dict, x: float, y: float, size: float, rotation: float) -> tuple[Image.Image | None, str]:
    """The Tile step's map with the tile as the controls have it, and the line that explains it."""
    scene = RUNNER.load(state["world"]).get("scene") if state["world"] else None
    if not scene:
        return None, "Build the world first."
    half = scene["map_half"]
    settings = TileSettings(center_x=float(x or 0), center_y=float(y or 0), size=float(size or 1),
                            rotation_deg=float(rotation or 0))
    hint = (f"Click the map to place the tile. Grid: {grid_step(half)} camera heights. "
            "The cross is the camera; ground near it is the most accurate.")
    return map_with_tile(Image.open(scene["map"]), half, settings), hint


def _pick(state: dict, event: gr.SelectData) -> tuple[float, float]:
    half = RUNNER.load(state["world"])["scene"]["map_half"]
    column, row = event.index
    return round(column / MAP_PIXELS * 2 * half - half, 2), round(half - row / MAP_PIXELS * 2 * half, 2)


# ---------------------------------------------------------------- layout

def build() -> gr.Blocks:
    panoramas = CONFIG["panoramas"]
    image_models = {key: spec for key, spec in panoramas.items() if spec["input"] == "image"}
    default_model = next(iter(image_models))
    defaults = CONFIG["tile"]["defaults"]
    face_labels = {v: k for k, v in CONFIG["print"]["faces"].items()}
    inputs: dict[str, gr.components.Component] = {}

    with gr.Blocks(title="Image to World", fill_width=True) as demo:
        session = gr.State({"world": None})
        preview = gr.State()
        inputs["session"] = session

        with gr.Row(elem_id="workspace", equal_height=False):
            with gr.Column(scale=0, min_width=380, elem_id="controls"):
                with gr.Tabs(selected=1, overflow_behavior="wrap") as tabs:
                    with gr.Tab("1 Panorama", id=1):
                        source = gr.Radio(SOURCES, value="image", show_label=False, container=False)
                        image = gr.Image(height=170, placeholder="Drop an image", **UPLOAD_STYLE)
                        prompt = gr.Textbox(label="Describe the scene (optional)")
                        model = gr.Radio([(spec["label"], key) for key, spec in image_models.items()],
                                         value=default_model, show_label=False, container=False)
                        with gr.Row():
                            seed = gr.Number(value=1234, precision=0, label="Seed", scale=2, min_width=100)
                            dice = gr.Button("Random seed", scale=1, min_width=100, elem_id="dice")
                        panorama_button = gr.Button("Make panorama", variant="primary")
                        model_hints = model_panels(panoramas, default_model, "pano", inputs, WORKERS)

                    with gr.Tab("2 World", id=2):
                        gr.Markdown("Name what should stand apart from the ground, or leave both empty.",
                                    elem_classes="hint")
                        labels_fg1 = gr.Textbox(label="Near objects", placeholder="trees, stones")
                        labels_fg2 = gr.Textbox(label="Objects behind them", placeholder="houses")
                        classes = gr.Radio(["outdoor", "indoor"], value="outdoor", show_label=False, container=False)
                        world_button = gr.Button("Build world", variant="primary")
                        model_panels({"scene": CONFIG["scene"]}, "scene", "scene", inputs, WORKERS)

                    with gr.Tab("3 Tile", id=3):
                        map_view = gr.Image(show_label=False, interactive=False, height=230, type="pil",
                                            format="png", elem_id="map")
                        map_hint = gr.Markdown("Build the world first.", elem_classes="hint")
                        with gr.Row():
                            center_x = gr.Number(value=0, label="Centre X", min_width=90)
                            center_y = gr.Number(value=0, label="Centre Y", min_width=90)
                            size = gr.Number(value=defaults["size"], minimum=0.5, label="Size", min_width=90)
                        with gr.Row():
                            rotation = gr.Number(value=0, label="Rotation °", min_width=90)
                            width_mm = gr.Number(value=defaults["width_mm"], minimum=5, label="Width (mm)", min_width=90)
                            base_mm = gr.Number(value=defaults["base_mm"], minimum=0.5, label="Base (mm)", min_width=90)
                        tile_button = gr.Button("Cut tile", variant="primary")
                        with gr.Accordion("Advanced", open=False):
                            faces = gr.Radio(list(CONFIG["print"]["faces"]), label="Triangle budget",
                                             value=face_labels[CONFIG["print"]["defaults"]["target_faces"]])
                            formats = gr.CheckboxGroup(["stl", "3mf"], value=["stl", "3mf"], label="Formats")
                            grid = gr.Slider(100, 1000, value=defaults["grid"], step=50, label="Height samples per side")
                            exaggeration = gr.Slider(0.5, 4, value=defaults["exaggeration"], step=0.1,
                                                     label="Height exaggeration")
                            smoothing = gr.Slider(0, 3, value=defaults["smoothing"], step=0.1, label="Smoothing")

                runs = gr.Dropdown(choices=[], label="Open a previous world", elem_id="runs")

            with gr.Column(scale=1, elem_id="result"):
                stage = gr.Radio([], show_label=False, container=False, visible=False, elem_id="stage")
                panorama_view = gr.Image(show_label=False, interactive=False, visible=False, elem_id="panorama")
                viewer = gr.Model3D(show_label=False, interactive=False, elem_id="viewer", **WORLD_VIEWER_STYLE)
                status = gr.Markdown(elem_id="status")
                downloads = gr.File(show_label=False, file_count="multiple", interactive=False, visible=False,
                                    elem_id="downloads")
                with gr.Accordion("Details", open=False):
                    details = gr.JSON(show_label=False)
                    log = gr.Textbox(lines=12, max_lines=12, autoscroll=True, show_label=False)

        inputs.update(
            source=source, image=image, prompt=prompt, model=model, seed=seed, labels_fg1=labels_fg1,
            labels_fg2=labels_fg2, classes=classes, center_x=center_x, center_y=center_y, size=size,
            rotation=rotation, width_mm=width_mm, base_mm=base_mm, faces=faces, formats=formats, grid=grid,
            exaggeration=exaggeration, smoothing=smoothing,
        )
        shown = [session, tabs, status, log, panorama_view, preview, stage, downloads, details, runs]
        tile_controls = [session, center_x, center_y, size, rotation]

        def show(state: dict, tab: int, line: str) -> dict:
            record = RUNNER.load(state["world"]) if state["world"] else None
            stages = {PANORAMA: None, **RUNNER.previews(record)} if record else {}
            selected = next(reversed(stages), None)
            files = RUNNER.files(record) if record else []
            return {
                session: state, tabs: gr.Tabs(selected=tab), status: line, log: WORKERS.tail(WORKERS.current),
                panorama_view: record["panorama"] if record else None,
                preview: stages.get(selected),
                stage: gr.Radio(choices=list(stages), value=selected, visible=len(stages) > 1),
                downloads: gr.File(value=files, visible=bool(files)),
                details: record,
                runs: gr.Dropdown(choices=_history_choices(), value=state["world"]),
            }

        def swap_view(event):
            """The right pane shows the panorama until a stage has a mesh to show.

            Model3D stays blank if a generator feeds it, or if its value comes with its visibility.
            """
            def view(path: str | None, state: dict) -> tuple:
                flat = bool(state["world"]) and path is None
                return gr.Image(visible=flat), gr.Model3D(visible=not flat)

            return event.then(view, [preview, session], [panorama_view, viewer], show_progress="hidden").then(
                lambda path: path, preview, viewer, show_progress="hidden")

        def refresh(event) -> None:
            swap_view(event).then(_map, tile_controls, [map_view, map_hint], show_progress="hidden")

        def on_click(button: gr.Button, step: Step) -> None:
            def handler(*values):
                v = dict(zip(inputs, values))
                state = dict(v["session"])
                try:
                    summary, seconds = yield from watch(step.title, lambda say: step.work(v, state, say),
                                                        status, log, WORKERS)
                except Exception as exc:
                    first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                    yield {status: f"✕ **{step.title}:** {first_line}", log: WORKERS.tail(WORKERS.current)}
                    return
                # The panorama shows in its own tab, so stay there; the viewer shows the rest.
                following = step.tab if step.tab == 1 else min(step.tab + 1, len(STEPS))
                yield show(state, following, f"✓ **{step.title}** · {seconds:.0f}s · {summary}")

            refresh(button.click(handler, list(inputs.values()), shown, api_name=step.title.lower(),
                                 show_progress="hidden", concurrency_limit=1, concurrency_id="gpu"))

        for button, step in zip((panorama_button, world_button, tile_button), STEPS):
            on_click(button, step)

        def show_source(selected: str, chosen: str) -> dict:
            hint = chosen if selected == "image" else _text_model() if selected == "text" else None
            placeholder = {"image": "Drop an image", "upload": "Drop a 2:1 panorama"}.get(selected)
            return {
                image: gr.Image(visible=selected != "text", placeholder=placeholder),
                prompt: gr.Textbox(visible=selected != "upload",
                                   label="Describe the scene" + (" (optional)" if selected == "image" else "")),
                model: gr.Radio(visible=selected == "image"),
                **{panel: gr.Column(visible=key == hint) for key, panel in model_hints.items()},
            }

        chooser = [image, prompt, model, *model_hints.values()]
        source.input(show_source, [source, model], chooser, show_progress="hidden")
        model.input(show_source, [source, model], chooser, show_progress="hidden")

        def open_world(world_dir: str) -> dict:
            tab = 3 if RUNNER.load(world_dir).get("scene") else 2
            return show({"world": world_dir}, tab, f"Opened {Path(world_dir).name}")

        refresh(runs.input(open_world, runs, shown, show_progress="hidden"))
        swap_view(stage.input(lambda state, name: RUNNER.previews(RUNNER.load(state["world"])).get(name),
                              [session, stage], preview, show_progress="hidden"))
        dice.click(lambda: random.randint(0, 2**31 - 1), None, seed, show_progress="hidden")
        map_view.select(_pick, session, [center_x, center_y], show_progress="hidden")
        for control in (center_x, center_y, size, rotation):
            control.change(_map, tile_controls, [map_view, map_hint], show_progress="hidden")
        demo.load(lambda: gr.Dropdown(choices=_history_choices()), None, runs)
    return demo


if __name__ == "__main__":
    launch(build())
