"""Gradio UI: take character art through five steps to a print-ready miniature."""

from __future__ import annotations

import atexit
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import gradio as gr
from PIL import Image

from .imageprep import prepare_input
from .multiview import SLOT_LABELS, SLOTS, split_sheet
from .orchestrator import Prepared, Runner, Status, WorkerManager, load_config, safe_name
from .printprep import PrintSettings

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace" if Path("/workspace").is_dir() else Path.cwd()))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DATA_DIR / "outputs"))
LOG_DIR = Path(os.environ.get("LOG_DIR", DATA_DIR / "logs"))
CSS_PATH = Path(__file__).with_name("ui.css")
NO_REFINER = "none"
ROTATIONS = [0, 90, 180, 270]
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

CONFIG = load_config()
WORKERS = WorkerManager(CONFIG["workers"], LOG_DIR, mock=os.environ.get("MOCK_WORKERS") == "1")
RUNNER = Runner(CONFIG, OUTPUT_DIR, WORKERS)
atexit.register(WORKERS.stop)


# ---------------------------------------------------------------- controls

def _control(spec: dict) -> gr.components.Component:
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


def _model_panels(models: dict, selected: str, prefix: str, inputs: dict) -> dict:
    """One hint plus an Advanced panel per model; only the selected model's panel is visible."""
    panels = {}
    for key, spec in models.items():
        missing = "" if WORKERS.available(spec["worker"]) else " **Not installed in this image.**"
        with gr.Column(visible=key == selected) as panel:
            gr.Markdown(f"{spec.get('hint', '')}{missing}", elem_classes="hint")
            with gr.Accordion("Advanced", open=False):
                for name, param in spec["params"].items():
                    inputs[f"{prefix}.{key}.{name}"] = _control(param)
        panels[key] = panel
    return panels


def _params(values: dict, prefix: str, key: str) -> dict:
    start = f"{prefix}.{key}."
    return {name[len(start):]: value for name, value in values.items() if name.startswith(start)}


def _print_settings(v: dict) -> PrintSettings:
    faces = CONFIG["print"]["faces"].get(v["faces"], v["faces"])
    return PrintSettings(
        target_height_mm=float(v["height"]) if v["height"] else None,
        target_faces=int(faces) or None,
        flatten_base_mm=float(v["flatten"]),
        min_component_ratio=float(v["min_ratio"]),
        rotate_x_deg=float(v["rot_x"]),
        rotate_z_deg=float(v["rot_z"]),
        remesh_if_open=bool(v["remesh"]),
        remesh_cell_mm=float(v["cell"]),
        formats=tuple(v["formats"]) or ("stl",),
    )


# ---------------------------------------------------------------- steps

@dataclass(frozen=True)
class Step:
    tab: int
    title: str
    # Viewer stage the step produces, see Runner.previews.
    stage: str | None
    # Takes the control values, the session and a status callback; returns a summary line.
    work: Callable[[dict, dict, Status], str]


def _generator_for(mode: str) -> str:
    return next(key for key, spec in CONFIG["generators"].items() if spec.get("input", "single") == mode)


def _split_sheet(sheet: str, order: str, rembg_model: str, say: Status) -> tuple[dict[str, Path], str]:
    names = [name.strip().lower() for name in order.replace(";", ",").split(",") if name.strip()]
    unknown = [name for name in names if name not in (*SLOTS, "skip")]
    if unknown:
        raise ValueError(f"Unknown view name(s) {', '.join(unknown)}; use {', '.join(SLOTS)} or skip.")
    say("Splitting the sheet")
    folder = OUTPUT_DIR / "sheets" / f"{datetime.now():%Y%m%d_%H%M%S}_{safe_name(Path(sheet).stem)}"
    cutout, _ = prepare_input(sheet, folder / "sheet_cutout.png", rembg_model)
    crops = split_sheet(Image.open(cutout))
    views = {}
    for name, crop in zip(names, crops):
        if name in SLOTS:
            views[name] = folder / f"{name}.png"
            crop.save(views[name])
    note = f"found {len(crops)} figure(s), used {len(views)}"
    if len(crops) != len(names):
        note += f", but the sheet order lists {len(names)}: check the views"
    return views, note


def _step_image(v: dict, session: dict, say: Status) -> str:
    if v["mode"] == "multiview":
        if v["sheet"]:
            views, note = _split_sheet(v["sheet"], v["order"], v["rembg_model"], say)
        else:
            views = {slot: Path(v[slot]) for slot in SLOTS if v[slot]}
            note = f"{len(views)} view(s)"
        if "front" not in views:
            raise ValueError("Add a turnaround sheet, or at least the front view.")
        name = safe_name(Path(v["sheet"]).stem) if v["sheet"] else ""
        prepared = RUNNER.prepare_image(views["front"], v["rembg_model"], views, name, say)
    else:
        if not v["image"]:
            raise ValueError("Upload an image first.")
        prepared = RUNNER.prepare_image(Path(v["image"]), v["rembg_model"], status=say)
        note = "background removed" if prepared.background_removed else "transparency kept"
    session.update(prepared=prepared, run=None)
    return note


def _step_shape(v: dict, session: dict, say: Status) -> str:
    if session["prepared"] is None:
        _step_image(v, session, say)
    prepared = session["prepared"]
    generator = _generator_for("multiview" if prepared.views else "single")
    seed = int(v["seed"])
    session["run"] = str(RUNNER.generate(prepared, generator, seed, _params(v, "gen", generator), say))
    return f"{_faces(RUNNER.load(session['run'])['generator'])}seed {seed}"


def _need_run(session: dict) -> str:
    if not session["run"]:
        raise ValueError("Generate a shape first.")
    return session["run"]


def _step_refine(v: dict, session: dict, say: Status) -> str:
    run_dir = _need_run(session)
    if v["refiner"] == NO_REFINER:
        raise ValueError("Pick a refiner, or go on to the next step.")
    record = RUNNER.refine(run_dir, v["refiner"], _params(v, "ref", v["refiner"]), say)
    return f"{_faces(record['refiner'])}{record['refiner']['key']}"


def _step_color(v: dict, session: dict, say: Status) -> str:
    run_dir = _need_run(session)
    painter = next(iter(CONFIG["painters"]))
    params = _params(v, "paint", painter)
    RUNNER.paint(run_dir, painter, params, say)
    return f"{int(params['faces']):,} faces · textured.glb"


def _step_print(v: dict, session: dict, say: Status) -> str:
    report = RUNNER.print_ready(_need_run(session), _print_settings(v), say)
    size = " × ".join(f"{value:.1f}" for value in report.size_mm)
    parts = [f"{report.faces:,} faces", f"{size} mm"]
    parts.append("watertight" if report.watertight else "⚠ not watertight")
    if report.volume_cm3:
        parts.append(f"{report.volume_cm3:g} cm³")
    return " · ".join(parts + [f"⚠ {warning}" for warning in report.warnings if report.watertight])


def _faces(stage: dict) -> str:
    faces = stage.get("info", {}).get("faces")
    return f"{faces:,} faces · " if faces else ""


STEPS = [
    Step(1, "Image", None, _step_image),
    Step(2, "Shape", "Shape", _step_shape),
    Step(3, "Refine", "Refined", _step_refine),
    Step(4, "Color", "Color", _step_color),
    Step(5, "Print", "Print", _step_print),
]


def _steps_from(v: dict) -> list[Step]:
    """What Run all does: the current step and later, without the steps that are left off or done."""
    skipped = {3} if v["refiner"] == NO_REFINER else set()
    if not v["include_color"]:
        skipped.add(4)
    if v["session"]["prepared"]:
        skipped.add(1)
    return [step for step in STEPS[v["current"] - 1:] if step.tab not in skipped]


# ---------------------------------------------------------------- results

def _history_choices() -> list[tuple[str, str]]:
    choices = []
    for run_dir in RUNNER.history():
        try:
            record = RUNNER.load(run_dir)
        except (OSError, json.JSONDecodeError):
            continue
        stages = record["generator"]["key"] + (f" + {record['refiner']['key']}" if record.get("refiner") else "")
        if record.get("paint"):
            stages += " + color"
        created = record["created"][5:16].replace("T", " ")
        choices.append((f"{created} · {record['name']} · {stages} · s{record['seed']}", str(run_dir)))
    return choices


def _gallery(prepared: Prepared | None) -> list[tuple[str, str | None]]:
    if prepared is None:
        return []
    if prepared.views:
        return [(str(path), SLOT_LABELS[slot]) for slot, path in prepared.views.items()]
    return [(str(prepared.image), None)]


def _run_files(run_dir: str, record: dict) -> list[str]:
    folder = Path(run_dir)
    files = sorted(str(path) for path in folder.iterdir() if path.suffix in (".stl", ".3mf"))
    if record.get("paint"):
        files.append(record["paint"]["mesh"])
    return [*files, record["final_mesh"]]


# ---------------------------------------------------------------- layout

def build() -> gr.Blocks:
    generators, refiners, painters = CONFIG["generators"], CONFIG["refiners"], CONFIG["painters"]
    defaults = CONFIG["print"]["defaults"]
    face_labels = {v: k for k, v in CONFIG["print"]["faces"].items()}
    default_gen = _generator_for("single")
    inputs: dict[str, gr.components.Component] = {}

    with gr.Blocks(title="Image to Miniature", fill_width=True) as demo:
        session = gr.State({"prepared": None, "run": None})
        current = gr.State(1)
        preview = gr.State()
        inputs.update(session=session, current=current)

        with gr.Row(elem_id="workspace", equal_height=False):
            with gr.Column(scale=0, min_width=380, elem_id="controls"):
                with gr.Tabs(selected=1, overflow_behavior="wrap") as tabs:
                    with gr.Tab("1 Image", id=1) as tab_image:
                        mode = gr.Radio(
                            [("One image", "single"), ("Turnaround", "multiview")],
                            value="single", show_label=False, container=False,
                        )
                        with gr.Column(visible=True) as single_panel:
                            image = gr.Image(height=240, placeholder="Drop an image", **UPLOAD_STYLE)
                        with gr.Column(visible=False) as turnaround_panel:
                            sheet = gr.Image(height=130, placeholder="Drop a sheet with the figures side by side",
                                             **UPLOAD_STYLE)
                            gr.Markdown("Or one image per view. Only the front is required. Leave out 3/4 shots: "
                                        "views must be straight-on.", elem_classes="hint")
                            with gr.Row():
                                view_slots = [
                                    gr.Image(height=96, min_width=80, placeholder=slot.capitalize(), **UPLOAD_STYLE)
                                    for slot in SLOTS
                                ]
                        cutouts = gr.Gallery(show_label=False, columns=4, fit_columns=False, rows=1, height=120,
                                             object_fit="contain", allow_preview=False, interactive=False,
                                             visible=False)
                        prepare_button = gr.Button("Prepare", variant="primary")
                        with gr.Accordion("Advanced", open=False):
                            order = gr.Textbox(
                                value="front, left, back, right", label="Views on the sheet, left to right",
                                info="front / left / back / right, or skip. Left = the figure faces left on the page.",
                                visible=False,
                            )
                            rembg_model = gr.Dropdown(
                                CONFIG["background_removal"]["models"], value=CONFIG["background_removal"]["default"],
                                label="Background removal model", info="Only for images without transparency",
                            )

                    with gr.Tab("2 Shape", id=2) as tab_shape:
                        with gr.Row():
                            seed = gr.Number(value=1234, precision=0, label="Seed", scale=2, min_width=100)
                            dice = gr.Button("Random seed", scale=1, min_width=100, elem_id="dice")
                        shape_button = gr.Button("Generate", variant="primary")
                        gen_panels = _model_panels(generators, default_gen, "gen", inputs)

                    with gr.Tab("3 Refine", id=3) as tab_refine:
                        refiner = gr.Radio(
                            [("None", NO_REFINER)] + [(spec["label"], key) for key, spec in refiners.items()],
                            value=NO_REFINER, show_label=False, container=False,
                        )
                        refine_button = gr.Button("Refine", variant="primary")
                        ref_panels = _model_panels(refiners, NO_REFINER, "ref", inputs)

                    with gr.Tab("4 Color", id=4) as tab_color:
                        include_color = gr.Checkbox(value=False, label="Include in Run all steps")
                        color_button = gr.Button("Paint", variant="primary")
                        _model_panels(painters, next(iter(painters)), "paint", inputs)

                    with gr.Tab("5 Print", id=5) as tab_print:
                        with gr.Row():
                            height = gr.Number(value=defaults["target_height_mm"], label="Height (mm)", min_width=100)
                            formats = gr.CheckboxGroup(["stl", "3mf"], value=["stl", "3mf"], label="Formats",
                                                       min_width=100)
                        faces = gr.Radio(list(CONFIG["print"]["faces"]), value=face_labels[defaults["target_faces"]],
                                         label="Triangle budget")
                        print_button = gr.Button("Prepare for print", variant="primary")
                        with gr.Accordion("Advanced", open=False):
                            flatten = gr.Slider(0, 2, value=defaults["flatten_base_mm"], step=0.1,
                                                label="Flatten base (mm cut off the bottom)")
                            min_ratio = gr.Slider(0, 0.2, value=defaults["min_component_ratio"], step=0.01,
                                                  label="Remove loose parts below this share of the main body")
                            rot_x = gr.Radio(ROTATIONS, value=0, label="Extra rotation around X")
                            rot_z = gr.Radio(ROTATIONS, value=0, label="Extra rotation around Z")
                            remesh = gr.Checkbox(value=defaults["remesh_if_open"],
                                                 label="Rebuild as a solid if not watertight")
                            cell = gr.Slider(0.02, 0.3, value=defaults["remesh_cell_mm"], step=0.01,
                                             label="Solid rebuild voxel size (mm)")

                run_all = gr.Button("Run all steps")
                runs = gr.Dropdown(choices=[], label="Open a previous run", elem_id="runs")

            with gr.Column(scale=1, elem_id="result"):
                stage = gr.Radio([], show_label=False, container=False, visible=False, elem_id="stage")
                viewer = gr.Model3D(show_label=False, interactive=False, elem_id="viewer", **VIEWER_STYLE)
                status = gr.Markdown(elem_id="status")
                downloads = gr.File(show_label=False, file_count="multiple", interactive=False, visible=False,
                                    elem_id="downloads")
                with gr.Accordion("Details", open=False):
                    details = gr.JSON(show_label=False)
                    log = gr.Textbox(lines=12, max_lines=12, autoscroll=True, show_label=False)

        inputs.update(
            mode=mode, image=image, sheet=sheet, order=order, rembg_model=rembg_model, seed=seed,
            refiner=refiner, include_color=include_color, height=height, formats=formats, faces=faces,
            flatten=flatten, min_ratio=min_ratio, rot_x=rot_x, rot_z=rot_z, remesh=remesh, cell=cell,
            **dict(zip(SLOTS, view_slots)),
        )
        shown = [session, current, tabs, status, log, cutouts, preview, stage, downloads, details, runs,
                 mode, single_panel, turnaround_panel, order, *gen_panels.values()]

        def show(state: dict, tab: int, line: str, stage_name: str | None = None) -> dict:
            record = RUNNER.load(state["run"]) if state["run"] else None
            stages = RUNNER.previews(record) if record else {}
            selected = stage_name if stage_name in stages else next(reversed(stages), None)
            items = _gallery(state["prepared"])
            files = _run_files(state["run"], record) if record else []
            return {
                session: state, current: tab, tabs: gr.Tabs(selected=tab), status: line,
                log: WORKERS.tail(WORKERS.current),
                cutouts: gr.Gallery(value=items, visible=bool(items)),
                preview: stages.get(selected),
                stage: gr.Radio(choices=list(stages), value=selected, visible=len(stages) > 1),
                downloads: gr.File(value=files, visible=bool(files)),
                details: record,
                runs: gr.Dropdown(choices=_history_choices(), value=state["run"]),
            }

        def show_mode(selected: str) -> dict:
            multiview = selected == "multiview"
            shape_model = _generator_for(selected)
            return {
                mode: selected, single_panel: gr.Column(visible=not multiview),
                turnaround_panel: gr.Column(visible=multiview), order: gr.Textbox(visible=multiview),
                **{panel: gr.Column(visible=key == shape_model) for key, panel in gen_panels.items()},
            }

        def watch(title: str, work: Callable[[Status], str]):
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
                       log: WORKERS.tail(WORKERS.current)}
                thread.join(timeout=1.5)
            if progress["error"] is not None:
                raise progress["error"]
            return progress["result"], time.time() - started

        def run_steps(steps: list[Step], v: dict):
            state = dict(v["session"])
            for step in steps:
                try:
                    summary, seconds = yield from watch(step.title, lambda say: step.work(v, state, say))
                except Exception as exc:
                    first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                    yield {status: f"✕ **{step.title}:** {first_line}", log: WORKERS.tail(WORKERS.current)}
                    return
                # The Image step shows its result in its own tab, so stay there; the viewer shows the rest.
                following = step.tab if step.stage is None else min(step.tab + 1, len(STEPS))
                yield show(state, following, f"✓ **{step.title}** · {seconds:.0f}s · {summary}", step.stage)

        def on_click(button: gr.Button, name: str, choose: Callable[[dict], list[Step]]) -> None:
            def handler(*values):
                v = dict(zip(inputs, values))
                yield from run_steps(choose(v), v)

            button.click(
                handler, list(inputs.values()), shown, api_name=name, show_progress="hidden",
                concurrency_limit=1, concurrency_id="gpu",
            ).then(lambda path: path, preview, viewer, show_progress="hidden")  # Model3D stays blank if a generator feeds it

        for button, step in zip((prepare_button, shape_button, refine_button, color_button, print_button), STEPS):
            on_click(button, step.title.lower(), lambda _v, step=step: [step])
        on_click(run_all, "run_all", _steps_from)

        def forget_run(selected: str) -> dict:
            return {**show({"prepared": None, "run": None}, 1, ""), **show_mode(selected)}

        for event in (mode.input, *(e for s in (image, sheet, *view_slots) for e in (s.upload, s.clear))):
            event(forget_run, mode, shown, show_progress="hidden").then(
                lambda path: path, preview, viewer, show_progress="hidden")

        def open_run(run_dir: str) -> dict:
            prepared = RUNNER.prepared_from(run_dir)
            selected = "multiview" if prepared.views else "single"
            state = {"prepared": prepared, "run": run_dir}
            return {**show(state, len(STEPS), f"Opened {Path(run_dir).name}"), **show_mode(selected)}

        runs.input(open_run, runs, shown, show_progress="hidden").then(
            lambda path: path, preview, viewer, show_progress="hidden")
        stage.input(lambda state, name: RUNNER.previews(RUNNER.load(state["run"]))[name], [session, stage], viewer,
                    show_progress="hidden")
        refiner.input(lambda key: [gr.Column(visible=k == key) for k in ref_panels], refiner, list(ref_panels.values()),
                      show_progress="hidden")
        dice.click(lambda: random.randint(0, 2**31 - 1), None, seed, show_progress="hidden")
        for tab, step in zip((tab_image, tab_shape, tab_refine, tab_color, tab_print), STEPS):
            tab.select(lambda number=step.tab: number, None, current, show_progress="hidden")
        demo.load(lambda: gr.Dropdown(choices=_history_choices()), None, runs)
    return demo


def main() -> None:
    password = os.environ.get("UI_PASSWORD")
    auth = (os.environ.get("UI_USERNAME", "admin"), password) if password else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build().queue(default_concurrency_limit=1).launch(
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


if __name__ == "__main__":
    main()
