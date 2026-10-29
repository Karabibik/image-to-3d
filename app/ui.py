"""Gradio UI: upload character art, pick a model, download a print-ready miniature."""

from __future__ import annotations

import atexit
import json
import os
import random
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr

from .orchestrator import Job, Runner, WorkerManager, load_config
from .printprep import PrintSettings

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace" if Path("/workspace").is_dir() else Path.cwd()))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DATA_DIR / "outputs"))
LOG_DIR = Path(os.environ.get("LOG_DIR", DATA_DIR / "logs"))
NO_REFINER = "none"
ROTATIONS = [0, 90, 180, 270]
VIEWER_STYLE = {"display_mode": "solid", "clear_color": (0.15, 0.15, 0.17, 1.0)}

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


def _model_panels(models: dict, selected: str) -> tuple[dict, list]:
    """One collapsible settings group per model; only the selected model's group is visible."""
    groups, controls = {}, []
    for key, spec in models.items():
        status = "" if WORKERS.available(spec["worker"]) else " — **not installed in this image**"
        with gr.Group(visible=key == selected) as group:
            gr.Markdown(f"**{spec['label']}**{status}  \n{spec.get('notes', '')}")
            advanced = []
            for name, param in spec["params"].items():
                if param.get("advanced"):
                    advanced.append((name, param))
                else:
                    controls.append((key, name, _control(param)))
            if advanced:
                with gr.Accordion("Advanced", open=False):
                    for name, param in advanced:
                        controls.append((key, name, _control(param)))
        groups[key] = group
    return groups, controls


def _collect(controls: list, values: list, key: str | None) -> dict:
    return {name: value for (model, name, _), value in zip(controls, values) if model == key}


def _print_settings(height, faces_label, flatten, min_ratio, rot_x, rot_z, remesh, cell, formats) -> PrintSettings:
    faces = CONFIG["print"]["faces"].get(faces_label, faces_label)
    return PrintSettings(
        target_height_mm=float(height) if height else None,
        target_faces=int(faces) or None,
        flatten_base_mm=float(flatten),
        min_component_ratio=float(min_ratio),
        rotate_x_deg=float(rot_x),
        rotate_z_deg=float(rot_z),
        remesh_if_open=bool(remesh),
        remesh_cell_mm=float(cell),
        formats=tuple(formats) or ("stl",),
    )


def _files(report) -> list[str]:
    return [path for key, path in report.files.items() if key != "preview"]


# ---------------------------------------------------------------- handlers

def _make_handlers(gen_controls, ref_controls):
    n_gen, n_ref = len(gen_controls), len(ref_controls)

    def run_batch(files, generator, refiner, seed, randomize, rembg_model, *values):
        if not files:
            raise gr.Error("Upload at least one image.")
        gen_values, ref_values, print_values = values[:n_gen], values[n_gen:n_gen + n_ref], values[n_gen + n_ref:]
        refiner = None if refiner == NO_REFINER else refiner
        settings = _print_settings(*print_values)
        gen_params = _collect(gen_controls, gen_values, generator)
        ref_params = _collect(ref_controls, ref_values, refiner)

        reports, downloads, preview, lines = {}, [], None, []
        for index, file in enumerate(files, start=1):
            path = Path(file if isinstance(file, str) else file.name)
            job = Job(
                image=path, generator=generator, refiner=refiner,
                seed=random.randint(0, 2**31 - 1) if randomize else int(seed),
                gen_params=gen_params, ref_params=ref_params,
                print_settings=settings, rembg_model=rembg_model,
            )
            state = {"status": "Queued", "result": None, "error": None}

            def work(job=job, state=state):
                try:
                    state["result"] = RUNNER.run(job, status=lambda msg: state.__setitem__("status", msg))
                except Exception as exc:  # shown in the UI, batch continues
                    state["error"] = exc

            thread = threading.Thread(target=work, daemon=True)
            started = time.time()
            thread.start()
            while thread.is_alive():
                header = f"**{index}/{len(files)} · {path.name}** · {state['status']} · {time.time() - started:.0f}s"
                # Regular yields also keep the RunPod proxy connection alive on long jobs.
                yield gr.skip(), "\n\n".join([*lines, header]), gr.skip(), gr.skip(), WORKERS.tail(WORKERS.current)
                thread.join(timeout=1.5)

            if state["error"] is not None:
                lines.append(f"❌ **{path.name}** failed after {time.time() - started:.0f}s: `{state['error']}`")
                continue
            result = state["result"]
            reports[result.record["name"]] = result.report.as_dict()
            downloads += _files(result.report)
            preview = result.report.files["preview"]
            watertight = "watertight" if result.report.watertight else "⚠ not watertight"
            size = " × ".join(f"{v:.1f}" for v in result.report.size_mm)
            lines.append(
                f"✅ **{path.name}** · {time.time() - started:.0f}s · {result.report.faces:,} faces · "
                f"{size} mm · {watertight} · seed {job.seed}"
            )

        if len(downloads) > 2:
            batch_zip = OUTPUT_DIR / "batches" / f"batch_{datetime.now():%Y%m%d_%H%M%S}.zip"
            batch_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(batch_zip, "w", zipfile.ZIP_DEFLATED) as archive:
                for item in downloads:
                    archive.write(item, arcname=Path(item).name)
            downloads.insert(0, str(batch_zip))
        report = next(iter(reports.values())) if len(reports) == 1 else reports
        yield preview, "\n\n".join(lines), report, downloads, WORKERS.tail(WORKERS.current)

    return run_batch


def _history_choices() -> list[tuple[str, str]]:
    choices = []
    for run_dir in RUNNER.history():
        try:
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stages = record["generator"]["key"] + (f" + {record['refiner']['key']}" if record.get("refiner") else "")
        label = f"{record['created'].replace('T', ' ')} · {record['name']} · {stages} · seed {record['seed']}"
        choices.append((label, str(run_dir)))
    return choices


def _show_run(run_dir: str):
    if not run_dir:
        return None, None, None, []
    record = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    files = record["report"]["files"]
    downloads = sorted(str(p) for p in Path(run_dir).iterdir() if p.suffix in (".stl", ".3mf"))
    downloads.append(record["final_mesh"])
    return files.get("preview"), str(Path(run_dir) / "input.png"), record, downloads


def _reprint(run_dir: str, *print_values):
    if not run_dir:
        raise gr.Error("Pick a run first.")
    report = RUNNER.reprint(Path(run_dir), _print_settings(*print_values))
    return report.files["preview"], report.as_dict(), _files(report)


# ---------------------------------------------------------------- layout

def build() -> gr.Blocks:
    generators, refiners = CONFIG["generators"], CONFIG["refiners"]
    defaults = CONFIG["print"]["defaults"]
    face_labels = {v: k for k, v in CONFIG["print"]["faces"].items()}
    default_gen = next(iter(generators))

    with gr.Blocks(title="Image to Miniature") as demo:
        gr.Markdown("## Image → printable miniature\nUpload character art, pick a model, download STL/3MF.")
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1, min_width=360):
                    images = gr.File(
                        label="Images (PNG with transparency is used as-is; several = batch)",
                        file_count="multiple", file_types=["image"],
                    )
                    generator = gr.Dropdown(
                        [(g["label"], k) for k, g in generators.items()], value=default_gen, label="Generator"
                    )
                    gen_groups, gen_controls = _model_panels(generators, default_gen)
                    refiner = gr.Dropdown(
                        [("None", NO_REFINER)] + [(r["label"], k) for k, r in refiners.items()],
                        value=NO_REFINER, label="Refiner",
                    )
                    ref_groups, ref_controls = _model_panels(refiners, NO_REFINER)
                    with gr.Row():
                        seed = gr.Number(value=1234, precision=0, label="Seed")
                        randomize = gr.Checkbox(value=False, label="Random seed per image")

                    with gr.Accordion("Print settings", open=True):
                        height = gr.Number(
                            value=defaults["target_height_mm"], label="Height (mm)",
                            info="Common scales: " + ", ".join(f"{h} mm" for h in CONFIG["print"]["heights_mm"]),
                        )
                        faces = gr.Radio(
                            list(CONFIG["print"]["faces"]), value=face_labels[defaults["target_faces"]],
                            label="Triangle budget",
                        )
                        flatten = gr.Slider(
                            0, 2, value=defaults["flatten_base_mm"], step=0.1,
                            label="Flatten base (mm cut off the bottom, 0 = off)",
                        )
                        with gr.Accordion("More", open=False):
                            min_ratio = gr.Slider(0, 0.2, value=defaults["min_component_ratio"], step=0.01,
                                                  label="Remove loose parts smaller than this share of the main body")
                            rot_x = gr.Radio(ROTATIONS, value=0, label="Extra rotation around X (if the result lies down)")
                            rot_z = gr.Radio(ROTATIONS, value=0, label="Extra rotation around Z")
                            remesh = gr.Checkbox(
                                value=defaults["remesh_if_open"],
                                label="Voxel-remesh if the mesh is not watertight (TRELLIS.2/Pixal3D; adds a few minutes)",
                            )
                            cell = gr.Slider(
                                0.02, 0.3, value=defaults["remesh_cell_mm"], step=0.01, label="Remesh voxel size (mm)"
                            )
                            formats = gr.CheckboxGroup(["stl", "3mf"], value=["stl", "3mf"], label="Formats")
                            rembg_model = gr.Dropdown(
                                CONFIG["background_removal"]["models"], value=CONFIG["background_removal"]["default"],
                                label="Background removal model (only for images without transparency)",
                            )
                    run = gr.Button("Generate", variant="primary")

                with gr.Column(scale=2):
                    viewer = gr.Model3D(label="Print-ready mesh", height=560, **VIEWER_STYLE)
                    status = gr.Markdown()
                    downloads = gr.File(label="Downloads", file_count="multiple")
                    with gr.Accordion("Report", open=False):
                        report = gr.JSON()
                    with gr.Accordion("Worker log", open=False):
                        log = gr.Textbox(lines=14, max_lines=14, autoscroll=True, show_label=False)

        print_inputs = [height, faces, flatten, min_ratio, rot_x, rot_z, remesh, cell, formats]

        with gr.Tab("History"):
            with gr.Row():
                runs = gr.Dropdown(choices=[], label="Previous runs", scale=4)
                refresh = gr.Button("Refresh", scale=1)
            with gr.Row():
                with gr.Column(scale=1):
                    history_input = gr.Image(label="Input", type="filepath", interactive=False, height=320)
                    reprint = gr.Button("Re-run print prep with the current print settings")
                    history_files = gr.File(label="Files", file_count="multiple")
                with gr.Column(scale=2):
                    history_view = gr.Model3D(label="Preview", height=480, **VIEWER_STYLE)
                    history_record = gr.JSON(label="Run record")

        generator.change(lambda key: [gr.update(visible=k == key) for k in gen_groups], generator, list(gen_groups.values()))
        refiner.change(lambda key: [gr.update(visible=k == key) for k in ref_groups], refiner, list(ref_groups.values()))
        run.click(
            _make_handlers(gen_controls, ref_controls),
            inputs=[images, generator, refiner, seed, randomize, rembg_model,
                    *[c for *_, c in gen_controls], *[c for *_, c in ref_controls], *print_inputs],
            outputs=[viewer, status, report, downloads, log],
        )
        refresh.click(lambda: gr.update(choices=_history_choices()), None, runs)
        demo.load(lambda: gr.update(choices=_history_choices()), None, runs)
        runs.change(_show_run, runs, [history_view, history_input, history_record, history_files])
        reprint.click(_reprint, [runs, *print_inputs], [history_view, history_record, history_files])
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
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
