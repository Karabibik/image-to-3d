# Image → printable miniature (RunPod template)

Take character art through five steps, **Image, Shape, Refine, Color, Print**, and download a print-ready STL/3MF:
watertight, standing on Z, scaled to a height in millimetres and reduced to a triangle budget.
Refine and Color are optional, and Color also gives you a textured GLB.
Everything runs on a rented RunPod GPU behind one web UI.

| Model | Role | Peak VRAM, measured on an A40 | License |
|---|---|---|---|
| **Hunyuan3D 2.1** | generator from one image, shape only | 11.5 GB | Tencent Hunyuan Community (excludes EU, UK, South Korea) |
| **Hunyuan3D-2mv** | generator from turnaround views (front/left/back/right), the best results | 9.5 GB, Turbo 6.1 GB | Tencent Hunyuan 3D 2.0 Community (excludes EU, UK, South Korea) |
| **UltraShape 1.0** | refiner, the sharpest detail | 15.0 GB | Apache-2.0 |
| **Hi3D v1.0** *(experimental)* | refiner, up to 1536 | 20.4 GB at 1024, 10.9 GB with CPU offload (1536 only tested upstream on 96 GB) | Hi3D Open License |
| **Hunyuan3D-Paint 2.1** | color: textured GLB with PBR maps | Upstream says 21 GB; not measured here yet | Tencent Hunyuan Community (excludes EU, UK, South Korea) |

These peaks are what PyTorch allocated; the card has to hold more (UltraShape occupied 18.4 GB of an A40).

TRELLIS.2 and Pixal3D were removed in v0.3.0 after side-by-side tests on painted-mini renders; tag `v0.2.1` still has them.

Timings on an A40 for one model, excluding print prep: Hunyuan3D 44 s (about 100 s on first use), Hunyuan3D-2mv about 2 minutes, UltraShape about 10 minutes at 50 steps
(use 12 steps for a quick pass), Hi3D at 1024 about 7 minutes, or 11 with CPU offload.
Whole runs from the image to the STL, with default settings: Hunyuan3D alone about 4 minutes, with Hi3D about 12, with UltraShape about 17, Hunyuan3D-2mv with UltraShape about 18.

Meshes that are still open or non-manifold after repair are rebuilt by print prep as a solid at 0.05 mm
(on by default, about 1-2 minutes of CPU time). Hi3D needs this because its surfaces are open.
The rebuild treats the surface as walls, plugs holes, and floods the outside. Holes up to about 40% of a part's width fill cleanly; bigger ones leave a dent.
Through-gaps and shallow concavities stay open, while deep blind pits fill in, which also stops them trapping resin.
The report warns when a result still looks like a hollow shell (volume under 3% of its bounding box; solid minis are ~10%).

## How it fits together

```
browser ─► RunPod proxy :7860 ─► Gradio UI + orchestrator (/opt/venvs/ui)
                                   │ starts one model worker at a time (whole GPU each)
                                   ├─ hunyuan / hunyuan2mv / ultrashape / hi3d / hunyuan_paint   (py3.10, torch 2.5.1 cu124)
                                   └─ print prep (CPU): repair → Z-up → scale → flatten → decimate → STL/3MF
/workspace (pod volume): HF weights cache, outputs/, logs/
```

- `app/`: UI (`ui.py`, `ui.css`), orchestrator with one method per step, print prep, model catalogue (`models.yaml`, which defines every UI control).
- `workers/`: one small HTTP worker per model, each run with its own venv's Python.
- `scripts/build/`: image build scripts. **Every upstream repo is pinned to a commit here.**
- `docker/`: Dockerfile and pod startup hook. `.github/workflows/build.yml` builds and pushes the image.

## One-time setup

1. **GitHub:** create a repo (public is simplest; private gets 2,000 free Actions minutes/month) and push this one:
   ```
   git remote add origin https://github.com/<you>/image-to-3d.git
   git branch -M main
   git push -u origin main
   ```
2. **Docker Hub:** create an access token (Account settings → Personal access tokens, read/write).
   In the GitHub repo, open Settings → Secrets and variables → Actions and add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.
3. **Build:** every push to `main` runs tests and builds `docker.io/<you>/image-to-3d:main`.
   For a release, push a tag (`git tag v0.1.0 && git push --tags`) to get `:v0.1.0`. The first build takes roughly 1-2 h; later builds reuse the layer cache and take about 15 minutes.
4. **RunPod secret:** in the RunPod console, open Secrets and create `ui_password`. The UI asks for user `admin` with this password.
5. **RunPod template:** [runpod/template.json](runpod/template.json) is the request body for RunPod's REST v2 `POST /templates`, and what Claude sends through the RunPod MCP.
   To create it in the console instead, open Templates → New Template and enter:
   - Image: `docker.io/<you>/image-to-3d:v0.4.0`
   - Container disk: 60 GB. Volume disk: 80 GB at `/workspace` (all models' weights take ~44 GB).
   - HTTP ports: `7860` (UI) and `8888` (Jupyter). TCP port: `22`.
   - Env: `UI_PASSWORD` = `{{ RUNPOD_SECRET_ui_password }}`, plus optionally `PREFETCH_MODELS` = `hunyuan hunyuan2mv ultrashape hunyuan_paint` (or `all`).
   - Allowed CUDA versions: 12.6 or newer.

## Each session

1. Deploy a pod from the template on an **A40 48 GB** (cheap, runs everything except Hi3D at 1536).
   An RTX A6000 or L40S is faster. Use an A100/H100 80 GB for Hi3D at 1536.
   Cards with 20-24 GB should fit if you turn on Hi3D's CPU offload, but none has been tested.
   Pick a pod with at least 32 GB of RAM: the UI held about 20 GB after a few runs, mostly background removal, and print prep adds up to 9 GB.
   **Avoid Blackwell** (RTX 5090, RTX PRO 6000, B200): this image has no kernels for them.
2. Open `https://<pod-id>-7860.proxy.runpod.net` and log in.
3. Work through the steps. The 3D viewer stays beside them, and a row above it switches between what each step made.
   - **1 Image:** upload one image or a turnaround sheet (see below) and click **Prepare** to check the cut-out. PNGs with transparency are used as-is; other images get their background removed.
   - **2 Shape:** pick a seed and click **Generate**. Hunyuan3D 2.1 runs for one image, Hunyuan3D-2mv for turnaround views. Every Generate starts a new run, so an earlier seed is never overwritten.
   - **3 Refine** (optional): UltraShape or Hi3D. Refining again starts from the Shape result and clears that run's Color and Print files.
   - **4 Color** (optional): click **Paint** for a textured GLB, see below.
   - **5 Print:** choose a height and triangle budget and click **Prepare for print**. Repeat it for other sizes without regenerating.
   - **Run all steps** goes from the current step to Print without stopping. It skips Refine when that step is set to None, and Color unless *Include in Run all steps* is ticked.
   - The first use of each model downloads its weights (minutes). Watch progress under **Details**.
4. **Open a previous run** loads any earlier run back into the steps, so you can continue from any of them.
5. Download what you need, then **terminate** the pod. A stopped pod keeps billing for its 80 GB volume (~$16/month).
   If you run sessions often, a network volume (~$7/month per 100 GB, 80 GB recommended) keeps the weights between pods, but ties you to one data centre's GPU stock.

Outputs per run (`/workspace/outputs/<date>/<time>_<name>_<generator>_s<seed>/`):
`input.png` (the cut-out actually used), `raw.glb`, `refined.glb`, `preview_<step>.glb`, `<name>_<height>_<faces>.stl`, the same `.3mf` and `_preview.glb`, `run.json` (all settings and the report, rewritten after every step).
Color adds `textured.glb`, plus `textured.obj` with its material and texture maps.
Turnaround runs add `view_<slot>.png` (each view as the model saw it).

## Turnaround sheets (multi-view)

Choose **Turnaround** in the Image step when you have the same figure from several sides. You can:
- **Upload one sheet** with the figures side by side and click **Prepare**. If the views are not front, left, back, right from left to right, set the order under **Advanced** (e.g. `front, left, back`).
  Captions are dropped; use `skip` for views the models cannot use, such as 3/4 views. A sheet takes priority over the four slots.
- **Or drop separate images** into the four slots.

The slots are named by what the picture shows. *Left* means the figure faces left on the page, so you see its left side; *right* is the opposite.
Every view is cut out and framed at the same figure height before it reaches the model.
The front is required; any of the other three can stay empty.

The Shape step then uses **Hunyuan3D-2mv**, Hunyuan's multi-view shape model. It copes with views that are a little off,
including renders from above, and gave accurate figures on painted-mini renders. Add **UltraShape** for the sharpest detail
(rivets, fingers, wrapped handles): about 18 minutes in total. Refiners use the front view.
*Turbo* (5-10 steps) is faster with a small quality cost. The first use of each variant downloads ~5 GB.

## Color

The Color step paints the current mesh with **Hunyuan3D-Paint 2.1** and writes a textured GLB. STL and 3MF stay geometry-only.
- **Only the front image guides the color.** The model takes one reference image, so the back and sides are invented, even on turnaround runs.
- **The painted model is a separate, lighter copy:** 40k triangles by default, with about 2K texture maps. The print mesh keeps its full detail.
- The first use downloads about 11 GB unless `hunyuan_paint` is in `PREFETCH_MODELS`.

## Print settings

| Setting | Default | Notes |
|---|---|---|
| Height | 32 mm | Z height after flattening. Typical: 28 / 32 mm heroic, 54 mm, 75 mm. |
| Triangle budget | 200k | 40k matches your earlier STLs; "Keep all" skips decimation. |
| Flatten base | 0.3 mm | Cuts the bottom flat and caps it, so the base sits on the plate. |
| Loose parts | 2 % | Deletes floating fragments smaller than this share of the main body. |
| Extra rotation | 0° | Use it if a model comes out lying down or facing backwards. |
| Solid rebuild | on | For meshes that stay open after repair; smooths detail finer than the voxel size. |

## Checking a new image build on a pod

Over SSH or a Jupyter terminal:
```
/opt/venvs/ui/bin/python /opt/app/scripts/smoke_test.py /workspace/test.png
```
It runs every generator/refiner combination and writes a table to `/workspace/outputs/smoke_<time>.md` with time, peak VRAM, faces and watertightness. Add `--paint` to run the Color step on each.
Add `--views <folder>` with `front.png` (plus any of `left.png`, `back.png`, `right.png`) to include the turnaround generators.
Use those numbers to correct the VRAM notes in `app/models.yaml`.

## Local development (no GPU)

PowerShell:
```
uv venv .venv --python 3.11
uv pip install --python .venv -r app/requirements.txt
.venv\Scripts\python -m pytest -q tests
$env:MOCK_WORKERS = "1"; .venv\Scripts\python -m app.ui      # http://localhost:7860
```
Mock workers return a placeholder figure, or any mesh you point `MOCK_MESH` at, and paint it with a gradient, so the UI, every step and print prep can be exercised without a GPU.

## Updating a model

Change the commit SHA in `scripts/build/install_<model>.sh` (and requirement pins in `envs/` if upstream changed them), push, then tag a new version and point the template at it.
To add a model, give it a worker under `workers/`, an install script, and entries in `app/models.yaml`.

## Troubleshooting

- **UI not reachable:** read `/workspace/logs/ui.log`. The UI restarts itself if it crashes.
- **A model fails:** read `/workspace/logs/<worker>.log` (also visible under **Details** in the UI).
- **Out of GPU memory:** lower the resolution, enable UltraShape's low-VRAM mode or Hi3D's CPU offload, or pick a bigger GPU.
- **"may be a hollow shell" warning:** do not print that STL as-is; try another seed or generator and report it.
- **A model comes out facing backwards or lying down:** use the extra rotations under Print → Advanced.
- **Long jobs:** RunPod's proxy closes connections that are silent for about 100 s. The UI sends progress every ~1.5 s, and every result is saved to disk, so after a disconnect use **Open a previous run**.
