# Image → printable miniature (RunPod template)

Upload character art, pick an image-to-3D model, and download a print-ready STL/3MF:
watertight, standing on Z, scaled to a height in millimetres and reduced to a triangle budget.
Everything runs on a rented RunPod GPU behind one web UI.

| Model | Role | Peak VRAM, measured on an A40 | License |
|---|---|---|---|
| **Hunyuan3D 2.1** | generator, shape only | 11.5 GB | Tencent Hunyuan Community (excludes EU, UK, South Korea) |
| **TRELLIS.2** | generator, 512 / 1024 / 1536 | 9.9 GB at 1024 | MIT |
| **Pixal3D** | generator, pixel-aligned (follows the art closely) | 35.2 GB at 1536 (use low-VRAM mode below 48 GB) | MIT |
| **UltraShape 1.0** | refiner, best after Hunyuan3D | 15.0 GB | Apache-2.0 |
| **Hi3D v1.0** *(experimental)* | refiner, up to 1536 | 1536 only tested upstream on 96 GB | Hi3D Open License |

Timings from the first smoke test (A40, every model's first run, so they include loading
and, for TRELLIS.2/Pixal3D, one-off Triton kernel tuning): Hunyuan3D 44 s,
+ UltraShape 611 s at 50 steps (use 12 steps for a quick pass), TRELLIS.2 1024 530 s,
Pixal3D 1536 566 s. Later runs on the same pod are much faster.

Meshes that are still open or non-manifold after repair are rebuilt by print prep with a
signed 0.05 mm voxel remesh (on by default; a few minutes of CPU time). The report warns when a
result looks like a hollow shell (volume under 3% of its bounding box; solid minis are ~10%).
TRELLIS.2/Pixal3D are exported without their upstream narrow-band remesh, which produced
exactly such hollow double-skinned shells in v0.1.0.

## How it fits together

```
browser ─► RunPod proxy :7860 ─► Gradio UI + orchestrator (/opt/venvs/ui)
                                   │ starts one model worker at a time (whole GPU each)
                                   ├─ hunyuan / ultrashape / hi3d   (py3.10, torch 2.5.1 cu124)
                                   └─ trellis / pixal3d             (py3.11, torch 2.7.0 cu126)
                                   └─ print prep (CPU): repair → Z-up → scale → flatten → decimate → STL/3MF
/workspace (pod volume): HF weights cache, outputs/, logs/
```

- `app/`: UI (`ui.py`), orchestrator, print prep, model catalogue (`models.yaml`, which defines every UI control).
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
   For a release, push a tag (`git tag v0.1.0 && git push --tags`) to get `:v0.1.0`. The first build takes roughly 1-2 h; later builds reuse the layer cache.
4. **RunPod secret:** in the RunPod console, open Secrets and create `ui_password`. The UI asks for user `admin` with this password.
5. **RunPod template:** [runpod/template.json](runpod/template.json) is the request body for RunPod's REST v2 `POST /templates`, and what Claude sends through the RunPod MCP.
   To create it in the console instead, open Templates → New Template and enter:
   - Image: `docker.io/<you>/image-to-3d:v0.1.1`
   - Container disk: 60 GB. Volume disk: 100 GB at `/workspace`.
   - HTTP ports: `7860` (UI) and `8888` (Jupyter). TCP port: `22`.
   - Env: `UI_PASSWORD` = `{{ RUNPOD_SECRET_ui_password }}`, plus optionally `PREFETCH_MODELS` = `hunyuan trellis` (or `all`).
   - Allowed CUDA versions: 12.6 or newer (the TRELLIS.2/Pixal3D stack uses CUDA 12.6 wheels).

## Each session

1. Deploy a pod from the template on an **A40 48 GB** (cheap, runs everything except Hi3D at 1536).
   An RTX A6000 or L40S is faster. Use an A100/H100 80 GB for Hi3D at 1536.
   **Avoid Blackwell** (RTX 5090, RTX PRO 6000, B200): this image has no kernels for them.
2. Open `https://<pod-id>-7860.proxy.runpod.net` and log in.
3. Upload one or more images. PNGs with transparency are used as-is; other images get their background removed.
   Choose a generator, an optional refiner and print settings, then click **Generate**.
   - The first use of each model downloads its weights (minutes). Watch progress under **Worker log**.
   - Several images form a batch, and the downloads then include a zip of all STL/3MF files.
4. **History** lists every run. Pick one and click *Re-run print prep* to get another height or triangle budget without regenerating.
5. Download what you need, then **terminate** the pod. A stopped pod keeps billing for its 100 GB volume (~$20/month).
   If you run sessions often, a network volume (~$7/month per 100 GB) keeps the weights between pods, but ties you to one data centre's GPU stock.

Outputs per run (`/workspace/outputs/<date>/<time>_<name>_<models>_s<seed>/`):
`input.png` (the cut-out actually used), `raw.glb`, `refined.glb`, `<name>.stl`, `<name>.3mf`, `<name>_preview.glb`, `run.json` (all settings and the report).

## Print settings

| Setting | Default | Notes |
|---|---|---|
| Height | 32 mm | Z height after flattening. Typical: 28 / 32 mm heroic, 54 mm, 75 mm. |
| Triangle budget | 200k | 40k matches your earlier STLs; "Keep all" skips decimation. |
| Flatten base | 0.3 mm | Cuts the bottom flat and caps it, so the base sits on the plate. |
| Loose parts | 2 % | Deletes floating fragments smaller than this share of the main body. |
| Extra rotation | 0° | Use it if a model comes out lying down or facing backwards. |
| Voxel remesh | off | Last resort for meshes that stay open after repair; smooths detail finer than the voxel size. |

## Checking a new image build on a pod

Over SSH or a Jupyter terminal:
```
/opt/venvs/ui/bin/python /opt/app/scripts/smoke_test.py /workspace/test.png
```
It runs every generator/refiner combination and writes a table to `/workspace/outputs/smoke_<time>.md` with time, peak VRAM, faces and watertightness.
Use those numbers to correct the VRAM notes in `app/models.yaml`.

## Local development (no GPU)

PowerShell:
```
uv venv .venv --python 3.11
uv pip install --python .venv -r app/requirements.txt
.venv\Scripts\python -m pytest -q tests
$env:MOCK_WORKERS = "1"; .venv\Scripts\python -m app.ui      # http://localhost:7860
```
Mock workers return a placeholder figure, or any mesh you point `MOCK_MESH` at, so the UI, batching, history and print prep can be exercised without a GPU.

## Updating a model

Change the commit SHA in `scripts/build/install_<model>.sh` (and requirement pins in `envs/` if upstream changed them), push, then tag a new version and point the template at it.
To add a model, give it a worker under `workers/`, an install script, and entries in `app/models.yaml`.

## Troubleshooting

- **UI not reachable:** read `/workspace/logs/ui.log`. The UI restarts itself if it crashes.
- **A model fails:** read `/workspace/logs/<worker>.log` (also visible in the UI's Worker log panel).
- **Out of GPU memory:** lower the resolution, enable the model's low-VRAM option, or pick a bigger GPU.
- **"may be a hollow shell" warning:** do not print that STL as-is; try another seed or generator and report it.
- **A model comes out facing backwards or lying down:** use the extra rotations under Print settings → More,
  and if it is systematic for a generator, set its `rotate_z` in `app/models.yaml` (Pixal3D already has 180).
- **Gated Hugging Face models:** TRELLIS.2 and Pixal3D reference `facebook/dinov3-…` and `briaai/RMBG-2.0`.
  The workers swap these for ungated copies (`camenduru/dinov3-…`, `ZhengPeng7/BiRefNet`).
  Set `DINOV3_REPO`, `BIREFNET_REPO` and `HF_TOKEN` to use the originals.
- **Long jobs:** RunPod's proxy closes connections that are silent for about 100 s. The UI sends progress every ~1.5 s, and every result is saved to disk, so after a disconnect check the **History** tab.
