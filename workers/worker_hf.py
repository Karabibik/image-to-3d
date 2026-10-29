"""Hugging Face helpers shared by the TRELLIS.2-based workers."""

from __future__ import annotations

import os
from pathlib import Path

# TRELLIS.2 and Pixal3D reference two gated repos in their pipeline.json. These ungated
# copies hold the same weights (Pixal3D's own inference.py uses the DINOv3 one), so the
# template runs without an HF token. Point the env vars back at the originals if you
# have accepted their licenses.
MIRRORS = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": os.environ.get("DINOV3_REPO", "camenduru/dinov3-vitl16-pretrain-lvd1689m"),
    "briaai/RMBG-2.0": os.environ.get("BIREFNET_REPO", "ZhengPeng7/BiRefNet"),
}


def patched_pipeline_dir(repo_id: str, replacements: dict[str, str] = MIRRORS) -> str:
    """Download `repo_id` and return a local dir whose pipeline.json uses `replacements`.

    Everything except pipeline.json is symlinked to the HF snapshot, so nothing is copied.
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id))
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    target = hf_home / "patched" / repo_id.replace("/", "--")
    target.mkdir(parents=True, exist_ok=True)
    for entry in snapshot.iterdir():
        if entry.name == "pipeline.json":
            continue
        link = target / entry.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry.resolve() if entry.is_file() else entry)
    text = (snapshot / "pipeline.json").read_text()
    for old, new in replacements.items():
        text = text.replace(f'"{old}"', f'"{new}"')
    (target / "pipeline.json").write_text(text)
    return str(target)
