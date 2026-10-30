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


def patched_pipeline_dir(
    repo_id: str,
    config: str = "pipeline.json",
    allow: list[str] | None = None,
    ignore: list[str] | None = None,
    replacements: dict[str, str] = MIRRORS,
) -> str:
    """Download `repo_id` and return a local dir whose `config` uses `replacements`.

    `allow` / `ignore` limit the download (e.g. one pipeline's checkpoints). Everything
    except `config` is symlinked to the HF snapshot, so nothing is copied.
    """
    from huggingface_hub import snapshot_download

    if allow is not None:
        allow = [config, *allow]
    snapshot = Path(snapshot_download(repo_id, allow_patterns=allow, ignore_patterns=ignore))
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    target = hf_home / "patched" / repo_id.replace("/", "--")
    target.mkdir(parents=True, exist_ok=True)
    for entry in snapshot.iterdir():
        link = target / entry.name
        if link.is_symlink() or link.exists():
            link.unlink()  # never write through an old link into the HF cache
        if entry.name != config:
            link.symlink_to(entry.resolve() if entry.is_file() else entry)
    text = (snapshot / config).read_text()
    for old, new in replacements.items():
        text = text.replace(f'"{old}"', f'"{new}"')
    (target / config).write_text(text)
    return str(target)
