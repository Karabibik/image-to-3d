"""Mesh export shared by the TRELLIS.2-based workers."""

from __future__ import annotations

import trimesh


def geometry_to_glb(vertices, faces, decimation_target: int = 1_000_000, **_texture_args) -> trimesh.Trimesh:
    """Stand-in for o_voxel.postprocess.to_glb that keeps only the geometry.

    to_glb ends with a UV unwrap and texture bake that print prep never uses. On messy
    surfaces the unwrap ran for over 15 minutes and filled the pod's 50 GB of RAM.
    The narrow-band remesh it offers (remesh=True, as upstream calls it) is skipped too: it
    wraps every surface in an inner and outer skin, a hollow shell. Print prep's solidify()
    closes the surface instead.
    """
    import cumesh

    mesh = cumesh.CuMesh()
    mesh.init(vertices.cuda(), faces.cuda())
    mesh.fill_holes(max_hole_perimeter=3e-2)
    mesh.simplify(decimation_target)
    vertices, faces = (tensor.cpu().numpy() for tensor in mesh.read())
    # The axis change to_glb ends with; callers rely on it for their own rotations.
    vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
