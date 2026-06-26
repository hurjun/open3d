"""Shared helpers for the 3D perception tour.

These utilities remove the duplication that used to live across the individual
``dayN_*.py`` stage scripts: loading the Stanford Bunny, converting between
Open3D and trimesh, generating the synthetic "autonomous-driving" scene used by
the segmentation/clustering stage, and rendering geometries either to an
interactive window or to a PNG file for headless / CI runs.
"""
from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np
import open3d as o3d


# ── Command-line / output helpers ────────────────────────────────────────────
def stage_out_dir(description: str) -> str | None:
    """Parse common stage-script flags and return the figure output directory.

    Every ``dayN_*.py`` script accepts ``--headless`` (render to PNG instead of
    opening interactive windows) and ``--out-dir`` (where the PNGs go). This
    returns the directory when ``--headless`` is set, otherwise ``None`` (the
    signal used throughout to mean "open an interactive window").
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render to PNG files instead of opening interactive windows.",
    )
    parser.add_argument(
        "--out-dir",
        default="figures",
        help="Directory for headless screenshots (default: figures).",
    )
    args = parser.parse_args()
    return args.out_dir if args.headless else None


def fig_path(out_dir: str | None, filename: str) -> str | None:
    """Join ``out_dir`` and ``filename``, or return ``None`` for interactive mode."""
    return os.path.join(out_dir, filename) if out_dir else None


# ── Data loading ─────────────────────────────────────────────────────────────
def load_bunny_mesh() -> o3d.geometry.TriangleMesh:
    """Load the Stanford Bunny triangle mesh with per-vertex normals.

    Open3D ships the Bunny as a built-in dataset; the first call downloads it to
    ``~/open3d_data`` and caches it for subsequent runs.
    """
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    mesh.compute_vertex_normals()
    return mesh


def bunny_point_cloud(
    n_points: int = 50_000,
    voxel_size: float | None = 0.005,
    with_normals: bool = False,
    normal_radius: float = 0.01,
    normal_max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """Sample a point cloud from the Bunny mesh, optionally downsampled + normals.

    Args:
        n_points: Number of points to sample uniformly from the mesh surface.
        voxel_size: Voxel edge length for downsampling; ``None`` skips it.
        with_normals: If ``True``, estimate and camera-orient normals.
        normal_radius: Neighbour search radius for normal estimation.
        normal_max_nn: Maximum neighbours used to fit each local plane.

    Returns:
        The processed :class:`open3d.geometry.PointCloud`.
    """
    pcd = load_bunny_mesh().sample_points_uniformly(number_of_points=n_points)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if with_normals:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 1.0]))
    return pcd


def make_driving_scene(seed: int = 42, scale: float = 1.0) -> o3d.geometry.PointCloud:
    """Build a synthetic LiDAR-like scene: a flat ground plane plus a few boxes.

    The scene mimics an autonomous-driving frame so that the RANSAC ground
    segmentation and DBSCAN clustering stages have a predictable target: one
    dominant plane at ``z = 0`` and three box-shaped object clusters resting on
    it. ``scale`` shrinks both the extent and the point counts, which keeps the
    smoke test fast while preserving the geometry the algorithms rely on.

    Args:
        seed: Seed for the random generator (reproducible scenes).
        scale: Multiplier in ``(0, 1]`` for point counts (1.0 = full scene).

    Returns:
        A coloured-less :class:`open3d.geometry.PointCloud` of the whole scene.
    """
    rng = np.random.default_rng(seed)

    def n(count: int) -> int:
        return max(4, int(count * scale))

    # Ground: z ≈ 0 plane over a 10 x 10 m patch with mild sensor noise.
    ground = np.column_stack([
        rng.uniform(-5, 5, n(5000)),
        rng.uniform(-5, 5, n(5000)),
        rng.normal(0, 0.02, n(5000)),
    ])
    # Object 1: vehicle-sized box centred at (1, 0).
    obj1 = rng.uniform([-0.5, -1.0, 0.0], [0.5, 1.0, 1.5], (n(800), 3)) + [1.0, 0.0, 0.0]
    # Object 2: pedestrian-sized box at (-2, 1).
    obj2 = rng.uniform([-0.2, -0.2, 0.0], [0.2, 0.2, 1.8], (n(300), 3)) + [-2.0, 1.0, 0.0]
    # Object 3: small object at (3, -2).
    obj3 = rng.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.6], (n(200), 3)) + [3.0, -2.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([ground, obj1, obj2, obj3]))
    return pcd


# ── Open3D <-> trimesh conversion ────────────────────────────────────────────
def o3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh):
    """Convert an Open3D triangle mesh to a :class:`trimesh.Trimesh`.

    Imported lazily so the rest of the module does not require trimesh.
    """
    import trimesh

    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )


def trimesh_to_o3d(tm) -> o3d.geometry.TriangleMesh:
    """Convert a :class:`trimesh.Trimesh` back to an Open3D triangle mesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    mesh.compute_vertex_normals()
    return mesh


# ── Visualization ────────────────────────────────────────────────────────────
def render(
    geometries: Sequence,
    window_name: str = "Open3D",
    out_path: str | None = None,
    width: int = 1024,
    height: int = 768,
    point_show_normal: bool = False,
    mesh_show_back_face: bool = False,
) -> None:
    """Show geometries interactively, or save a PNG when ``out_path`` is given.

    When ``out_path`` is ``None`` this opens the usual Open3D viewer
    (left-drag = rotate, right-drag = pan, scroll = zoom, ``Q`` = close). When
    ``out_path`` is set, the same scene is rendered off-screen and written to
    disk, which lets the stage scripts run headlessly (no display) and produce
    the figures embedded in the README.

    Setting the environment variable ``OPEN3D_HEADLESS`` to a directory forces
    every call to save into that directory instead of opening a window.
    """
    headless_dir = os.environ.get("OPEN3D_HEADLESS")
    if out_path is None and headless_dir:
        safe = "".join(c if c.isalnum() else "_" for c in window_name)
        out_path = os.path.join(headless_dir, f"{safe}.png")

    if out_path is None:
        o3d.visualization.draw_geometries(
            list(geometries),
            window_name=window_name,
            width=width,
            height=height,
            point_show_normal=point_show_normal,
            mesh_show_back_face=mesh_show_back_face,
        )
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height, visible=False)
    for geom in geometries:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.point_show_normal = point_show_normal
    opt.mesh_show_back_face = mesh_show_back_face
    vis.poll_events()
    vis.update_renderer()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()
    print(f"  [saved] {out_path}")
