"""Headless smoke tests for the core perception stages.

These run without a display and without any external dataset: a tiny synthetic
"ground plus boxes" scene is generated in-process, and each test asserts an
algorithmic invariant rather than an exact rendered image. They exercise the
same code paths the day3 stage script uses.
"""
import numpy as np
import open3d as o3d

from o3d_utils import make_driving_scene, o3d_to_trimesh, trimesh_to_o3d


def tiny_scene() -> o3d.geometry.PointCloud:
    """A small synthetic scene, generated in-test (no files, no network)."""
    return make_driving_scene(seed=0, scale=0.1)


def test_scene_is_nonempty():
    pcd = tiny_scene()
    assert len(pcd.points) > 100


def test_voxel_downsample_reduces_count():
    pcd = tiny_scene()
    before = len(pcd.points)
    down = pcd.voxel_down_sample(voxel_size=0.2)
    after = len(down.points)
    assert 0 < after <= before
    assert after < before  # a 0.2 m voxel must merge some of the dense ground


def test_ransac_plane_is_horizontal():
    """The dominant plane of the scene is the ground, so its normal ~ +/- z."""
    pcd = tiny_scene()
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05, ransac_n=3, num_iterations=200
    )
    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    # |z component| close to 1 means the plane is (near) horizontal.
    assert abs(normal[2]) > 0.95
    # The ground dominates the point count.
    assert len(inliers) > 0.5 * len(pcd.points)


def test_dbscan_finds_objects_after_ground_removal():
    pcd = tiny_scene()
    _, inliers = pcd.segment_plane(
        distance_threshold=0.05, ransac_n=3, num_iterations=200
    )
    objects = pcd.select_by_index(inliers, invert=True)
    labels = np.array(objects.cluster_dbscan(eps=0.3, min_points=5))
    n_clusters = labels.max() + 1
    # The scene plants three boxes; clustering should recover at least one.
    assert n_clusters >= 1


def test_trimesh_roundtrip_preserves_counts():
    """o3d -> trimesh -> o3d keeps vertex and triangle counts."""
    mesh = o3d.geometry.TriangleMesh.create_box()
    tm = o3d_to_trimesh(mesh)
    assert len(tm.vertices) == len(mesh.vertices)
    assert len(tm.faces) == len(mesh.triangles)
    back = trimesh_to_o3d(tm)
    assert len(back.vertices) == len(mesh.vertices)
    assert len(back.triangles) == len(mesh.triangles)
