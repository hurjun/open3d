"""Day 4 - ICP registration: point-to-point vs. point-to-plane.

Iterative Closest Point aligns a moved/rotated "source" cloud back onto a fixed
"target" - the core of multi-view scan registration and LiDAR odometry. This
stage compares the two standard error metrics.

Run:
    python day4_icp.py
    python day4_icp.py --headless
"""
import numpy as np
import open3d as o3d

from o3d_utils import bunny_point_cloud, fig_path, render, stage_out_dir


def apply_transform(
    pcd: o3d.geometry.PointCloud, tx: float, ty: float, tz: float, angle_deg: float
) -> o3d.geometry.PointCloud:
    """Return a copy translated by (tx, ty, tz) and rotated about the z-axis."""
    pcd2 = o3d.geometry.PointCloud(pcd)
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    pcd2.transform(T)
    return pcd2


def main(out_dir: str | None = None) -> None:
    pcd_target = bunny_point_cloud(n_points=30_000, voxel_size=0.005)
    # The source is the target nudged + rotated 15deg (a second viewpoint).
    pcd_source = apply_transform(pcd_target, tx=0.05, ty=0.03, tz=0.01, angle_deg=15)
    pcd_target.paint_uniform_color([0.3, 0.6, 1.0])  # blue: fixed target
    pcd_source.paint_uniform_color([1.0, 0.4, 0.3])  # red: source to align

    print("=" * 60)
    print("ICP setup: align the red source onto the blue target")
    print("=" * 60)
    render(
        [pcd_target, pcd_source],
        window_name="Day4 - before ICP (blue=target, red=source)",
        out_path=fig_path(out_dir, "day4_before.png"),
    )

    # STEP 1 - point-to-point ICP: minimise distance between matched point pairs.
    #   max_correspondence_distance gates which pairs are considered matches.
    print("\n" + "=" * 60)
    print("STEP 1: point-to-point ICP")
    print("=" * 60)
    result_p2p = o3d.pipelines.registration.registration_icp(
        source=pcd_source,
        target=pcd_target,
        max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    print(f"  converged (>0.9): {result_p2p.fitness > 0.9}")
    print(f"  fitness         : {result_p2p.fitness:.4f}  (matched-point ratio, 1.0=best)")
    print(f"  inlier_rmse     : {result_p2p.inlier_rmse:.6f}  (lower=better)")
    print(f"  transform:\n{result_p2p.transformation.round(3)}")

    pcd_source_p2p = o3d.geometry.PointCloud(pcd_source)
    pcd_source_p2p.transform(result_p2p.transformation)
    pcd_source_p2p.paint_uniform_color([1.0, 0.4, 0.3])
    render(
        [pcd_target, pcd_source_p2p],
        window_name="Day4 - point-to-point result",
        out_path=fig_path(out_dir, "day4_point_to_point.png"),
    )

    # STEP 2 - point-to-plane ICP: minimise distance to the target's surface
    #   (uses target normals), which usually converges faster and more accurately.
    print("\n" + "=" * 60)
    print("STEP 2: point-to-plane ICP")
    print("=" * 60)
    pcd_source_n = o3d.geometry.PointCloud(pcd_source)
    pcd_target_n = o3d.geometry.PointCloud(pcd_target)
    for p in (pcd_source_n, pcd_target_n):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    result_p2l = o3d.pipelines.registration.registration_icp(
        source=pcd_source_n,
        target=pcd_target_n,
        max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    print(f"  fitness         : {result_p2l.fitness:.4f}")
    print(f"  inlier_rmse     : {result_p2l.inlier_rmse:.6f}")

    pcd_source_p2l = o3d.geometry.PointCloud(pcd_source)
    pcd_source_p2l.transform(result_p2l.transformation)
    pcd_source_p2l.paint_uniform_color([1.0, 0.4, 0.3])
    render(
        [pcd_target, pcd_source_p2l],
        window_name="Day4 - point-to-plane result",
        out_path=fig_path(out_dir, "day4_point_to_plane.png"),
    )

    print("\n" + "=" * 60)
    print("comparison")
    print("=" * 60)
    print(f"  point-to-point  fitness={result_p2p.fitness:.4f}  rmse={result_p2p.inlier_rmse:.6f}")
    print(f"  point-to-plane  fitness={result_p2l.fitness:.4f}  rmse={result_p2l.inlier_rmse:.6f}")


# Takeaways
# 1. ICP returns the 4x4 transform that maps source onto target.
# 2. fitness = matched points / total points; closer to 1.0 is better aligned.
# 3. inlier_rmse = mean distance of matched pairs; lower is more precise.
# 4. Point-to-plane uses normals and generally converges faster/more accurately.
# 5. ICP needs a decent initial guess; too far apart and it diverges.

if __name__ == "__main__":
    main(out_dir=stage_out_dir(__doc__))
