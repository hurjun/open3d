"""Day 2 - Outlier removal: statistical vs. radius filtering.

Real scans carry stray noise points. This stage adds synthetic noise to a clean
cloud and compares the two classic Open3D denoisers side by side.

Run:
    python day2_outlier_removal.py
    python day2_outlier_removal.py --headless
"""
import numpy as np
import open3d as o3d

from o3d_utils import bunny_point_cloud, fig_path, render, stage_out_dir


def add_noise(pcd: o3d.geometry.PointCloud, n_noise: int = 500) -> o3d.geometry.PointCloud:
    """Add uniformly random points inside the cloud's bounding box (fake noise)."""
    pts = np.asarray(pcd.points)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    noise_pts = np.random.uniform(mins, maxs, size=(n_noise, 3))
    noisy = o3d.geometry.PointCloud()
    noisy.points = o3d.utility.Vector3dVector(np.vstack([pts, noise_pts]))
    return noisy


def show_split(inlier, outlier, title: str, out_path: str | None) -> None:
    """Render inliers (grey) and removed outliers (red) together."""
    inlier.paint_uniform_color([0.6, 0.6, 0.6])
    outlier.paint_uniform_color([1.0, 0.0, 0.0])
    print(f"\n[{title}]  inlier={len(inlier.points):,}  outlier={len(outlier.points):,}")
    render([inlier, outlier], window_name=title, out_path=out_path)


def main(out_dir: str | None = None) -> None:
    np.random.seed(42)
    pcd_clean = bunny_point_cloud(n_points=50_000, voxel_size=0.005)
    pcd_noisy = add_noise(pcd_clean, n_noise=500)
    pcd_noisy.paint_uniform_color([0.6, 0.6, 0.6])

    print("=" * 60)
    print(f"clean points    : {len(pcd_clean.points):,}")
    print(f"after +500 noise: {len(pcd_noisy.points):,}")
    print("=" * 60)
    render(
        [pcd_noisy],
        window_name="Day2 - noisy input",
        out_path=fig_path(out_dir, "day2_noisy.png"),
    )

    # Method 1 - statistical outlier removal:
    #   For each point, look at `nb_neighbors` neighbours and drop points whose
    #   mean neighbour distance exceeds global_mean + std_ratio * sigma. Smaller
    #   std_ratio is more aggressive.
    print("\n" + "=" * 60)
    print("Method 1: statistical outlier removal (nb_neighbors=20, std_ratio=2.0)")
    print("=" * 60)
    inlier_stat, idx_stat = pcd_noisy.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    outlier_stat = pcd_noisy.select_by_index(idx_stat, invert=True)
    show_split(
        inlier_stat, outlier_stat,
        "Day2 - statistical: inlier (grey) + outlier (red)",
        fig_path(out_dir, "day2_statistical.png"),
    )

    # Method 2 - radius outlier removal:
    #   Drop points that have fewer than `nb_points` neighbours within `radius`.
    #   Judges local density, so it is stricter on uneven real-world LiDAR.
    print("\n" + "=" * 60)
    print("Method 2: radius outlier removal (nb_points=16, radius=0.02)")
    print("=" * 60)
    inlier_rad, idx_rad = pcd_noisy.remove_radius_outlier(nb_points=16, radius=0.02)
    outlier_rad = pcd_noisy.select_by_index(idx_rad, invert=True)
    show_split(
        inlier_rad, outlier_rad,
        "Day2 - radius: inlier (grey) + outlier (red)",
        fig_path(out_dir, "day2_radius.png"),
    )

    print("\n" + "=" * 60)
    print("comparison")
    print("=" * 60)
    total = len(pcd_noisy.points)
    print(f"  total points      : {total:,}")
    print(f"  statistical removed: {len(outlier_stat.points):,}  ({len(outlier_stat.points)/total:.1%})")
    print(f"  radius removed     : {len(outlier_rad.points):,}  ({len(outlier_rad.points)/total:.1%})")


# Takeaways
# 1. Statistical: thresholds on the global distance distribution; intuitive.
# 2. Radius: thresholds on local density; stricter on uneven real LiDAR.
# 3. remove_*_outlier returns (filtered_pcd, inlier_indices).
# 4. select_by_index(..., invert=True) extracts the removed outliers.
# 5. A common real-world recipe is statistical then radius in two passes.

if __name__ == "__main__":
    main(out_dir=stage_out_dir(__doc__))
