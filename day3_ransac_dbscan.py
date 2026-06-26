"""Day 3 - RANSAC ground segmentation + DBSCAN clustering.

The canonical first two steps of a LiDAR object-detection pipeline: fit and
remove the dominant ground plane with RANSAC, then group the remaining points
into objects with DBSCAN (no need to know the object count in advance).

Run:
    python day3_ransac_dbscan.py
    python day3_ransac_dbscan.py --headless
"""
import numpy as np
import open3d as o3d

from o3d_utils import fig_path, make_driving_scene, render, stage_out_dir


def main(out_dir: str | None = None) -> None:
    pcd = make_driving_scene(seed=42)
    print("=" * 60)
    print(f"total points: {len(pcd.points):,}")
    print("=" * 60)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    render(
        [pcd],
        window_name="Day3 - input scene",
        out_path=fig_path(out_dir, "day3_input.png"),
    )

    # STEP 1 - RANSAC plane segmentation.
    #   segment_plane returns plane_model [a, b, c, d] for ax+by+cz+d=0 and the
    #   indices of the inlier (ground) points. RANSAC samples `ransac_n` points,
    #   fits a candidate plane, counts inliers within `distance_threshold`, and
    #   keeps the best over `num_iterations` - robust to the object outliers.
    print("\n" + "=" * 60)
    print("STEP 1: RANSAC ground segmentation")
    print("=" * 60)
    plane_model, inlier_idx = pcd.segment_plane(
        distance_threshold=0.05, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    print(f"  plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    print(f"  normal (a,b,c): ({a:.3f}, {b:.3f}, {c:.3f})  <- should be near +z")
    print(f"  ground points : {len(inlier_idx):,}")

    ground_pcd = pcd.select_by_index(inlier_idx)
    objects_pcd = pcd.select_by_index(inlier_idx, invert=True)
    ground_pcd.paint_uniform_color([0.3, 0.8, 0.3])   # green: ground
    objects_pcd.paint_uniform_color([0.8, 0.3, 0.3])  # red: objects
    print(f"  object points : {len(objects_pcd.points):,}")
    render(
        [ground_pcd, objects_pcd],
        window_name="Day3 - RANSAC (green=ground, red=objects)",
        out_path=fig_path(out_dir, "day3_ransac.png"),
    )

    # STEP 2 - DBSCAN clustering on the non-ground points.
    #   Density-based clustering: points within `eps` of a core point (a point
    #   with >= `min_points` neighbours) form a cluster; label -1 is noise.
    print("\n" + "=" * 60)
    print("STEP 2: DBSCAN clustering (non-ground points)")
    print("=" * 60)
    labels = np.array(
        objects_pcd.cluster_dbscan(eps=0.3, min_points=10, print_progress=False)
    )
    n_clusters = labels.max() + 1  # excludes the -1 noise label
    n_noise = int((labels == -1).sum())
    print(f"  clusters found: {n_clusters}")
    print(f"  noise points  : {n_noise}")
    for i in range(n_clusters):
        print(f"  cluster {i}     : {int((labels == i).sum())} points")

    # STEP 3 - colour each cluster (noise stays black) and visualize.
    print("\n" + "=" * 60)
    print("STEP 3: cluster visualization")
    print("=" * 60)
    palette = np.random.default_rng(0).uniform(0.3, 1.0, size=(max(n_clusters, 1), 3))
    colors = np.where(
        labels[:, None] == -1,
        [0.0, 0.0, 0.0],
        palette[np.clip(labels, 0, max(n_clusters - 1, 0))],
    )
    objects_pcd.colors = o3d.utility.Vector3dVector(colors)
    render(
        [ground_pcd, objects_pcd],
        window_name="Day3 - RANSAC + DBSCAN result",
        out_path=fig_path(out_dir, "day3_clusters.png"),
    )
    print("\ndone.")


# Takeaways
# 1. RANSAC fits an outlier-robust plane by random sampling - the ground-detection standard.
# 2. segment_plane's inlier_idx splits ground/non-ground via select_by_index.
# 3. DBSCAN needs no preset cluster count - ideal when the object count is unknown.
# 4. label == -1 means a point belongs to no cluster (noise).
# 5. Tuning eps and min_points is the crux; too-large eps merges objects.

if __name__ == "__main__":
    main(out_dir=stage_out_dir(__doc__))
