"""Day 1 - Point cloud basics: load, inspect, voxel downsample, estimate normals.

Introduces Open3D's core ``PointCloud`` type and the front of every perception
pipeline: convert a mesh to points, reduce density with a voxel grid, and
estimate surface normals via local PCA.

Run:
    python day1_pointcloud_basics.py            # interactive viewer
    python day1_pointcloud_basics.py --headless # save PNGs to ./figures
"""
import numpy as np
import open3d as o3d

from o3d_utils import bunny_point_cloud, fig_path, render, stage_out_dir


def inspect(pcd: o3d.geometry.PointCloud, label: str) -> None:
    """Print basic facts about a point cloud (count, extent, attributes)."""
    pts = np.asarray(pcd.points)
    print(f"\n[{label}]")
    print(f"  points          : {len(pts):,}")
    print(f"  X range         : {pts[:, 0].min():.3f} ~ {pts[:, 0].max():.3f}")
    print(f"  Y range         : {pts[:, 1].min():.3f} ~ {pts[:, 1].max():.3f}")
    print(f"  Z range         : {pts[:, 2].min():.3f} ~ {pts[:, 2].max():.3f}")
    print(f"  has colors      : {pcd.has_colors()}")
    print(f"  has normals     : {pcd.has_normals()}")


def main(out_dir: str | None = None) -> None:
    voxel_size = 0.005  # in Bunny model units (the Bunny is ~0.3 across)

    # 1. Load a sample cloud. In production a laspy-read .las file would go here;
    #    Open3D's built-in Stanford Bunny is the standard stand-in.
    print("=" * 60)
    print("STEP 1: load sample point cloud")
    print("=" * 60)
    pcd_original = bunny_point_cloud(n_points=100_000, voxel_size=None)
    inspect(pcd_original, "original PCD")

    # 2. Voxel downsampling: partition space into a grid of edge `voxel_size` and
    #    replace the points in each cell with their centroid. Fewer points and
    #    uniform density, at the cost of fine edge detail.
    print("\n" + "=" * 60)
    print("STEP 2: voxel downsampling")
    print("=" * 60)
    pcd_down = pcd_original.voxel_down_sample(voxel_size=voxel_size)
    inspect(pcd_down, f"downsampled (voxel_size={voxel_size})")
    n_orig = len(pcd_original.points)
    n_down = len(pcd_down.points)
    print(f"  kept            : {n_down / n_orig:.1%} ({n_orig:,} -> {n_down:,})")

    # 3. Normal estimation: for each point, fit a local plane to its neighbours
    #    (PCA) and take the plane normal. Hybrid search caps the neighbourhood by
    #    both radius and count.
    print("\n" + "=" * 60)
    print("STEP 3: normal estimation")
    print("=" * 60)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    # Resolve the sign ambiguity by orienting normals towards a virtual camera.
    pcd_down.orient_normals_towards_camera_location(np.array([0.0, 0.0, 1.0]))
    inspect(pcd_down, "after normal estimation")
    print(f"  sample normal   : {np.asarray(pcd_down.normals)[0].round(3)}")

    # 4. Visualize.
    print("\n" + "=" * 60)
    print("STEP 4: visualization (controls: drag=rotate, scroll=zoom, Q=close)")
    print("=" * 60)
    pcd_original.paint_uniform_color([0.7, 0.7, 0.7])  # grey: original
    pcd_down.paint_uniform_color([1.0, 0.3, 0.3])      # red: downsampled

    render(
        [pcd_original],
        window_name="Day1 - original PCD (100,000 pts)",
        out_path=fig_path(out_dir, "day1_original.png"),
    )
    render(
        [pcd_down],
        window_name="Day1 - downsampled PCD with normals",
        out_path=fig_path(out_dir, "day1_downsampled_normals.png"),
        point_show_normal=True,
    )
    pcd_shifted = o3d.geometry.PointCloud(pcd_down)
    pcd_shifted.translate([0.15, 0, 0])  # offset along x to show side by side
    render(
        [pcd_original, pcd_shifted],
        window_name="Day1 - original (grey) vs downsampled (red)",
        out_path=fig_path(out_dir, "day1_compare.png"),
        width=1200,
    )

    # 5. Summary.
    print("\n" + "=" * 60)
    print("STEP 5: summary")
    print("=" * 60)
    print(f"  original points : {n_orig:,}")
    print(f"  downsampled     : {n_down:,}  (voxel_size={voxel_size})")
    print(f"  reduction       : {(1 - n_down / n_orig):.1%}")
    print(f"  normals ready   : {pcd_down.has_normals()}")


# Takeaways
# 1. A PointCloud wraps an (N, 3) array; .points/.colors/.normals expose numpy.
# 2. voxel_down_sample compresses by a 3D grid; larger voxels lose more detail.
# 3. estimate_normals does a KD-tree neighbour search then PCA for the normal.
# 4. orient_normals_towards_camera_location fixes the normal sign ambiguity.
# 5. draw_geometries renders any list of geometries together.

if __name__ == "__main__":
    main(out_dir=stage_out_dir(__doc__))
