"""Day 7 - LiDAR .las ingestion with laspy.

Reads the LAS point format (xyz plus intensity, classification, ...), inspects
its header and per-class distribution, and converts it into an Open3D point
cloud coloured by classification. A sample .las is written to the OS temp
directory so the stage runs without any external dataset.

Run:
    python day7_laspy_basics.py
    python day7_laspy_basics.py --headless
"""
import os
import tempfile

import laspy
import numpy as np
import open3d as o3d

from o3d_utils import fig_path, render, stage_out_dir

# ASPRS LAS classification codes used in the synthetic scene.
CLASS_LABELS = {0: "unclassified", 2: "ground", 3: "vegetation", 6: "building"}
CLASS_COLORS = {
    0: [1.0, 0.3, 0.3],  # red:   unclassified (vehicle)
    2: [0.3, 0.8, 0.3],  # green: ground
    3: [0.5, 1.0, 0.2],  # lime:  vegetation
    6: [0.3, 0.3, 1.0],  # blue:  building
}


def write_sample_las(path: str) -> int:
    """Write a synthetic autonomous-driving scene to a LAS 1.4 file.

    Returns the number of points written.
    """
    rng = np.random.default_rng(42)

    def block(n, xr, yr, zr, cls, ir):
        x = rng.uniform(*xr, n)
        y = rng.uniform(*yr, n)
        z = rng.uniform(*zr, n) if zr[0] != zr[1] else rng.normal(0, 0.05, n)
        return x, y, z, np.full(n, cls, np.uint8), rng.integers(*ir, n, dtype=np.uint16)

    ground = block(8000, (-20, 20), (-20, 20), (0, 0), 2, (100, 300))
    building = block(3000, (5, 10), (5, 10), (0, 8), 6, (200, 500))
    veg = block(2000, (-10, -5), (-10, -5), (0, 4), 3, (50, 150))
    vehicle = block(800, (-2, 2), (-2, 2), (0, 1.5), 0, (150, 400))
    cols = [np.concatenate(parts) for parts in zip(ground, building, veg, vehicle)]
    x, y, z, cls, intensity = cols

    header = laspy.LasHeader(point_format=1, version="1.4")
    header.offsets = np.array([x.min(), y.min(), z.min()])
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header=header)
    las.x, las.y, las.z = x, y, z
    las.classification = cls
    las.intensity = intensity
    las.write(path)
    return len(x)


def main(out_dir: str | None = None) -> None:
    las_path = os.path.join(tempfile.gettempdir(), "sample_scene.las")

    print("=" * 60)
    print("STEP 1: write a sample .las file")
    print("=" * 60)
    n_written = write_sample_las(las_path)
    print(f"  path   : {las_path}")
    print(f"  points : {n_written:,}")

    # STEP 2 - read it back and inspect the header.
    print("\n" + "=" * 60)
    print("STEP 2: read .las and inspect structure")
    print("=" * 60)
    las = laspy.read(las_path)
    print(f"  points        : {len(las.x):,}")
    print(f"  version       : {las.header.version}")
    print(f"  point format  : {las.header.point_format.id}")
    print(f"  fields        : {list(las.point_format.dimension_names)}")
    print(f"  X range       : {las.x.min():.2f} ~ {las.x.max():.2f}")
    print(f"  Y range       : {las.y.min():.2f} ~ {las.y.max():.2f}")
    print(f"  Z range       : {las.z.min():.2f} ~ {las.z.max():.2f}")

    cls_array = np.array(las.classification)

    # STEP 3 - per-class point distribution.
    print("\n" + "=" * 60)
    print("STEP 3: classification distribution")
    print("=" * 60)
    for code, label in CLASS_LABELS.items():
        count = int((cls_array == code).sum())
        print(f"  [{code}] {label:13s}: {count:,}  ({count/len(cls_array):.1%})")

    # STEP 4 - convert to an Open3D cloud coloured by class.
    print("\n" + "=" * 60)
    print("STEP 4: laspy -> Open3D")
    print("=" * 60)
    xyz = np.column_stack([np.array(las.x), np.array(las.y), np.array(las.z)])
    colors = np.array([CLASS_COLORS.get(c, [0.5, 0.5, 0.5]) for c in cls_array])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"  Open3D points : {len(pcd.points):,}")
    print(f"  has colors    : {pcd.has_colors()}")

    # STEP 5 - visualize whole scene and the non-ground subset.
    print("\n" + "=" * 60)
    print("STEP 5: visualization "
          "(green=ground, blue=building, lime=vegetation, red=vehicle)")
    print("=" * 60)
    render(
        [pcd],
        window_name="Day7 - full LiDAR scene (by class)",
        out_path=fig_path(out_dir, "day7_full.png"),
    )
    ground_mask = cls_array == 2
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(xyz[~ground_mask])
    pcd_objects.colors = o3d.utility.Vector3dVector(colors[~ground_mask])
    render(
        [pcd_objects],
        window_name="Day7 - ground removed (building/vegetation/vehicle)",
        out_path=fig_path(out_dir, "day7_no_ground.png"),
    )

    # STEP 6 - intensity by class (laser return strength ~ material).
    print("\n" + "=" * 60)
    print("STEP 6: mean intensity by class")
    print("=" * 60)
    intensity = np.array(las.intensity)
    for code, label in CLASS_LABELS.items():
        mask = cls_array == code
        if mask.sum() > 0:
            print(f"  [{code}] {label:13s}: mean intensity {intensity[mask].mean():.1f}")
    print("\ndone.")


# Takeaways
# 1. LAS carries intensity, classification, return_number, ... beyond xyz.
# 2. Classification often pre-labels ground/building/vegetation - usable before RANSAC.
# 3. np.column_stack of las.x/y/z feeds straight into an Open3D point cloud.
# 4. Intensity helps material discrimination (metal vehicles reflect strongly).
# 5. Real pipeline: laspy read -> class filter -> Open3D preprocessing -> meshing.

if __name__ == "__main__":
    main(out_dir=stage_out_dir(__doc__))
