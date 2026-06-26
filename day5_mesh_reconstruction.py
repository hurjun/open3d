"""Day 5 - Surface reconstruction: Poisson meshing from a point cloud.

Turns an oriented point cloud into a watertight triangle mesh, then trims the
low-confidence (low-density) regions that Poisson reconstruction hallucinates
outside the sampled surface.

Run:
    python day5_mesh_reconstruction.py
    python day5_mesh_reconstruction.py --headless
"""
import numpy as np
import open3d as o3d

from o3d_utils import bunny_point_cloud, fig_path, render, stage_out_dir


def main(out_dir: str | None = None) -> None:
    # Poisson requires oriented normals, so request them at sampling time.
    pcd = bunny_point_cloud(n_points=50_000, voxel_size=0.003, with_normals=True)
    print("=" * 60)
    print(f"PCD points : {len(pcd.points):,}")
    print(f"has normals: {pcd.has_normals()}")
    print("=" * 60)
    render(
        [pcd],
        window_name="Day5 - input PCD",
        out_path=fig_path(out_dir, "day5_input.png"),
    )

    # STEP 1 - Poisson reconstruction.
    #   Solves for an indicator function whose gradient matches the oriented
    #   normals; `depth` sets the octree resolution (higher = finer, slower).
    #   `densities` records how many samples support each output vertex.
    print("\n" + "=" * 60)
    print("STEP 1: Poisson surface reconstruction (depth=9)")
    print("=" * 60)
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )
    print(f"  vertices  : {len(mesh_poisson.vertices):,}")
    print(f"  triangles : {len(mesh_poisson.triangles):,}")

    # STEP 2 - trim low-density vertices (spurious surface away from real points).
    print("\n" + "=" * 60)
    print("STEP 2: trim low-density regions (bottom 5%)")
    print("=" * 60)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    to_remove = densities < threshold
    # remove_vertices_by_mask mutates in place and returns None; copy first.
    mesh_clean = o3d.geometry.TriangleMesh(mesh_poisson)
    mesh_clean.remove_vertices_by_mask(to_remove)
    mesh_clean.compute_vertex_normals()
    print(f"  vertices before : {len(mesh_poisson.vertices):,}")
    print(f"  vertices after  : {len(mesh_clean.vertices):,}")
    print(f"  triangles after : {len(mesh_clean.triangles):,}")

    # STEP 3 - peek at the mesh data structures.
    print("\n" + "=" * 60)
    print("STEP 3: mesh structure")
    print("=" * 60)
    verts = np.asarray(mesh_clean.vertices)
    tris = np.asarray(mesh_clean.triangles)
    print(f"  vertices shape : {verts.shape}  <- (N, 3) coordinates")
    print(f"  triangles shape: {tris.shape}  <- (M, 3) vertex indices")
    print(f"  triangle 0     : {tris[0]}  <- connects vertices {tris[0].tolist()}")

    # STEP 4 - visualize: colour the raw mesh by density, then the trimmed mesh.
    print("\n" + "=" * 60)
    print("STEP 4: visualization")
    print("=" * 60)
    d = (densities - densities.min()) / (densities.max() - densities.min())
    mesh_poisson.vertex_colors = o3d.utility.Vector3dVector(
        np.column_stack([d, d * 0.5, 1.0 - d])  # blue=low density, yellow=high
    )
    render(
        [mesh_poisson],
        window_name="Day5 - Poisson mesh (density colour)",
        out_path=fig_path(out_dir, "day5_density.png"),
        mesh_show_back_face=True,
    )

    mesh_clean.paint_uniform_color([0.7, 0.7, 0.7])
    render(
        [mesh_clean],
        window_name="Day5 - final mesh (low-density trimmed)",
        out_path=fig_path(out_dir, "day5_final.png"),
        mesh_show_back_face=True,
    )

    pcd_vis = o3d.geometry.PointCloud(pcd)
    pcd_vis.paint_uniform_color([1.0, 0.4, 0.3])
    mesh_shifted = o3d.geometry.TriangleMesh(mesh_clean)
    mesh_shifted.translate([0.2, 0, 0])
    render(
        [pcd_vis, mesh_shifted],
        window_name="Day5 - PCD (red) vs mesh (grey)",
        out_path=fig_path(out_dir, "day5_compare.png"),
        width=1200,
        mesh_show_back_face=True,
    )


# Takeaways
# 1. Poisson reconstruction uses oriented normals to build a watertight mesh.
# 2. Higher depth = more detail but slower; 8-10 is a practical range.
# 3. Low-density vertices sit where few real points exist -> trim them.
# 4. A mesh is vertices (coordinates) + triangles (triples of vertex indices).
# 5. Point clouds suit perception/analysis; meshes suit rendering/simulation.

if __name__ == "__main__":
    main(out_dir=stage_out_dir(__doc__))
