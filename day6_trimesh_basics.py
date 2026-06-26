"""Day 6 - Mesh quality validation with trimesh.

Before a mesh is usable downstream (collision, ray casting, simulation) it must
be checked: is it watertight, are the face windings consistent, and is it a
single connected component or a body plus stray fragments? This stage runs
those checks and keeps the largest component.

Run:
    python day6_trimesh_basics.py
    python day6_trimesh_basics.py --headless
"""
import trimesh

from o3d_utils import (
    fig_path,
    load_bunny_mesh,
    o3d_to_trimesh,
    render,
    stage_out_dir,
    trimesh_to_o3d,
)


def main(out_dir: str | None = None) -> None:
    print("=" * 60)
    print("Open3D mesh -> trimesh")
    print("=" * 60)
    mesh = o3d_to_trimesh(load_bunny_mesh())
    print(f"  vertices : {len(mesh.vertices):,}")
    print(f"  faces    : {len(mesh.faces):,}")

    # STEP 1 - quality metrics.
    print("\n" + "=" * 60)
    print("STEP 1: quality checks")
    print("=" * 60)
    # watertight: fully closed (no holes); required for volume and ray casting.
    print(f"  watertight         : {mesh.is_watertight}")
    # winding consistent: all face normals point the same way (in/out).
    print(f"  winding consistent : {mesh.is_winding_consistent}")
    if mesh.is_watertight:
        print(f"  volume             : {mesh.volume:.6f}")
    else:
        print("  volume             : n/a (not watertight)")
    print(f"  bounds min         : {mesh.bounds[0].round(3)}")
    print(f"  bounds max         : {mesh.bounds[1].round(3)}")
    print(f"  center of mass     : {mesh.center_mass.round(3)}")

    # STEP 2 - find holes via boundary edges (edges used by only one face).
    print("\n" + "=" * 60)
    print("STEP 2: hole detection")
    print("=" * 60)
    boundary_edges = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    print(f"  boundary edges : {len(boundary_edges)}")
    print("  -> no holes." if len(boundary_edges) == 0
          else f"  -> {len(boundary_edges)} open boundary edge(s).")

    # STEP 3 - connected component analysis.
    print("\n" + "=" * 60)
    print("STEP 3: connected components")
    print("=" * 60)
    components = mesh.split(only_watertight=False, repair=False)
    print(f"  components : {len(components)}")
    for i, comp in enumerate(components):
        print(f"  component {i}: vertices={len(comp.vertices):,}  faces={len(comp.faces):,}")

    # STEP 4 - keep only the largest component (drop fragments).
    print("\n" + "=" * 60)
    print("STEP 4: keep main component")
    print("=" * 60)
    main_mesh = max(components, key=lambda c: len(c.faces)) if components else mesh
    print(f"  main component faces : {len(main_mesh.faces):,}")
    print(f"  watertight           : {main_mesh.is_watertight}")

    # STEP 5 - visualize through the Open3D viewer.
    print("\n" + "=" * 60)
    print("STEP 5: visualization")
    print("=" * 60)
    o3d_full = trimesh_to_o3d(mesh)
    o3d_full.paint_uniform_color([0.7, 0.7, 0.7])
    render(
        [o3d_full],
        window_name="Day6 - full mesh",
        out_path=fig_path(out_dir, "day6_full.png"),
        mesh_show_back_face=True,
    )
    o3d_main = trimesh_to_o3d(main_mesh)
    o3d_main.paint_uniform_color([0.7, 0.7, 0.7])
    render(
        [o3d_main],
        window_name="Day6 - main component",
        out_path=fig_path(out_dir, "day6_main.png"),
        mesh_show_back_face=True,
    )

    # STEP 6 - summary report.
    print("\n" + "=" * 60)
    print("quality report")
    print("=" * 60)
    ok = lambda b: "OK" if b else "FAIL"  # noqa: E731
    print(f"  vertices           : {len(mesh.vertices):,}")
    print(f"  faces              : {len(mesh.faces):,}")
    print(f"  watertight         : {ok(mesh.is_watertight)}")
    print(f"  winding consistent : {ok(mesh.is_winding_consistent)}")
    print(f"  boundary edges     : {len(boundary_edges)} ({ok(len(boundary_edges) == 0)})")
    print(f"  components         : {len(components)} "
          f"({'OK' if len(components) == 1 else 'fragments present'})")


# Takeaways
# 1. trimesh specialises in mesh validation: watertight, winding, volume, boundary.
# 2. watertight = closed mesh with no holes; required for ray casting/collision.
# 3. boundary edges are edges used by a single face = the rim of a hole.
# 4. split() separates connected components so fragments can be dropped.
# 5. Open3D (point clouds/preprocessing) + trimesh (mesh QA) pair well together.

if __name__ == "__main__":
    main(out_dir=stage_out_dir(__doc__))
