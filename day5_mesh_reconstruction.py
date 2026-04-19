"""
Day 5: PCD → Mesh 변환 (Poisson Surface Reconstruction)
Point Cloud의 점들을 삼각형으로 연결해 연속적인 표면을 만든다
"""
import open3d as o3d
import numpy as np


def load_bunny_pcd() -> o3d.geometry.PointCloud:
    """Stanford Bunny PCD 로드 + normal 추정."""
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=50_000)
    pcd = pcd.voxel_down_sample(voxel_size=0.003)

    # Poisson reconstruction은 normal이 반드시 필요
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 1.0]))
    return pcd


# ── 준비: PCD 로드 ───────────────────────────────────────────────────────────
pcd = load_bunny_pcd()
print("=" * 60)
print(f"PCD 포인트 수: {len(pcd.points):,}")
print(f"normal 있음 : {pcd.has_normals()}")
print("=" * 60)

print("\n[원본 PCD]")
o3d.visualization.draw_geometries(
    [pcd],
    window_name="원본 PCD",
    width=1024, height=768,
    point_show_normal=False,
)


# ── STEP 1: Poisson Surface Reconstruction ───────────────────────────────────
# depth: 재구성 해상도. 높을수록 디테일하지만 느리고 메모리 많이 씀 (보통 8~11)
# densities: 각 vertex가 얼마나 많은 점으로 지지되는지 (낮으면 신뢰도 낮은 영역)
print("\n" + "=" * 60)
print("STEP 1: Poisson Surface Reconstruction (depth=9)")
print("=" * 60)

mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=9,        # 재구성 해상도
    width=0,
    scale=1.1,
    linear_fit=False,
)

print(f"  vertices  (꼭짓점) : {len(mesh_poisson.vertices):,}")
print(f"  triangles (삼각형) : {len(mesh_poisson.triangles):,}")


# ── STEP 2: 저밀도 영역 제거 ─────────────────────────────────────────────────
# Poisson은 PCD 바깥쪽에도 가상의 표면을 만들 수 있음
# densities가 낮은 vertex = 실제 점이 거의 없는 허구 영역 → 제거
print("\n" + "=" * 60)
print("STEP 2: 저밀도 영역 제거")
print("=" * 60)

densities = np.asarray(densities)
density_threshold = np.quantile(densities, 0.05)  # 하위 5% 제거
vertices_to_remove = densities < density_threshold
mesh_clean = mesh_poisson.remove_vertices_by_mask(vertices_to_remove)
mesh_clean.compute_vertex_normals()

print(f"  제거 전 vertices: {len(mesh_poisson.vertices):,}")
print(f"  제거 후 vertices: {len(mesh_clean.vertices):,}")
print(f"  제거 후 triangles: {len(mesh_clean.triangles):,}")


# ── STEP 3: Mesh 구조 탐색 ───────────────────────────────────────────────────
# PCD의 .points 처럼 Mesh도 numpy로 접근 가능
print("\n" + "=" * 60)
print("STEP 3: Mesh 내부 구조")
print("=" * 60)

verts = np.asarray(mesh_clean.vertices)
tris = np.asarray(mesh_clean.triangles)
print(f"  vertices shape  : {verts.shape}  ← (N, 3) 꼭짓점 좌표")
print(f"  triangles shape : {tris.shape}  ← (M, 3) 삼각형 꼭짓점 인덱스")
print(f"  삼각형 0번      : {tris[0]}  ← 꼭짓점 {tris[0][0]}, {tris[0][1]}, {tris[0][2]}번을 연결")


# ── STEP 4: 시각화 비교 ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: 시각화")
print("=" * 60)

# density를 색상으로 표현 (낮은=파랑, 높은=노랑)
density_colors = (densities - densities.min()) / (densities.max() - densities.min())
density_color_map = np.column_stack([
    density_colors,
    density_colors * 0.5,
    1.0 - density_colors,
])
mesh_poisson.vertex_colors = o3d.utility.Vector3dVector(density_color_map)

print("\n[Poisson Mesh — 색상=density (파랑=낮음/허구영역, 노랑=높음/신뢰영역)]")
o3d.visualization.draw_geometries(
    [mesh_poisson],
    window_name="Poisson Mesh (density 시각화)",
    width=1024, height=768,
    mesh_show_back_face=True,
)

mesh_clean.paint_uniform_color([0.7, 0.7, 0.7])
print("\n[저밀도 제거 후 최종 Mesh]")
o3d.visualization.draw_geometries(
    [mesh_clean],
    window_name="최종 Mesh (저밀도 제거)",
    width=1024, height=768,
    mesh_show_back_face=True,
)

print("\n[PCD vs Mesh 나란히 비교]")
pcd_vis = o3d.geometry.PointCloud(pcd)
pcd_vis.paint_uniform_color([1.0, 0.4, 0.3])
mesh_shifted = o3d.geometry.TriangleMesh(mesh_clean)
mesh_shifted.translate([0.2, 0, 0])
o3d.visualization.draw_geometries(
    [pcd_vis, mesh_shifted],
    window_name="PCD(빨강) vs Mesh(회색)",
    width=1200, height=768,
    mesh_show_back_face=True,
)

# ── 배운 점 ──────────────────────────────────────────────────────────────────
# 1. Poisson reconstruction은 normal을 이용해 PCD를 watertight Mesh로 변환
# 2. depth가 높을수록 디테일하지만 느림 — 실무에서는 8~10이 적당
# 3. densities가 낮은 vertex는 실제 점이 없는 허구 영역 → 제거 필요
# 4. Mesh의 핵심 구조: vertices(꼭짓점 좌표) + triangles(꼭짓점 인덱스 3개)
# 5. PCD는 인식/분석용, Mesh는 렌더링/시뮬레이션용 — MORAI에서 둘 다 쓰임
