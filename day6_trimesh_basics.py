"""
Day 6: trimesh 기초 — Mesh 품질 검증
Open3D로 만든 Mesh를 trimesh로 분석하고 품질을 검증한다
JD 키워드: "Mesh 재구성, 데이터 정합 및 품질 검증 로직 개발"
"""
import open3d as o3d
import trimesh
import numpy as np


def make_mesh_with_open3d() -> o3d.geometry.TriangleMesh:
    """Open3D로 Stanford Bunny Mesh 로드."""
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    mesh.compute_vertex_normals()
    return mesh


def o3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Open3D Mesh → trimesh 변환."""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ── 준비: Mesh 로드 ──────────────────────────────────────────────────────────
print("=" * 60)
print("Open3D Mesh → trimesh 변환")
print("=" * 60)

o3d_mesh = make_mesh_with_open3d()
mesh = o3d_to_trimesh(o3d_mesh)

print(f"  vertices  : {len(mesh.vertices):,}")
print(f"  faces     : {len(mesh.faces):,}")


# ── STEP 1: Mesh 품질 검증 ───────────────────────────────────────────────────
# MORAI JD: "품질 검증 로직 개발"
# 디지털트윈에 올리기 전에 Mesh가 정상인지 확인하는 과정
print("\n" + "=" * 60)
print("STEP 1: Mesh 품질 검증")
print("=" * 60)

# watertight: Mesh가 완전히 닫혀있는가 (구멍이 없는가)
# 시뮬레이터에서 레이캐스팅, 충돌 계산에 watertight Mesh가 필요
print(f"  watertight (구멍 없음)   : {mesh.is_watertight}")

# winding consistent: 모든 삼각형의 법선이 같은 방향인가
print(f"  winding consistent       : {mesh.is_winding_consistent}")

# volume: watertight일 때만 의미있음 (단위: m³)
if mesh.is_watertight:
    print(f"  volume (부피)            : {mesh.volume:.6f} m³")
else:
    print(f"  volume                   : 계산 불가 (watertight 아님)")

# bounds: bounding box 범위
print(f"  bounds min               : {mesh.bounds[0].round(3)}")
print(f"  bounds max               : {mesh.bounds[1].round(3)}")

# 무게중심
print(f"  center_mass (무게중심)   : {mesh.center_mass.round(3)}")


# ── STEP 2: 구멍(hole) 찾기 ─────────────────────────────────────────────────
# boundary_edges: 한쪽 면에만 연결된 edge = 구멍의 경계
print("\n" + "=" * 60)
print("STEP 2: 구멍(hole) 탐지")
print("=" * 60)

boundary_edges = trimesh.grouping.group_rows(
    mesh.edges_sorted, require_count=1
)
print(f"  boundary edge 수 : {len(boundary_edges)}")
if len(boundary_edges) == 0:
    print("  → 구멍 없음. 완전한 Mesh.")
else:
    print(f"  → 구멍 있음. {len(boundary_edges)}개 edge가 닫히지 않음.")


# ── STEP 3: 컴포넌트 분리 ────────────────────────────────────────────────────
# 연결되지 않은 덩어리가 몇 개인지 확인
# 예: 토끼 몸통 + 분리된 귀 파편 = 2개 컴포넌트
print("\n" + "=" * 60)
print("STEP 3: 연결 컴포넌트 분석")
print("=" * 60)

components = mesh.split(only_watertight=False, repair=False)
print(f"  컴포넌트 수 : {len(components)}")
for i, comp in enumerate(components):
    print(f"  컴포넌트 {i}  : vertices={len(comp.vertices):,}  faces={len(comp.faces):,}")


# ── STEP 4: 가장 큰 컴포넌트만 남기기 ──────────────────────────────────────
# 품질 검증 후 파편 제거 — 디지털트윈에는 메인 물체만 올림
print("\n" + "=" * 60)
print("STEP 4: 메인 컴포넌트 추출 (파편 제거)")
print("=" * 60)

main_mesh = max(components, key=lambda c: len(c.faces))
print(f"  메인 컴포넌트 faces : {len(main_mesh.faces):,}")
print(f"  watertight          : {main_mesh.is_watertight}")


# ── STEP 5: 시각화 (Open3D 뷰어 사용) ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: 시각화")
print("=" * 60)

def trimesh_to_o3d(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """trimesh → Open3D Mesh 변환."""
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(tm.vertices)
    m.triangles = o3d.utility.Vector3iVector(tm.faces)
    m.compute_vertex_normals()
    return m

print("\n[전체 Mesh — boundary edge(구멍 경계)를 빨간색으로 표시]")
o3d_full = trimesh_to_o3d(mesh)
o3d_full.paint_uniform_color([0.7, 0.7, 0.7])
o3d.visualization.draw_geometries(
    [o3d_full],
    window_name="전체 Mesh",
    width=1024, height=768,
    mesh_show_back_face=True,
)

print("\n[메인 컴포넌트만]")
o3d_main = trimesh_to_o3d(main_mesh)
o3d_main.paint_uniform_color([0.7, 0.7, 0.7])
o3d.visualization.draw_geometries(
    [o3d_main],
    window_name="메인 컴포넌트",
    width=1024, height=768,
    mesh_show_back_face=True,
)


# ── STEP 6: 품질 검증 요약 리포트 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("품질 검증 리포트")
print("=" * 60)
print(f"  vertices          : {len(mesh.vertices):,}")
print(f"  faces             : {len(mesh.faces):,}")
print(f"  watertight        : {'✅' if mesh.is_watertight else '❌'}")
print(f"  winding consistent: {'✅' if mesh.is_winding_consistent else '❌'}")
print(f"  boundary edges    : {len(boundary_edges)} {'✅' if len(boundary_edges) == 0 else '❌'}")
print(f"  컴포넌트 수       : {len(components)} {'✅' if len(components) == 1 else '⚠️ 파편 있음'}")

# ── 배운 점 ──────────────────────────────────────────────────────────────────
# 1. trimesh는 Mesh 품질 검증에 특화 — watertight, winding, volume, boundary 검사
# 2. watertight = 구멍 없는 닫힌 Mesh — 시뮬레이터 레이캐스팅에 필수
# 3. boundary_edges = 한쪽만 연결된 edge = 구멍의 경계
# 4. split()으로 연결 컴포넌트 분리 → 파편 제거 가능
# 5. Open3D(PCD/전처리) + trimesh(Mesh 품질검증) 조합이 MORAI JD 핵심 스택
