"""
Day 1: PointCloud 기초 — 로드, 구조 탐색, 시각화, 전처리 입문
면접 대비: Open3D의 핵심 데이터 타입과 기본 파이프라인 감각 잡기
"""
import open3d as o3d
import numpy as np


def load_and_inspect(pcd: o3d.geometry.PointCloud, label: str) -> None:
    """PCD 기본 정보를 출력한다."""
    pts = np.asarray(pcd.points)
    print(f"\n[{label}]")
    print(f"  포인트 수       : {len(pts):,}")
    print(f"  좌표 범위 X     : {pts[:, 0].min():.3f} ~ {pts[:, 0].max():.3f}")
    print(f"  좌표 범위 Y     : {pts[:, 1].min():.3f} ~ {pts[:, 1].max():.3f}")
    print(f"  좌표 범위 Z     : {pts[:, 2].min():.3f} ~ {pts[:, 2].max():.3f}")
    print(f"  color 있음      : {pcd.has_colors()}")
    print(f"  normal 있음     : {pcd.has_normals()}")


# ── 1. 샘플 데이터 로드 ──────────────────────────────────────────────────────
# Open3D 내장 샘플: Stanford Bunny (토끼 모양 PCD, 연구용 표준 벤치마크 모델)
# 실무에서는 이 자리에 laspy로 읽은 .las 파일이 들어온다
print("=" * 60)
print("STEP 1: 샘플 데이터 로드")
print("=" * 60)

demo_data = o3d.data.BunnyMesh()
# BunnyMesh는 mesh 파일이지만, PCD로 변환해서 사용
mesh = o3d.io.read_triangle_mesh(demo_data.path)
mesh.compute_vertex_normals()

# mesh → PCD 변환 (표면에서 균일하게 샘플링)
# number_of_points: 샘플링할 포인트 수
pcd_original = mesh.sample_points_uniformly(number_of_points=100_000)
load_and_inspect(pcd_original, "원본 PCD")


# ── 2. Voxel Downsampling ────────────────────────────────────────────────────
# 3D 공간을 voxel_size 크기 격자로 나누고 각 격자 내 점들을 무게중심 하나로 대체
# 효과: 포인트 수 감소 + 밀도 균일화 / 단점: edge 디테일 손실
print("\n" + "=" * 60)
print("STEP 2: Voxel Downsampling")
print("=" * 60)

voxel_size = 0.005  # 단위는 mesh 좌표계 기준 (Bunny는 ~0.3 크기)
pcd_down = pcd_original.voxel_down_sample(voxel_size=voxel_size)
load_and_inspect(pcd_down, f"Downsampled (voxel_size={voxel_size})")

ratio = len(np.asarray(pcd_down.points)) / len(np.asarray(pcd_original.points))
print(f"  압축률          : {ratio:.1%} ({len(np.asarray(pcd_original.points)):,} → {len(np.asarray(pcd_down.points)):,})")


# ── 3. Normal Estimation ─────────────────────────────────────────────────────
# 각 점 주변의 이웃 점들로 로컬 평면을 피팅 → 그 평면의 법선 벡터 계산 (PCA 기반)
# KDTreeSearchParamHybrid: radius 내에서 최대 max_nn개 이웃을 탐색
print("\n" + "=" * 60)
print("STEP 3: Normal Estimation")
print("=" * 60)

pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.01,   # 이 반경 내 이웃점으로 평면 피팅
        max_nn=30      # 최대 이웃 수 (너무 크면 곡면에서 부정확)
    )
)
# 법선 방향을 카메라(원점) 기준으로 정렬 — 방향이 뒤집히는 ambiguity 해결
pcd_down.orient_normals_towards_camera_location(np.array([0.0, 0.0, 1.0]))
load_and_inspect(pcd_down, "Normal 추정 후")

normals = np.asarray(pcd_down.normals)
print(f"  normal 벡터 샘플: {normals[0].round(3)}")


# ── 4. 시각화 ────────────────────────────────────────────────────────────────
# draw_geometries: Open3D 기본 뷰어 (마우스로 회전/확대 가능)
# - 좌클릭 드래그: 회전
# - 우클릭 드래그: 이동
# - 스크롤: 확대/축소
print("\n" + "=" * 60)
print("STEP 4: 시각화")
print("  조작법: 좌클릭=회전, 우클릭=이동, 스크롤=확대, Q=종료")
print("=" * 60)

# 원본과 다운샘플된 PCD를 색상으로 구분해 비교
pcd_original.paint_uniform_color([0.7, 0.7, 0.7])  # 회색: 원본
pcd_down.paint_uniform_color([1.0, 0.3, 0.3])      # 빨강: 다운샘플

print("\n[원본 PCD 시각화]")
o3d.visualization.draw_geometries(
    [pcd_original],
    window_name="Day1 - 원본 PCD (100,000 pts)",
    width=1024, height=768,
)

print("\n[Downsampled + Normal 시각화]")
o3d.visualization.draw_geometries(
    [pcd_down],
    window_name="Day1 - Downsampled PCD + Normals",
    width=1024, height=768,
    point_show_normal=True,  # 각 점에서 법선 벡터를 화살표로 표시
)

print("\n[원본 vs 다운샘플 비교]")
# 두 PCD를 x축으로 0.15 이동해서 나란히 표시
pcd_shifted = o3d.geometry.PointCloud(pcd_down)
pcd_shifted.translate([0.15, 0, 0])
o3d.visualization.draw_geometries(
    [pcd_original, pcd_shifted],
    window_name="Day1 - 원본(회색) vs Downsampled(빨강)",
    width=1200, height=768,
)


# ── 5. 핵심 수치 요약 ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: 결과 요약")
print("=" * 60)
pts_orig = len(np.asarray(pcd_original.points))
pts_down = len(np.asarray(pcd_down.points))
print(f"  원본 포인트 수   : {pts_orig:,}")
print(f"  다운샘플 후      : {pts_down:,}  (voxel_size={voxel_size})")
print(f"  감소율           : {(1 - pts_down/pts_orig):.1%}")
print(f"  normal 추정 완료 : {pcd_down.has_normals()}")

# ── 배운 점 ──────────────────────────────────────────────────────────────────
# 1. PointCloud = (N,3) 좌표 배열의 래퍼. .points/.colors/.normals로 numpy 접근
# 2. voxel_down_sample은 3D 격자 기반 포인트 압축. voxel_size ↑ → 압축률 ↑ → 디테일 손실
# 3. estimate_normals은 KDTree로 이웃 탐색 후 PCA로 로컬 평면 법선 계산
# 4. orient_normals_towards_camera_location으로 법선 방향 ambiguity 해결
# 5. draw_geometries에 여러 geometry를 리스트로 넘기면 동시 렌더링 가능
