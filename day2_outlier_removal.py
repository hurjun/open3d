"""
Day 2: Outlier Removal — 노이즈 점 제거
Statistical vs Radius 두 방법 비교 시각화
"""
import open3d as o3d
import numpy as np


def load_bunny_pcd(n_points: int = 50_000) -> o3d.geometry.PointCloud:
    """Stanford Bunny mesh → PCD 변환."""
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    return pcd


def add_noise(pcd: o3d.geometry.PointCloud, n_noise: int = 500) -> o3d.geometry.PointCloud:
    """실제 LiDAR 노이즈를 흉내내어 랜덤 점을 추가한다."""
    pts = np.asarray(pcd.points)
    # 기존 점들의 bounding box 범위 안에서 랜덤하게 노이즈 생성
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    noise_pts = np.random.uniform(mins, maxs, size=(n_noise, 3))

    noisy = o3d.geometry.PointCloud()
    noisy.points = o3d.utility.Vector3dVector(
        np.vstack([pts, noise_pts])
    )
    return noisy


def visualize_with_outliers(
    inlier: o3d.geometry.PointCloud,
    outlier: o3d.geometry.PointCloud,
    title: str,
) -> None:
    """inlier(회색)와 outlier(빨강)를 함께 시각화한다."""
    inlier.paint_uniform_color([0.6, 0.6, 0.6])
    outlier.paint_uniform_color([1.0, 0.0, 0.0])
    print(f"\n[{title}]  inlier={len(inlier.points):,}  outlier={len(outlier.points):,}")
    o3d.visualization.draw_geometries(
        [inlier, outlier],
        window_name=title,
        width=1024, height=768,
    )


# ── 준비: 노이즈 섞인 PCD 만들기 ────────────────────────────────────────────
np.random.seed(42)
pcd_clean = load_bunny_pcd()
pcd_noisy = add_noise(pcd_clean, n_noise=500)
pcd_noisy.paint_uniform_color([0.6, 0.6, 0.6])

print("=" * 60)
print(f"원본 포인트 수  : {len(pcd_clean.points):,}")
print(f"노이즈 추가 후  : {len(pcd_noisy.points):,}  (+500 noise)")
print("=" * 60)

print("\n[노이즈 포함 PCD — 노이즈가 보이는지 확인해보세요]")
o3d.visualization.draw_geometries(
    [pcd_noisy],
    window_name="노이즈 포함 원본",
    width=1024, height=768,
)


# ── 방법 1: Statistical Outlier Removal ─────────────────────────────────────
# nb_neighbors: 각 점에서 이웃 몇 개까지 볼지
# std_ratio: 평균 거리 + (std_ratio × σ) 초과 시 제거
# std_ratio 작을수록 공격적 (더 많이 제거)
print("\n" + "=" * 60)
print("방법 1: Statistical Outlier Removal")
print("  nb_neighbors=20, std_ratio=2.0")
print("=" * 60)

inlier_stat, inlier_idx_stat = pcd_noisy.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0,
)
# 반환값은 inlier 인덱스 — invert=True로 outlier 추출
outlier_stat = pcd_noisy.select_by_index(inlier_idx_stat, invert=True)

visualize_with_outliers(
    inlier_stat, outlier_stat,
    "Statistical — inlier(회색) + outlier(빨강)",
)


# ── 방법 2: Radius Outlier Removal ──────────────────────────────────────────
# nb_points: 반경 안에 이 수 이상 이웃이 없으면 제거
# radius: 탐색 반경
print("\n" + "=" * 60)
print("방법 2: Radius Outlier Removal")
print("  nb_points=16, radius=0.02")
print("=" * 60)

inlier_rad, inlier_idx_rad = pcd_noisy.remove_radius_outlier(
    nb_points=16,
    radius=0.02,
)
outlier_rad = pcd_noisy.select_by_index(inlier_idx_rad, invert=True)

visualize_with_outliers(
    inlier_rad, outlier_rad,
    "Radius — inlier(회색) + outlier(빨강)",
)


# ── 비교: 두 방법이 제거한 점 수 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("결과 비교")
print("=" * 60)
total = len(pcd_noisy.points)
print(f"  전체 포인트       : {total:,}")
print(f"  Statistical 제거  : {len(outlier_stat.points):,}개  ({len(outlier_stat.points)/total:.1%})")
print(f"  Radius 제거       : {len(outlier_rad.points):,}개  ({len(outlier_rad.points)/total:.1%})")

# ── 배운 점 ──────────────────────────────────────────────────────────────────
# 1. Statistical: 전체 밀도 분포의 통계로 이상치 판단 — 파라미터 튜닝이 직관적
# 2. Radius: 로컬 밀도 기준 — 포인트가 고르지 않은 실제 LiDAR에서 더 엄격하게 작동
# 3. remove_*_outlier는 (inlier_pcd, outlier_indices) 튜플을 반환
# 4. select_by_index로 outlier 점만 따로 추출해 빨간색으로 시각화 가능
# 5. 실무에서는 Statistical → Radius 순서로 2-pass 적용하는 경우가 많음
