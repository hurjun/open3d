"""
Day 4: ICP (Iterative Closest Point) — 두 PCD 정합
같은 물체를 다른 위치에서 스캔한 두 PCD를 자동으로 겹치는 알고리즘
"""
import open3d as o3d
import numpy as np


def load_bunny() -> o3d.geometry.PointCloud:
    """Stanford Bunny PCD 로드 + voxel downsampling."""
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=30_000)
    return pcd.voxel_down_sample(voxel_size=0.005)


def apply_transform(pcd: o3d.geometry.PointCloud, tx: float, ty: float, tz: float, angle_deg: float) -> o3d.geometry.PointCloud:
    """PCD를 이동(tx, ty, tz) + z축 회전(angle_deg)시킨 복사본 반환."""
    pcd2 = o3d.geometry.PointCloud(pcd)

    # z축 기준 회전 행렬
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1],
    ])
    # 4x4 변환 행렬 (회전 + 이동)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    pcd2.transform(T)
    return pcd2


# ── 준비: source(움직일 것)와 target(고정) 만들기 ────────────────────────────
pcd_target = load_bunny()

# source = target을 살짝 이동 + 회전시킨 것 (현실에서는 다른 시점에서 스캔한 것)
pcd_source = apply_transform(pcd_target, tx=0.05, ty=0.03, tz=0.01, angle_deg=15)

pcd_target.paint_uniform_color([0.3, 0.6, 1.0])  # 파랑: target (고정)
pcd_source.paint_uniform_color([1.0, 0.4, 0.3])  # 빨강: source (맞춰야 할 것)

print("=" * 60)
print("ICP 실험 설정")
print(f"  target(파랑): 고정된 기준 PCD")
print(f"  source(빨강): 15도 회전 + 이동된 PCD  ← ICP로 target에 맞춤")
print("=" * 60)

print("\n[ICP 전 — 빨강이 파랑과 어긋나 있음]")
o3d.visualization.draw_geometries(
    [pcd_target, pcd_source],
    window_name="ICP 전: target(파랑) vs source(빨강)",
    width=1024, height=768,
)


# ── STEP 1: Point-to-Point ICP ───────────────────────────────────────────────
# 각 점에서 상대방의 가장 가까운 점까지 거리를 최소화
# max_correspondence_distance: 이 거리 이상 떨어진 점은 매칭 후보에서 제외
print("\n" + "=" * 60)
print("STEP 1: Point-to-Point ICP")
print("=" * 60)

result_p2p = o3d.pipelines.registration.registration_icp(
    source=pcd_source,
    target=pcd_target,
    max_correspondence_distance=0.02,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
)

print(f"  수렴 여부        : {result_p2p.fitness > 0.9}")
print(f"  fitness          : {result_p2p.fitness:.4f}  (1.0이 완벽 정합, 매칭된 점 비율)")
print(f"  inlier_rmse      : {result_p2p.inlier_rmse:.6f}  (낮을수록 정확)")
print(f"  검출된 변환 행렬:\n{result_p2p.transformation.round(3)}")

# source에 결과 변환 행렬 적용
pcd_source_p2p = o3d.geometry.PointCloud(pcd_source)
pcd_source_p2p.transform(result_p2p.transformation)
pcd_source_p2p.paint_uniform_color([1.0, 0.4, 0.3])

print("\n[Point-to-Point ICP 결과 — 빨강이 파랑에 겹쳐있으면 성공]")
o3d.visualization.draw_geometries(
    [pcd_target, pcd_source_p2p],
    window_name="Point-to-Point ICP 결과",
    width=1024, height=768,
)


# ── STEP 2: Point-to-Plane ICP ───────────────────────────────────────────────
# 점까지 거리가 아니라 상대방 평면(normal)까지 거리를 최소화 → 더 정확
# normal이 필요하므로 먼저 estimate_normals 실행
print("\n" + "=" * 60)
print("STEP 2: Point-to-Plane ICP")
print("=" * 60)

pcd_source_n = o3d.geometry.PointCloud(pcd_source)
pcd_target_n = o3d.geometry.PointCloud(pcd_target)

for p in [pcd_source_n, pcd_target_n]:
    p.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )

result_p2l = o3d.pipelines.registration.registration_icp(
    source=pcd_source_n,
    target=pcd_target_n,
    max_correspondence_distance=0.02,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
)

print(f"  fitness          : {result_p2l.fitness:.4f}")
print(f"  inlier_rmse      : {result_p2l.inlier_rmse:.6f}")

pcd_source_p2l = o3d.geometry.PointCloud(pcd_source)
pcd_source_p2l.transform(result_p2l.transformation)
pcd_source_p2l.paint_uniform_color([1.0, 0.4, 0.3])

print("\n[Point-to-Plane ICP 결과]")
o3d.visualization.draw_geometries(
    [pcd_target, pcd_source_p2l],
    window_name="Point-to-Plane ICP 결과",
    width=1024, height=768,
)


# ── 비교 요약 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("결과 비교")
print("=" * 60)
print(f"  Point-to-Point  fitness={result_p2p.fitness:.4f}  rmse={result_p2p.inlier_rmse:.6f}")
print(f"  Point-to-Plane  fitness={result_p2l.fitness:.4f}  rmse={result_p2l.inlier_rmse:.6f}")
print()
print("  fitness 가까울수록 1.0 = 잘 맞춰짐")
print("  rmse 낮을수록 = 매칭 오차 작음")

# ── 배운 점 ──────────────────────────────────────────────────────────────────
# 1. ICP는 source를 target에 맞추는 변환 행렬(4x4)을 반환
# 2. fitness = 매칭된 점 수 / 전체 점 수 — 1에 가까울수록 잘 정합됨
# 3. inlier_rmse = 매칭된 쌍들의 평균 거리 오차 — 낮을수록 정밀
# 4. Point-to-Plane이 normal을 활용해 일반적으로 더 정확하고 빠르게 수렴
# 5. ICP는 초기값이 중요 — 너무 멀리 떨어지면 엉뚱한 점끼리 매칭되어 발산
