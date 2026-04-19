"""
Day 3: RANSAC 지면 검출 + DBSCAN 클러스터링
자율주행 핵심 파이프라인: 지면 제거 → 객체 분리
"""
import open3d as o3d
import numpy as np


def make_scene() -> o3d.geometry.PointCloud:
    """
    가상의 자율주행 장면을 만든다.
    - 지면: z=0 평면에 넓게 분포
    - 객체 3개: 지면 위에 올려진 박스 형태 점군
    """
    rng = np.random.default_rng(42)

    # 지면: z=0 평면, 10x10m 범위, 5000점
    ground = np.column_stack([
        rng.uniform(-5, 5, 5000),   # x
        rng.uniform(-5, 5, 5000),   # y
        rng.normal(0, 0.02, 5000),  # z ≈ 0 (약간의 노이즈)
    ])

    # 객체 1: 차량 크기 박스 (x=1, y=0, z=0.75 중심)
    obj1 = rng.uniform([-0.5, -1.0, 0.0], [0.5, 1.0, 1.5], (800, 3))
    obj1 += [1.0, 0.0, 0.0]

    # 객체 2: 보행자 크기 박스 (x=-2, y=1)
    obj2 = rng.uniform([-0.2, -0.2, 0.0], [0.2, 0.2, 1.8], (300, 3))
    obj2 += [-2.0, 1.0, 0.0]

    # 객체 3: 작은 물체 (x=3, y=-2)
    obj3 = rng.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.6], (200, 3))
    obj3 += [3.0, -2.0, 0.0]

    all_pts = np.vstack([ground, obj1, obj2, obj3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    return pcd


# ── 장면 생성 ────────────────────────────────────────────────────────────────
pcd = make_scene()
print("=" * 60)
print(f"전체 포인트 수: {len(pcd.points):,}")
print("=" * 60)

pcd.paint_uniform_color([0.6, 0.6, 0.6])
print("\n[원본 장면 — 지면 + 객체 3개]")
o3d.visualization.draw_geometries(
    [pcd],
    window_name="원본 장면",
    width=1024, height=768,
)


# ── STEP 1: RANSAC 지면 검출 ─────────────────────────────────────────────────
# segment_plane 반환값:
#   plane_model: [a, b, c, d] — 평면 방정식 ax+by+cz+d=0
#   inliers: 지면에 속하는 점들의 인덱스
print("\n" + "=" * 60)
print("STEP 1: RANSAC 지면 검출")
print("=" * 60)

plane_model, inlier_idx = pcd.segment_plane(
    distance_threshold=0.05,  # 이 거리 이내면 지면으로 판단 (단위: m)
    ransac_n=3,               # 매 시도마다 랜덤하게 뽑는 점 수 (평면은 3점으로 결정)
    num_iterations=1000,      # 반복 횟수 — 많을수록 정확하지만 느림
)

a, b, c, d = plane_model
print(f"  검출된 평면 방정식: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
print(f"  법선 벡터 (a,b,c) : ({a:.3f}, {b:.3f}, {c:.3f})  ← z축과 얼마나 가까운지 확인")
print(f"  지면 포인트 수    : {len(inlier_idx):,}")

ground_pcd = pcd.select_by_index(inlier_idx)
objects_pcd = pcd.select_by_index(inlier_idx, invert=True)  # 지면 제외한 나머지

ground_pcd.paint_uniform_color([0.3, 0.8, 0.3])   # 초록: 지면
objects_pcd.paint_uniform_color([0.8, 0.3, 0.3])  # 빨강: 객체

print(f"  비지면 포인트 수  : {len(objects_pcd.points):,}")

print("\n[RANSAC 결과 — 초록=지면 / 빨강=객체]")
o3d.visualization.draw_geometries(
    [ground_pcd, objects_pcd],
    window_name="RANSAC 지면 검출",
    width=1024, height=768,
)


# ── STEP 2: DBSCAN 클러스터링 ────────────────────────────────────────────────
# 지면 제거 후 남은 점들을 클러스터로 묶는다
# labels: 각 점에 클러스터 번호 부여. -1은 어느 클러스터에도 속하지 않는 노이즈
print("\n" + "=" * 60)
print("STEP 2: DBSCAN 클러스터링 (지면 제외 점군)")
print("=" * 60)

labels = np.array(
    objects_pcd.cluster_dbscan(
        eps=0.3,        # 이 반경 안의 이웃을 같은 클러스터로 봄
        min_points=10,  # core point 조건: 반경 안에 최소 10개 이웃
        print_progress=False,
    )
)

n_clusters = labels.max() + 1  # -1(노이즈) 제외
n_noise = (labels == -1).sum()
print(f"  검출된 클러스터 수: {n_clusters}")
print(f"  노이즈 포인트 수  : {n_noise}")

for i in range(n_clusters):
    count = (labels == i).sum()
    print(f"  클러스터 {i}         : {count}개 포인트")


# ── STEP 3: 클러스터별 색상 시각화 ──────────────────────────────────────────
# 클러스터마다 다른 색을 입혀서 구분
print("\n" + "=" * 60)
print("STEP 3: 클러스터 시각화")
print("=" * 60)

# 클러스터 수만큼 색상 생성
palette = np.random.default_rng(0).uniform(0.3, 1.0, size=(n_clusters, 3))
# 노이즈(-1)는 검정
colors = np.where(
    (labels[:, None] == -1),  # 노이즈 조건
    [0.0, 0.0, 0.0],          # 검정
    palette[np.clip(labels, 0, n_clusters - 1)],  # 클러스터 색
)
objects_pcd.colors = o3d.utility.Vector3dVector(colors)

print("\n[최종 결과 — 초록=지면 / 클러스터별 색상 / 검정=노이즈]")
o3d.visualization.draw_geometries(
    [ground_pcd, objects_pcd],
    window_name="RANSAC + DBSCAN 최종 결과",
    width=1024, height=768,
)

print("\n완료.")

# ── 배운 점 ──────────────────────────────────────────────────────────────────
# 1. RANSAC은 랜덤 샘플링으로 outlier에 강건한 평면을 피팅 — 지면 검출의 표준 방법
# 2. segment_plane 반환값의 inlier_idx로 지면/비지면을 select_by_index로 분리
# 3. DBSCAN은 클러스터 수를 사전에 지정 불필요 — 자율주행처럼 객체 수 미지수인 환경에 적합
# 4. labels==-1인 점은 어느 클러스터에도 속하지 않는 노이즈
# 5. eps와 min_points 튜닝이 핵심 — eps 너무 크면 객체들이 하나로 합쳐짐
