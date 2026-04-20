"""
Day 7: laspy — 실제 LiDAR .las 파일 읽기
.las 포맷 구조 이해 + Open3D PointCloud로 변환 + classification 기반 분류
"""
import laspy
import numpy as np
import open3d as o3d
import tempfile
import os


# ── STEP 1: 샘플 .las 파일 생성 ─────────────────────────────────────────────
# 실제 현장에서는 이미 .las 파일이 있음
# 여기서는 실제 .las 구조와 동일하게 직접 생성
print("=" * 60)
print("STEP 1: 샘플 .las 파일 생성")
print("=" * 60)

rng = np.random.default_rng(42)

# 자율주행 장면 생성
# 지면 (classification=2)
n_ground = 8000
ground_x = rng.uniform(-20, 20, n_ground)
ground_y = rng.uniform(-20, 20, n_ground)
ground_z = rng.normal(0, 0.05, n_ground)
ground_cls = np.full(n_ground, 2, dtype=np.uint8)
ground_intensity = rng.integers(100, 300, n_ground, dtype=np.uint16)

# 건물 (classification=6)
n_building = 3000
building_x = rng.uniform(5, 10, n_building)
building_y = rng.uniform(5, 10, n_building)
building_z = rng.uniform(0, 8, n_building)
building_cls = np.full(n_building, 6, dtype=np.uint8)
building_intensity = rng.integers(200, 500, n_building, dtype=np.uint16)

# 식생 (classification=3)
n_veg = 2000
veg_x = rng.uniform(-10, -5, n_veg)
veg_y = rng.uniform(-10, -5, n_veg)
veg_z = rng.uniform(0, 4, n_veg)
veg_cls = np.full(n_veg, 3, dtype=np.uint8)
veg_intensity = rng.integers(50, 150, n_veg, dtype=np.uint16)

# 미분류 차량 (classification=0)
n_vehicle = 800
vehicle_x = rng.uniform(-2, 2, n_vehicle)
vehicle_y = rng.uniform(-2, 2, n_vehicle)
vehicle_z = rng.uniform(0, 1.5, n_vehicle)
vehicle_cls = np.full(n_vehicle, 0, dtype=np.uint8)
vehicle_intensity = rng.integers(150, 400, n_vehicle, dtype=np.uint16)

# 전체 합치기
all_x = np.concatenate([ground_x, building_x, veg_x, vehicle_x])
all_y = np.concatenate([ground_y, building_y, veg_y, vehicle_y])
all_z = np.concatenate([ground_z, building_z, veg_z, vehicle_z])
all_cls = np.concatenate([ground_cls, building_cls, veg_cls, vehicle_cls])
all_intensity = np.concatenate([ground_intensity, building_intensity, veg_intensity, vehicle_intensity])

# laspy로 .las 파일 작성
header = laspy.LasHeader(point_format=1, version="1.4")
header.offsets = np.array([all_x.min(), all_y.min(), all_z.min()])
header.scales = np.array([0.001, 0.001, 0.001])

las = laspy.LasData(header=header)
las.x = all_x
las.y = all_y
las.z = all_z
las.classification = all_cls
las.intensity = all_intensity

las_path = os.path.join(tempfile.gettempdir(), "sample_scene.las")
las.write(las_path)
print(f"  저장 경로  : {las_path}")
print(f"  총 포인트  : {len(all_x):,}")


# ── STEP 2: .las 파일 읽기 + 구조 탐색 ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: .las 파일 읽기 + 구조 탐색")
print("=" * 60)

las_read = laspy.read(las_path)

print(f"  포인트 수         : {len(las_read.x):,}")
print(f"  포맷 버전         : {las_read.header.version}")
print(f"  point format      : {las_read.header.point_format.id}")
print(f"  사용 가능한 필드  : {list(las_read.point_format.dimension_names)}")
print(f"\n  좌표 범위 X       : {las_read.x.min():.2f} ~ {las_read.x.max():.2f}")
print(f"  좌표 범위 Y       : {las_read.y.min():.2f} ~ {las_read.y.max():.2f}")
print(f"  좌표 범위 Z       : {las_read.z.min():.2f} ~ {las_read.z.max():.2f}")


# ── STEP 3: classification 분포 확인 ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Classification 분포")
print("=" * 60)

cls_labels = {0: "미분류", 2: "지면", 3: "식생", 6: "건물"}
cls_array = np.array(las_read.classification)

for code, label in cls_labels.items():
    count = (cls_array == code).sum()
    print(f"  [{code}] {label:6s} : {count:,}개  ({count/len(cls_array):.1%})")


# ── STEP 4: laspy → Open3D PointCloud 변환 ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: laspy → Open3D 변환")
print("=" * 60)

# x, y, z를 (N, 3) 배열로 합치기
xyz = np.column_stack([
    np.array(las_read.x),
    np.array(las_read.y),
    np.array(las_read.z),
])

# classification 기반 색상 매핑
# 지면=초록, 건물=파랑, 식생=연두, 미분류=빨강
color_map = {
    0: [1.0, 0.3, 0.3],  # 빨강: 미분류 (차량)
    2: [0.3, 0.8, 0.3],  # 초록: 지면
    3: [0.5, 1.0, 0.2],  # 연두: 식생
    6: [0.3, 0.3, 1.0],  # 파랑: 건물
}
colors = np.array([color_map.get(c, [0.5, 0.5, 0.5]) for c in cls_array])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"  Open3D PCD 포인트 수: {len(pcd.points):,}")
print(f"  color 있음          : {pcd.has_colors()}")


# ── STEP 5: classification별 분리 + 시각화 ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: 시각화")
print("  초록=지면 / 파랑=건물 / 연두=식생 / 빨강=미분류(차량)")
print("=" * 60)

print("\n[전체 PCD — classification 색상]")
o3d.visualization.draw_geometries(
    [pcd],
    window_name="전체 PCD (classification 색상)",
    width=1024, height=768,
)

# 지면만 따로
ground_mask = cls_array == 2
pcd_ground = o3d.geometry.PointCloud()
pcd_ground.points = o3d.utility.Vector3dVector(xyz[ground_mask])
pcd_ground.paint_uniform_color([0.3, 0.8, 0.3])

# 지면 제외
pcd_objects = o3d.geometry.PointCloud()
pcd_objects.points = o3d.utility.Vector3dVector(xyz[~ground_mask])
pcd_objects.colors = o3d.utility.Vector3dVector(colors[~ground_mask])

print("\n[지면 제거 후 — 건물/식생/차량만 남음]")
o3d.visualization.draw_geometries(
    [pcd_objects],
    window_name="지면 제거 후 (건물/식생/차량만)",
    width=1024, height=768,
)


# ── STEP 6: intensity 활용 ───────────────────────────────────────────────────
# intensity = 레이저 반사 강도. 재질 구분에 활용 가능
# 금속(차량) > 콘크리트(건물) > 아스팔트(지면) > 식생 순으로 강함
print("\n" + "=" * 60)
print("STEP 6: Intensity 분포")
print("=" * 60)

intensity = np.array(las_read.intensity)
for code, label in cls_labels.items():
    mask = cls_array == code
    if mask.sum() > 0:
        print(f"  [{code}] {label:6s} 평균 intensity: {intensity[mask].mean():.1f}")

print("\n완료.")

# ── 배운 점 ──────────────────────────────────────────────────────────────────
# 1. .las 파일은 x/y/z 외에 intensity, classification, return_number 등 필드를 가짐
# 2. classification 코드로 지면/건물/식생이 이미 구분된 경우가 많음 — RANSAC 전에 활용 가능
# 3. laspy로 읽은 x/y/z를 np.column_stack으로 합치면 바로 Open3D PCD로 변환 가능
# 4. intensity는 재질 구분에 활용 — 금속(차량)은 반사 강도가 높음
# 5. 실무 파이프라인: laspy 읽기 → classification 필터 → Open3D 전처리 → Mesh 변환
