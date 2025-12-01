from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import open3d as o3d

# -------------------------
# 1. Load color, real depth, intrinsics
# -------------------------
color_image = cv2.imread("../data/color_image.png")
real_depth = np.load("../data/depth_data.npy") / 1000.0  # mm -> m
intrinsic_matrix = np.load("../data/intrinsic_matrix.npy")

image_height, image_width = color_image.shape[:2]

# Use only valid (>0) real depth for stats
valid_real_mask = np.isfinite(real_depth) & (real_depth > 0)
real_min_depth = real_depth[valid_real_mask].min()
real_max_depth = real_depth[valid_real_mask].max()
print("valid real depth min:", real_min_depth, "valid real depth max:", real_max_depth)

# -------------------------
# 2. Run Depth Anything V2
# -------------------------
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

output = pipe(image_pil, return_dict=True)
print("keys of output:", output.keys())

relative_depth_map = np.array(output["predicted_depth"]).astype(np.float32)
if relative_depth_map.ndim == 3:
    relative_depth_map = relative_depth_map[0]

print("Relative depth map shape:", relative_depth_map.shape)
print("Relative depth map min value:", np.min(relative_depth_map))
print("Relative depth map max value:", np.max(relative_depth_map))

vis_image = np.array(output["depth"])
cv2.imwrite("../data/predicted_depth_vis.png", vis_image)

# -------------------------
# 3. Affine alignment (least-squares)
# -------------------------
valid_mask = np.isfinite(real_depth) & np.isfinite(relative_depth_map) & (real_depth > 0)

x = relative_depth_map[valid_mask].flatten().astype(np.float32)  # predicted
y = real_depth[valid_mask].flatten().astype(np.float32)          # real

x_mean = x.mean()
y_mean = y.mean()

num = ((x - x_mean) * (y - y_mean)).sum()
den = ((x - x_mean) ** 2).sum() + 1e-8

a = num / den
b = y_mean - a * x_mean

print("Affine alignment parameters: a (scale) =", a, ", b (shift) =", b)

aligned_depth = a * relative_depth_map + b

# Clip using valid real depth range
aligned_depth = np.clip(aligned_depth, real_min_depth, real_max_depth)
aligned_depth = aligned_depth.astype(np.float32)
print("Aligned depth min:", aligned_depth.min(), "max:", aligned_depth.max())

np.save("../data/depth_aligned.npy", aligned_depth)

aligned_norm = (aligned_depth - real_min_depth) / (real_max_depth - real_min_depth + 1e-8)
aligned_norm = np.clip(aligned_norm, 0.0, 1.0)
aligned_vis = (aligned_norm * 255.0).astype(np.uint8)
cv2.imwrite("../data/depth_aligned_vis.png", aligned_vis)

# -------------------------
# 4. Visualize PCD from aligned depth with Open3D
# -------------------------
intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.set_intrinsics(
    width=image_width,
    height=image_height,
    fx=float(intrinsic_matrix[0, 0]),
    fy=float(intrinsic_matrix[1, 1]),
    cx=float(intrinsic_matrix[0, 2]),
    cy=float(intrinsic_matrix[1, 2]),
)

# Open3D expects contiguous float32 image
depth_o3d = o3d.geometry.Image(np.ascontiguousarray(aligned_depth))
color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d,
    depth_o3d,
    depth_scale=1.0,          # depth is already in meters
    depth_trunc=real_max_depth + 0.5,  # allow slightly beyond max
    convert_rgb_to_intensity=False,
)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsics,
)

print("Number of points in PCD:", np.asarray(pcd.points).shape[0])

# Optional: flip to match Open3D's usual coordinate convention

o3d.io.write_point_cloud("../data/point_cloud.ply", pcd)
print("Saved PCD to ../data/aliged_pcd.ply")

# build a pcd with original depth image
real_depth_o3d = o3d.geometry.Image(np.ascontiguousarray(real_depth.astype(np.float32)))
rgbd_image_real = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d,
    real_depth_o3d,
    depth_scale=1.0,
    depth_trunc=real_max_depth + 0.5,
    convert_rgb_to_intensity=False,
)
pcd_real = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image_real,
    intrinsics,
)
o3d.io.write_point_cloud("../data/real_point_cloud.ply", pcd_real)
