from transformers import pipeline
from PIL import Image
import numpy as np
import yaml
import cv2
import open3d as o3d

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

image = cv2.imread("../data/room.jpg")
image_height, image_width = config["image_size"]["height"], config["image_size"]["width"]
image = cv2.resize(image, (image_width, image_height))

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
depth_map = pipe(image)["depth"]

depth_map = np.array(depth_map)

print("Depth map shape:", depth_map.shape)
print("Depth map min value:", np.min(depth_map))
print("Depth map max value:", np.max(depth_map))

# Plot depth map in 3D use pixel coordinates for x and y, and depth for z
h, w = depth_map.shape
xx, yy = np.meshgrid(np.arange(w), np.arange(h))
points = np.stack((xx.flatten(), yy.flatten(), depth_map.flatten()), axis=-1)
# get colors from original image (imager is already resized)
colors = np.asarray(image).reshape(-1, 3) / 255.0
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("../data/depth_point_cloud.ply", pcd)