from transformers import pipeline
from PIL import Image
import numpy as np
import yaml
import cv2

image = cv2.imread("../data/room.jpg")
# image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
depth_map = pipe(image)["depth"]

depth_map = np.array(depth_map)

print("Depth map shape:", depth_map.shape)
print("Depth map min value:", np.min(depth_map))
print("Depth map max value:", np.max(depth_map))