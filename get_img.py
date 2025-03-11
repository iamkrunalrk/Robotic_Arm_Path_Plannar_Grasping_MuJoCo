import numpy as np
import cv2
from dm_control import mujoco
from dm_control.mujoco import sim

model_path = '/home/robo20/projects/RRT_Grasping_Path_Planning/assets/test.xml'
model = mujoco.MjModel.from_xml_path(model_path)
sim = sim.MjSim(model)

renderer = mujoco.MjRenderContextOffscreen(sim, width=640, height=480)

sim.step() 

rgb_image = renderer.render(mode='rgb')

depth_image = renderer.render(mode='depth')

depth_image_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255
depth_image_normalized = depth_image_normalized.astype(np.uint8)

cv2.imwrite('rgb_image.png', rgb_image)

cv2.imwrite('depth_image.png', depth_image_normalized)

print("RGB")
