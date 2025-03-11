import cv2

image = cv2.imread('/home/robo20/projects/RRT_Grasping_Path_Planning/assets/models/011_banana/texture_map.png')

img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite('/home/robo20/projects/RRT_Grasping_Path_Planning/assets/models/011_banana/texture_map_bgr.png', img_bgr)