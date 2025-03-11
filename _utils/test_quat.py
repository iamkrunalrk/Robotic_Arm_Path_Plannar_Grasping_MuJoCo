from scipy.spatial.transform import Rotation as R
import numpy as np

quat = [0, 0, 0, 1] 

rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
rotation_matrix = rotation.as_matrix()

print(rotation_matrix)