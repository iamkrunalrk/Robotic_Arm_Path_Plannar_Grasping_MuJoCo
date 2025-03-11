from scipy.spatial.transform import Rotation as R
import numpy as np

euler_angles = [np.pi/2, 0, 0]  # yaw, pitch, roll

quat = R.from_euler('xyz', euler_angles).as_quat()  # [x, y, z, w]

quat_xml_format = [quat[3], quat[0], quat[1], quat[2]]
print("XML:", quat_xml_format)
