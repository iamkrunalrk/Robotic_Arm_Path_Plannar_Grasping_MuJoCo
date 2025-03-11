import PIL.ImageShow
from dm_control import mujoco
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import PIL.Image
import itertools

# MuJoCo
model_path = 'assets/test.xml'
physics = mujoco.Physics.from_xml_path(model_path)



def render_scene(sim):
    # RGB
    rgb_array = sim.render(camera_id=0, width=640, height=480)
    return rgb_array

def get_depth(sim):
    # depth is a float array, in meters.
    depth = sim.render(camera_id=0, height=480, width=640,depth=True)
    # Shift nearest values to the origin.
    depth -= depth.min()
    # Scale by 2 mean distances of near rays.
    depth /= 2*depth[depth <= 1].mean()
    # Scale to [0, 255]
    pixels = 255*np.clip(depth, 0, 1)
    image=PIL.Image.fromarray(pixels.astype(np.uint16))
    return image

def get_ground_truth_pose(sim):
    pos = physics.named.data.xpos["cube"]
    quat=physics.named.data.xquat["cube"] 
    return pos, quat

def save_data(image_id,rgb_image, depth_image,mask_image):
    
    cv2.imwrite('trainingset/rgb/'+str(image_id)+'.png', rgb_image)
    
    depth_image.save('trainingset/depth/'+str(image_id)+'.png')
    mask_image.save('trainingset/mask_visib/'+str(image_id)+'_000000.png')

def get_mat(sim,object_name):
    box_mat = sim.named.data.geom_xmat[str(object_name)].reshape(3, 3)
    return box_mat

def get_bbx(sim,object_name):
    box_size = sim.named.model.geom_size[str(object_name)]
    box_pos = physics.named.data.geom_xpos[str(object_name)]
    box_mat = physics.named.data.geom_xmat[str(object_name)].reshape(3, 3)

    offsets = np.array([[-box_size[0], box_size[0]], 
                       [-box_size[1], box_size[1]], 
                      [-box_size[2], box_size[2]]])
    
    xyz_local = np.stack(list(itertools.product(*offsets))).T 
    xyz_global = box_pos[:, None] + box_mat @ xyz_local 

    # Camera matrices multiply homogenous [x, y, z, 1] vectors.
    corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = xyz_global

    # Get the camera matrix.
    camera = mujoco.Camera(physics,camera_id=0)
    camera_matrix = camera.matrix

    xs, ys, s = camera_matrix @ corners_homogeneous
    # x and y are in the pixel coordinate system.
    x = xs / s
    y = ys / s
    #2d_pose
    twoD_pose=np.array((x, y)).T
    x_min, y_min = twoD_pose.min(axis=0)
    x_max, y_max = twoD_pose.max(axis=0)

    # pixels = camera.render()
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(pixels)
    # ax.plot(x, y, '+', c='w')
    # ax.set_axis_off()
    # plt.show()
    return x_min,y_min,x_max,y_max

def get_mask(sim):

    seg = physics.render(camera_id=0,segmentation=True)

    geom_ids = seg[:, :, 0]
    # Infinity is mapped to -1
    geom_ids = geom_ids.astype(np.float64) + 1

    target_geom_id =13 

    mask = (geom_ids == target_geom_id).astype(np.float64)

    pixels = 255 * mask
    pixels = pixels.astype(np.uint8)

    mask_resized = cv2.resize(pixels, (640, 480), interpolation=cv2.INTER_NEAREST)

    segmentation_image = PIL.Image.fromarray(mask_resized)
    return segmentation_image

def update_mug_position_and_quat(position, quaternion):
    physics.data.qpos[6:6+3]= position
    physics.data.qpos[9:9+4]= quaternion

def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return np.array([q_x, q_y, q_z, q_w])


def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return np.array([q_x, q_y, q_z, q_w])


def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
    


train_gt={}
train_gt_info={}
roll=0.0
rotation_increment=np.radians(5)

for id in range(10,5000):
    #randomly make position
    if np.random.rand() > 0.5:
        random_posx=np.random.uniform(0.05,0.25)
    else:
        random_posx=np.random.uniform(-0.05,-0.2)
    
    if np.random.rand()>0.5:
        random_posy=np.random.uniform(0.25,0.4)
    else:
        random_posy=np.random.uniform(0.55,0.8)

    random_posz=0.05
    position=np.array([random_posx,random_posy,random_posz])
    

    roll = np.random.uniform(0, 2 * np.pi)
    #randomly make quaternion
    quaternion=euler_to_quaternion(roll,0,0)

    update_mug_position_and_quat(position, [roll,0,0])
    physics.forward()
    rgb_image = physics.render(camera_id=0, width=640, height=480)

    x_min,y_min,x_max,y_max=get_bbx(physics,"red_box_geom")
    """
        bbx is N*6 (batch_ids, x1, y1, x2, y2, cls)
    """
    bbx=np.array([x_min,y_min,x_max,y_max])#1 means the cube
    bbx=bbx.tolist()
    pos=position.tolist()
    qua=quaternion.tolist()


    quaternion=euler_to_quaternion(roll,0,0)
    model_rotation=euler_to_rotation_matrix(roll,0,0)

    train_gt[str(id)]={
        0:{"cam_R_m2c":model_rotation.tolist(),
             "cam_t_m2c":position.tolist(),
             "obj_id":1
             }
    }
    train_gt_info[str(id)]={
        0:{"bbox_obj":bbx,
             "bbox_visib":bbx,
        }
    }

    depth_image = get_depth(physics) 

    mask_image=get_mask(physics)

    object_id= str(id).zfill(6)
    save_data(object_id,rgb_image, depth_image,mask_image)

with open('trainingset/train_gt.json','w') as js:
    json.dump(train_gt,js,indent=4)
with open('trainingset/train_gt_info.json','w') as json_file:
    json.dump(train_gt_info,json_file,indent=4)


# a,b=get_RTs()
# print(a)
# print(b)