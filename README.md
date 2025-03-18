# High-Dimensional Path Planning with Optimized RRT

This repository provides a simulation environment for a robotic arm to plan paths for grasping objects within a scene using Rapidly-exploring Random Trees (RRT), implemented in MuJoCo.

## Overview

The project enables a robotic arm to navigate a scene, avoiding obstacles, and generates a trajectory that allows it to reach and grasp an object. The RRT algorithm helps in determining feasible paths by exploring random samples in the configuration space, considering both kinematic and dynamic constraints of the robotic arm.

### RRT Algorithm

The underlying mathematical model relies on the **Rapidly-exploring Random Trees (RRT)** algorithm, which explores the robot’s configuration space by iteratively growing a tree from a start state towards a goal state. The tree is constructed by randomly selecting a sample in the space and expanding towards it while avoiding obstacles. In the context of grasping, the RRT considers:

1. **Robot Arm Kinematics**:
   - Denoted by the joint angles θ, the robot's configuration is represented as a vector $θ = [θ₁, θ₂, ..., θₖ]$ for a k-joint robotic arm.
   - The forward kinematics is modeled as:  

     $$
     \mathbf{x} = f(\mathbf{\theta})
     $$

     where $x$ is the position of the end effector in the workspace.

2. **Collision Detection**:
   - The space is discretized into samples that are checked for collisions with obstacles. This is typically handled by the **Signed Distance Function (SDF)**, where points in free space have positive values, and obstacles have negative values.

3. **RRT Path Planning**:
   - The RRT algorithm grows a tree from the initial configuration **θ₀** towards the target configuration $θ₁$, by iteratively:
     1. Selecting a random sample $θₖ$ in the space.
     2. Expanding the tree towards $θₖ$.
     3. Checking for collisions.
   - The trajectory generation for the grasping motion can be summarized by the following:

    $$
    \mathbf{θ}_\text{path} = \{ \mathbf{θ}_0, \mathbf{θ}_1, ..., \mathbf{θ}_n \}
    $$

     where each **θᵢ** is a configuration in the trajectory space.

4. **Grasping**:
   - A successful grasp involves not only a collision-free path but also precise end-effector positioning. The goal state incorporates constraints like orientation and position of the object in the scene.

## Requirements

Before running the simulation, you need to install the following dependencies:

### Step 1: Create a Conda Environment

```bash
conda create -n act_a1 python=3.8.10
conda activate act_a1
```

### Step 2: Install Required Python Packages

```bash
pip install torch torchvision pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib packaging einops h5py ipython
```

### Step 3: Run the Code

To generate the trajectory and visualize the robot’s motion, run:

```bash
python rrt_a1-final.py
```

## Visualization

The generated trajectory can be visualized in the provided video file: `rrt_robot_motion_demo.mp4`.

![Visualization](img/robot_motion_with_transfer.gif)
