# notes

## Sensors

### Cameras
SmolVLA needs camera inputs:

`robot.cameras="{ front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \ # <- Use your cameras`
https://huggingface.co/docs/lerobot/main/en/smolvla#evaluate-the-finetuned-model-and-run-it-in-real-time

Isaac sims sensors.camera extension can be used to create / reference cameras within the environment. Prim paths, resolution, frequency, position, and rotation, can all be addressed. The Camera class simply retrieves data from the render of the world and adds it to the Camera's buffer. For this reason, an adequete *render wait time* should be created to ensure the world has finished rendering before being added to the camera buffer. 
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html#standalone-python

Render at 5 simulation steps:

![5_step_render_image](./images/image_5.png)

(Notice the blurred and wavey edges)

Render at 30 simulation steps:

![30_step_render_image](./images/image_30.png)

When interacting with smolVLA, cameras MUST follow the standardised naming convention:

```
OBS_IMAGE_1: Top-down view, 
OBS_IMAGE_2: Wrist-mounted view, 
OBS_IMAGE_3+: Additional viewpoints
```
https://huggingface.co/blog/smolvla#standardizing-camera-views

## Optimisation

### SO100 ARM
When importing the URDF, the colliders option specifies the arms mesh to calculate collisions for the body. Convex hull creates a more general collision representation, less accurate but significantly cheaper to compute due to less nodes in the mesh. Convex Decomposition has much higher accuracy due to more nodes but requires significantly more compute. A good balance for manipulation tasks is to import as Convex and only increase the collision resolution on parts that require fine-grained representation such as the end-effector gripper.

Importing Configuration:
![URDF Configuration](./images/URDF_configuration.png)

Note the "Colliders" section in the bottom of the image. This is where "convex hull" gets selected. 

Final Collisions Mesh:
![collisions](./images/collisions.png)

To manually achieve optimised collisions, right-click each USD's (base, shoulder, upper_arm, lower_arm, wrist, gripper, jaw) "collisons" child USD, go "Add" > "Physics" > "MeshMergeCollision". From there, click on the collisions USD to bring up its properties, scroll down to the newly added "MeshMergeCollision" option, and under "includes" > "add target", add the parent USD (e.g jaw). Then for both the jaw and gripper, under the "collider" property above "MeshMergeCollision" set the "approximation" to "SDF". All other body-parts should be "convex hull".

## Coding
Manipulation within ISAAC can be done via Isaac's Core API or a ROS2 bridge. Core API is plain python and directly integrates within nvidia whilst ROS2 is in ROS and more useful for transfer to realworld robots.

As smolVLA is within the leROBOT ecosystem which is plain python, Core API is the more native solution, with lerobots ROS ports capable of ROS port communications if real-world transfer is needed.

Core API is the way to go.



# sources
NVIDIA + LycheeAI stream covering basics of setting up SO100 arm in isaac for RL: https://www.youtube.com/watch?v=_HMk7I-vSBQ

installing conda + lerobot (in step3, do git clone, and only install training for now! ): https://huggingface.co/docs/lerobot/en/installation

Rigging and file configuration: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/tutorial_generate_robot_config.html

## smolVLA
```
Model description
-Inputs: images (multi-view), proprio/state, optional language -instruction
-Outputs: continuous actions
-Training objective: flow matching
-Action representation: continuous
-Intended use: Base model to fine tune on your specific use case
```

### Rollout of VLA in isaac:
1. Inference Server : Downloads a pretrained model and launches an inference server on a websocket
2. Simulation Environment : Launches simulation environment which publishes observations and subscribes to predicted actions from the inference server.
3. Launch Simulation
4. Visualisation : Visualises the camera data and simulation data, passing it to the DDS bus. Allows streaming and visualising data outside of Isaac Sim.

Getting smolVLA into isaac sim!
https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-for-healthcare/latest/training-healthcare-robots-from-scratch/04-model-flywheel/03-deploy.html 