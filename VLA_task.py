

import os

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # set true for headless (reduced GPU consumption)






# enabling the ros2 bridge extension for ros2_control.py
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
enable_extension("omni.graph.window.action")
enable_extension("omni.graph.window.generic")
enable_extension("isaacsim.robot_motion.motion_generation")
simulation_app.update()

#--- omni library -----
import omni.usd # Imports the USD
import omni.ui as ui
from omni.kit.viewport.utility import get_active_viewport
import omni.kit.viewport.utility as vp_utils

#--- isaac sim library -----
from isaacsim.core.api import World
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.prims import RigidPrim, SingleArticulation # Grabs existing prims and enables articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

#---- General python library ------
import time, random, asyncio, datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import socket
import threading
from tcp_helper import send_msg, recv_msg




TCP_CUBE=None


'''
TO DO:

- I dont like smolvla_server and client, the TCP method seems over the top and inefficient.
    Instead, looking into ROS2 bridge to create a pub/sub method for smolVLA control.
    (Done) Ros2 install: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html 
    Camera: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_camera.html 
    Joint States: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_manipulation.html 
    Stand-alone application (code) ros2 : https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_python.html 
    ros2 validation: https://www.linkedin.com/posts/jialei-li-708970250_built-an-visionlanguageaction-pipeline-ugcPost-7411378852200878080-AnMq/ 


SmolVLA implemented into isaac sim but goes to a default joint angle. 
Create episodes to fine tune the model to pick up the cube.
https://huggingface.co/lerobot/smolvla_base
'''


#https://docs.isaacsim.omniverse.nvidia.com/5.1.0/core_api_tutorials/tutorial_core_adding_manipulator.html


'''
Creating the Task environment
- Tasks provide scene modularisation, scene creation, information, retrieval, and metric calculation.
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/core_api_tutorials/tutorial_core_adding_manipulator.html#what-is-a-task
'''

'''Listening for SMolVLA input over TCP, and sending back the action output.'''
class Streamer():

    def __init__(self):
        
        bind_ip = "127.0.0.1" # binding our client to local host
        port=8888
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # re-use address to allow quick rebind of a port
        self.server.bind((bind_ip,port))
        self.server.listen(5) # listening with a max backlog of 5
        print(f"TCP listening on port {bind_ip} : {port}")
        
        self.joint_angles=None
        self.output=None
        self.tcp_cube_pos=None

        

        return

    def IK_solver(self, input):
            '''
            Using the isaac lab built in IK solver to convert the controller inputs to joint angles.
            '''
            ik_solver = LulaKinematicsSolver(
                robot_description_path="/home/louis/finalYear/SO-ARM100/Simulation/SO100/SO-100_LRDF.yaml",  # Path to the robot's URDF file
                urdf_path="/home/louis/finalYear/SO-ARM100/Simulation/SO100/so100_tcp.urdf"
            )
            self.joint_angles,self.success = ik_solver.compute_inverse_kinematics(  #false if not reachable
                frame_name="tcp",
                target_position=input,  # [x, y, z] target position
                target_orientation=None,#[rx, ry, rz],
            )
    
            print(f"IK solver success: {self.success}, joint angles: {self.joint_angles}")
            return

    def handle_client(self, client_socket):
        try:
            while True:
                self.tcp_data = recv_msg(client_socket)           # tcp_data = dict: x,y,z
                print(f"Received TCP data: {self.tcp_data}")
                input=self.tcp_data.get("input")
              
                if input is not None:  #if the input is not none and the joint output is reachable
                    #x, y, z = input[0], input[1], input[2]
                    self.tcp_cube_pos = np.array([input[0], input[1], input[2]])
                    self.IK_solver(input)
                    if self.success is True:
                        self.output = self.joint_angles
                    else:
                        print(f"Target position {input} is not reachable. Sending previous joint angles.")
                send_msg(client_socket, {"joint_angles": self.output})    #always return response so it does hang
        except ConnectionError:
            print("Client disconnected")
        finally:
            client_socket.close()
        return

    
    def main(self):
        while True:
            client,addr = self.server.accept()
            print(f"Accepted connection from: {addr[0]}:{addr[1]}")
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_handler=threading.Thread(target=self.handle_client, args=(client,))
            client_handler.start()
        return



class Task(BaseTask):
    def __init__(self,name):
        super().__init__(name=name, offset=None)
        #self.starting_pos = [0.0,0.0,0.0,0.0,-90.0,0.0]
        self.goal_position = np.array([-0.3, -0.3, 0.0515/ 2.0])
        self.task_achieved = False
        self.tcp_cube = None
        return

#creating our random cube    
    def generate_cube(self, scene):
        #randomising characteristics
        rand_size=random.uniform(0.02,0.04)
        rand_size=0.04
        rand_pos=[random.uniform(-0.2,0.2),random.uniform(-0.2,0.2),0.1]
        
        rand_Rcolour=random.uniform(0,1)
        rand_Gcolour=random.uniform(0,1)
        rand_Bcolour=random.uniform(0,1)
        print(rand_size)
        print(rand_pos)

        #adding the cube to the scene
        return scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="random_cube",
                position=np.array(rand_pos),
                scale=np.array([rand_size, rand_size, rand_size]),
                color=np.array([rand_Rcolour, rand_Gcolour, rand_Bcolour]),
            )
        )
        
    #setting up the tasks scene
    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        self.tcp_cube=scene.add(
        VisualCuboid(
                prim_path="/World/TCP_marker",
                name="TCP_marker",
                position=np.array([-0.1, -0.1, 1]),
                scale=np.array([0.01, 0.01, 0.01]),
                color=np.array([0.3, 0.6, 0.1]),
            )
        )

        self.cube = self.generate_cube(scene)
        self.fancy_cube = scene.add(RigidPrim(prim_paths_expr="/World/cube/Cube", name="Cube"))  #  cube's path (the one with the rigidbody api enabled)
        self.arm = scene.add(SingleArticulation(prim_path="/so_arm100/base", name="so_arm100"))  # must be articulation root of arm

        print(f"{self.arm} DoF: {self.arm.dof_names} Indicies:")
        return

    #Task metadata for monitoring progress
    '''
    Replaced with ROS2 bridge
    '''
    def get_observations(self): 
        cube_pos, cube_orientation = self.cube.get_world_pose()
        joint_pos = self.arm.get_joint_positions()
        joint_vel= self.arm.get_joint_velocities()

        #observations dict
        observations = {
             self.arm.name:{
                  "joint_positions": joint_pos,
                  "joint_velocities": joint_vel,
             },
             self.cube.name:{
                  "position": cube_pos,
                  "orientation": cube_orientation,
             }
        }
        return observations
    #--------------------


    #called before each physics step
    def pre_step(self, control_index, simulation_time):
        #observation = self.get_observations()
       # if not self.task_achieved and np.mean(np.abs(self.goal_position - observation[self.cube.name]["position"])) < 0.02:
        #    self.task_achieved=True
        return

    #called after each reset
    def post_reset(self):
        #action = ArticulationAction(joint_positions=np.array(self.starting_pos)) #actions must be pre-defined
        #self.arm.apply_action(action) #actions can then be commited
        self.task_achieved=False
        return


'''
Establishing our constant environment, adding the tasks
'''
class VLAEnvironment():
    def __init__(self):
        
        self.world = World()
        self.task = Task(name="first_task")
        self.world.add_task(self.task)
        self.world.reset() # clean starting point

        self.arm = self.world.scene.get_object("so_arm100")
        self.arm.initialize() # initialises the controller

        #self.world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return


   
    '''
    Articulation controller inputs: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/articulation_controller.html#input-parameters 
    '''
   


    def establish_viewports(self):
        self.viewport1 = get_active_viewport()
        self.viewport1.camera_path = "/so_arm100/gripper/Top"

        self.window2 = vp_utils.create_viewport_window(name="Scene View")
        self.window2.viewport_api.camera_path = "/so_arm100/scene"

        # Get proper WindowHandle objects by title, for both windows
        self.main_handle = ui.Workspace.get_window("Viewport")
        self.scene_handle = ui.Workspace.get_window("Scene View")


        #---adding our cameras for smolVLA---
        self.camera_wrist = Camera(
            prim_path="/so_arm100/gripper/Top",
            frequency=30,
            resolution=(256,256)    #ensure the same value is allocated to ros2 Isaac Create Render Product Node!
        )
        self.camera_scene = Camera(
            prim_path="/so_arm100/scene",
            frequency=30,
            resolution=(256,256)
        )

        # --- initialising both cameras -----
        self.world.reset() #reset world before initialising cameras
        self.camera_scene.initialize()
        self.camera_wrist.initialize()
        self.camera_wrist.add_motion_vectors_to_frame()
        self.camera_scene.add_motion_vectors_to_frame()
        return

    def enter_starting_pos(self, pos):
                action = ArticulationAction(joint_positions=np.array(pos)) #actions must be pre-defined
                self.arm.apply_action(action) #actions can then be commited
                return
'''
    def create_action(self, pos):
            action = ArticulationAction(joint_positions=np.array(pos), joint_indices=np.array([3])) #actions must be pre-defined
            self.arm.apply_action(action) #actions can then be commited
            return '''


    

    
#---- Debug_func: Captures cameras view ----#
def capture_scene(image):

    if image is None:
        image=simulation.camera_wrist.get_rgba()

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = Path(f"/home/louis/nvidia/FinalYearProject/capture_scene_output/image_{date_time}.png")

    plt.imsave(f"/home/louis/nvidia/FinalYearProject/capture_scene_output/image_{date_time}.png", simulation.camera_wrist.get_rgba()[:, :, :3])
    if image_path.exists():
        print(f"Camera snapshot taken: {image_path}")
    else:
        print(f"Camera snapshot failed")
    return

'''
def debug_output():
    kps, kds = simulation.arm.get_articulation_controller().get_gains()
    print("kp:", kps, "kd:", kds)
    print("q:     ", np.round(simulation.arm.get_joint_positions(), 2))
    print("target:", np.round(target, 2))


    return
'''


# ------- Main program ---------

# Open the saved environment instead of building one from scratch
omni.usd.get_context().open_stage("/home/louis/finalYear/SO-100_TCP.usd")

#from smolvla_client import SmolVLARunner   # you can also move this to the top with your other imports

simulation = VLAEnvironment()
simulation.establish_viewports()
streamer = Streamer()

streamer.tcp_cube=simulation.task.tcp_cube


streamer_thread = threading.Thread(target=streamer.main)
streamer_thread.start()


#runner = SmolVLARunner(instruction="pick up the red cube",lower=simulation.arm.dof_properties["lower"],upper=simulation.arm.dof_properties["upper"])   # once only
policy_step = 2      # physics 60 Hz -> policy ~30 Hz. 

print(simulation.arm.dof_names)
print(simulation.arm.dof_properties["lower"], simulation.arm.dof_properties["upper"])

## Must run on the main thread
while simulation_app.is_running():
        simulation.world.step(render=True)
        streamer.tcp_cube.set_world_pose(position=streamer.tcp_cube_pos)
        if simulation.task.task_achieved:
                print("Task Achieved")
                simulation.world.reset()   # triggers Task.post_reset
                #runner.reset()

simulation_app.close()


'''

i = 0
while simulation_app.is_running():

    #rendering the sim environment
    simulation.world.step(render=True)

    #fetching the cameras buffer (directly taken from the world render)
    wrist_frame = simulation.camera_wrist.get_current_frame()
    scene_frame = simulation.camera_scene.get_current_frame()

    #Ensuring world render is complete and added to camera buffers before feeding to smolVLA
    if i >= 30 and wrist_frame["rgb"] is not None and scene_frame["rgb"] is not None:

        if i == 30:
            capture_scene()
            #print(simulation.camera_wrist.get_current_frame()["motion_vectors"])

    if simulation.task.task_achieved:
        print("Task Achieved")

    i +=1

simulation_app.close()
'''
