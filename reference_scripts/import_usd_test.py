

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # set true for headless (reduced GPU consumption)


import omni.usd # Imports the USD
import omni.ui as ui
from isaacsim.core.api import World
import numpy as np
from isaacsim.core.prims import RigidPrim, SingleArticulation # Grabs existing prims and enables articulation
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility import get_viewport_from_window_name
import omni.kit.viewport.utility as vp_utils
import time, random

#from key_functions import establish_viewports,create_action,enter_starting_pos #pulling necessary function


# Open your saved environment instead of building one from scratch
omni.usd.get_context().open_stage("/home/louis/finalYear/SO-ARM100_sim_test.usd")

'''
move into a class based coding
'''

class VLAEnvironment:
    def __init__(self):

        self.world = World()
        self.world.reset() # clean starting point
        self.fancy_cube = RigidPrim(prim_paths_expr="/World/cube/Cube", name="Cube")  #  cube's path (the one with the rigidbody api enabled)
        self.arm = SingleArticulation(prim_path="/so_arm100/base", name="so_arm100")  # must be articulation root of arm

        self.arm.initialize() # initialises the controller
        print(f"{self.arm} DoF: {self.arm.dof_names} Indicies:")

        return


    def establish_viewports(self):
        self.viewport1 = get_active_viewport()
        self.viewport1.camera_path = "/so_arm100/gripper/Top"

        self.window2 = vp_utils.create_viewport_window(name="Scene View")
        self.window2.viewport_api.camera_path = "/so_arm100/scene"

        # Get proper WindowHandle objects by title, for both windows
        self.main_handle = ui.Workspace.get_window("Viewport")
        self.scene_handle = ui.Workspace.get_window("Scene View")

        if self.main_handle and self.scene_handle:
            self.scene_handle.dock_in(self.main_handle, ui.DockPosition.RIGHT)
        else:
            print("Could not find window handles:", self.main_handle, self.scene_handle)
        return

    def create_action(self, pos):
            action = ArticulationAction(joint_positions=np.array(pos), joint_indices=np.array([3])) #actions must be pre-defined
            self.arm.apply_action(action) #actions can then be commited
            return


    def enter_starting_pos(self, pos):
            action = ArticulationAction(joint_positions=np.array(pos)) #actions must be pre-defined
            self.arm.apply_action(action) #actions can then be commited
            return

    #creating our random cube
    def generate_cube(self):
        #randomising characteristics
        rand_size=random.uniform(0.02,0.04)
        rand_size=0.04
        rand_Xpos=random.uniform(-0.2,0.2)
        rand_Ypos=random.uniform(-0.2,0.2)
        rand_Zpos=0.1
        rand_Rcolour=random.uniform(0,1)
        rand_Gcolour=random.uniform(0,1)
        rand_Bcolour=random.uniform(0,1)
        print(rand_size)
        print(rand_Xpos, rand_Ypos, rand_Zpos)
        

        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="random_cube",
                position=np.array([rand_Xpos, rand_Ypos, rand_Zpos]),
                scale=np.array([rand_size, rand_size, rand_size]),
                color=np.array([rand_Rcolour, rand_Gcolour, rand_Bcolour]),
            )
        ) 
        
        self.cube = self.world.scene.get_object("random_cube")
        return


# Main program, creating an instance of the class to reference

simulation = VLAEnvironment()
starting_pos = [0.0,0.0,0.0,0.0,-90.0,0.0]
#action_pos = [0.0,80.0,50.0,30.0,-90.0,0.0]


simulation.enter_starting_pos(starting_pos)
simulation.establish_viewports()
simulation.generate_cube()
#simulation.create_action(action_pos)
for i in range(500):
    
    positions, orientations = simulation.cube.get_world_pose()
    linear_velocities = simulation.fancy_cube.get_linear_velocities()
    #print("Cube position is : " + str(positions))
    #print("Cube's orientation is : " + str(orientations))
    #print("Cube's linear velocity is : " + str(linear_velocities))
    simulation.world.step(render=True)

    #simulation_app.close()