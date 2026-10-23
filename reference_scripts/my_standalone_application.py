# run with:
'''
cd ~/nvidia/isaacsim/_build/linux-x86_64/release 
./python.sh /home/louis/nvidia/FinalYearProject/my_application.py
'''

#   ./python.sh      A nvidia isaac python script to run python standalone applications. 
#                    This lives within the cd path so must be ran from there.
#  
#   /home/louis/nvidia/FinalYearProject/my_application.py       is the path to the script (this one).


#default first two lines in any standalone application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless for no rendering to reduce GPU consumption.

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

world = World()
world.scene.add_default_ground_plane()
fancy_cube =  world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="fancy_cube",
        position=np.array([0, 0, 1.0]),
        scale=np.array([0.5015, 0.5015, 0.5015]),
        color=np.array([0, 0, 1.0]),
    ))

fancy_cube2 =  world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube2",
        name="fancy_cube2",
        position=np.array([0, 1.0, 1.0]),
        scale=np.array([0.5015, 0.5015, 0.5015]),
        color=np.array([0, 1.0, 0]),
    ))
# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
world.reset()
for i in range(500):
    position, orientation = fancy_cube.get_world_pose()
    linear_velocity = fancy_cube.get_linear_velocity()
    # will be shown on terminal
    print("Cube position is : " + str(position))
    print("Cube's orientation is : " + str(orientation))
    print("Cube's linear velocity is : " + str(linear_velocity))
    # we have control over stepping physics and rendering in this workflow
    # things run in sync
    world.step(render=True) # execute one physics step and one rendering step

simulation_app.close() # close Isaac Sim