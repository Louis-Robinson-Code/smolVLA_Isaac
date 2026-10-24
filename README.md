# smolVLA_Isaac

Getting smolVLA in nvidia isaac sim to control a S0-100 arm. Then assessing if ensembles and vector field disagreement (VFD) as a form of Uncertainty Quantitfication (UQ) can be used as input to a UQ controller, statistically evaluating disagreement to form an optimised action during high uncertainty. 

## Overview
- VLA_task.py is the main script that runs the nvidia isaac standalone application, creating unique environments for each run.
- ros2_control.py sends the camera image, joint state (in degrees), and task string to this script over TCP to smolVLA.
- smolVLA.py receives the input, runs inference, and sends back the action output over TCP to ros2_control.py as a tensor of joint angles in degrees.
- ros2_control.py then converts this to radians and sends the action to the robot over ROS2.

## How to run?

Currently: 
1.Ensure lerobot is installed.
2. Open three terminals and enter the following (IN ORDER!). 
   - Terminal 1: ./run_standalone.sh (Sources all Isaac sim and Ros2 bridge dependencies before starting VLA_task.py)
   - Terminal 2: conda activate /home/louis/miniconda3/envs/lerobot && python3 /home/louis/nvidia/FinalYearProject/smolvla.py
   - Termainal 3: python3 ros2_control.py


