#!/bin/bash

# A script to run isaac with the ros2 bridge


# ROS 2 first, then your workspace
source /opt/ros/jazzy/setup.bash
source $HOME/nvidia/IsaacSim-ros_workspaces/jazzy_ws/install/local_setup.bash

export isaac_sim_package_path=$HOME/nvidia/isaacsim/_build/linux-x86_64/release
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Only append the bridge lib path once
bridge_lib=$isaac_sim_package_path/exts/isaacsim.ros2.bridge/jazzy/lib
case ":$LD_LIBRARY_PATH:" in
  *":$bridge_lib:"*) ;;
  *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$bridge_lib ;;
esac

# Run Isaac Sim
$isaac_sim_package_path/isaac-sim.sh