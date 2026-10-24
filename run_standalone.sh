#!/bin/bash
# Run a standalone Isaac Sim script with the ROS 2 bridge available

unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/opt/ros' | paste -sd:)

export isaac_sim_package_path=$HOME/nvidia/isaacsim/_build/linux-x86_64/release
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Only append the bridge lib path once
bridge_lib=$isaac_sim_package_path/exts/isaacsim.ros2.bridge/jazzy/lib
case ":$LD_LIBRARY_PATH:" in
  *":$bridge_lib:"*) ;;
  *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$bridge_lib ;;
esac

# python.sh sets up the Isaac environment itself and uses its bundled Python 3.11
$isaac_sim_package_path/python.sh /home/louis/nvidia/FinalYearProject/VLA_task.py