cmake_minimum_required(VERSION 3.10.2)
project(laser_udp_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf2_ros
  geometry_msgs
  nn_laser_spot_tracking
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/laser_udp_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
