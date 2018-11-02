cmake_minimum_required(VERSION 2.8.3)
project(object_identification)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  ras_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES object_identification
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)
