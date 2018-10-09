cmake_minimum_required(VERSION 2.8.3)
project(initialize_camera_tools)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
