cmake_minimum_required(VERSION 2.8.3)
project(object_identification)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  ras_msgs
  object_detection
  message_generation
  geometry_msgs
)

add_message_files(
  FILES
  ObjectList.msg
)

 generate_messages(
   DEPENDENCIES
  geometry_msgs std_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES object_identification
CATKIN_DEPENDS rospy sensor_msgs object_detection message_runtime
#  DEPENDS system_lib
)
