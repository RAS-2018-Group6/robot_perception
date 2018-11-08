cmake_minimum_required(VERSION 2.8.3)
project(object_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  message_generation
)

add_message_files(
  FILES
  Objects.msg
)

 generate_messages(
   DEPENDENCIES
  geometry_msgs std_msgs sensor_msgs
)


catkin_package(
  CATKIN_DEPENDS message_runtime
  )


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
