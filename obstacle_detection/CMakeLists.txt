cmake_minimum_required(VERSION 2.8.3)
project(obstacle_detection)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_ros
  roscpp
  sensor_msgs
  geometry_msgs
  std_msgs
  tf2
  message_generation
)

add_message_files(
  FILES
  Obstacle.msg
)

 generate_messages(
   DEPENDENCIES
  geometry_msgs std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES obstacle_detection
CATKIN_DEPENDS cv_bridge pcl_ros roscpp sensor_msgs std_msgs tf2 message_runtime
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(obstacle_detection_node src/obstacle_detection.cpp)

add_dependencies(obstacle_detection_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(obstacle_detection_node ${catkin_LIBRARIES})
