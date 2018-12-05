# robot_perception
Perception metapackage of the robot working with the Realsense camera.  
Is used for:

## initialize_camera_tools
Used to calculate different camera parameters and to perform testing. For example calculate HSV masks or test out the object detection/identification.

## object_detection
Contains the code for the object detection.
msg: Contain a self defined message type to publish all detected objects with the following information: cut out bonding  and the estimated position through the point cloud.  
src: in the final version used: object_detection_node.py

## object_identification
Package resposible for identifying the detected objects and performing the object memorization.  
msg: Self defined message to publish all the detected objects after the exploration round. Contains: estimated position, identification number and detected class

## obstacle_detection
Package responsible for the obstacle detection

