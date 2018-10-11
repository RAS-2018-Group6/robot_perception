#!/usr/bin/env python
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import cv2
import numpy as np
import sklearn


#Node to capture images from the camera and save them to the computer.

class object_detection:
    def __init__(self):
        self.pos_msg = Pose()
        self.pos_pub = rospy.Publisher('/object_position',Pose,queue_size = 1)
        self.clipped_image_pub = rospy.Publisher('/detected_image',numpy_msg(uint8),queue_size = 1) #Publishes the clipped image to the object clasification


    # Transforms an 1D-BGR array (3*height*width) to an 3D-RGB array (height,width,3)
    def transform_to_image(self,np_array,height,width):
        img = np.zeros((height,width,3))
        img[:,:,0] = np.reshape(np_array[2::3],(height,width))
        img[:,:,1] = np.reshape(np_array[1::3],(height,width))
        img[:,:,2] = np.reshape(np_array[0::3],(height,width))
        return img


    def scanImage(self,np_array):
        ### TODO: Decide on algorithm to detect objects: HOG vs. path_save
        ### TODO: Find way to extract position(Distance,Bearing) from Depth measurement
        ### TODO: Use tf to display the object in roboter frame
        image = np_array
        clipped_image = []
        position = [] #X,Y,Z =0
        found = False
        return found,position,clipped_image


    def callback_image(self,msg):
        # Convert received image data to CV2 format
        np_image_arr = np.fromstring(msg.data, np.uint8)
        #rospy.loginfo(np_image_arr.shape)
        #rospy.loginfo(msg.height)
        #rospy.loginfo(msg.width)
        np_image = self.transform_to_image(np_image_arr,msg.height,msg.width)

        found, detected_object, clipped_image = self.scanImage(np_image)

        if found:
            self.pos_msg.position.x = detected_object[0]
            self.pos_msg.position.y = detected_object[1]
            self.pos_msg.position.z = 0
            self.pos_pub.publish(self.pos_msg)

            self.clipped_image.publish(clipped_image




    def detect_objects(self):
        rospy.init_node('detect_objects')
        rospy.Subscriber('/camera/rgb/image_rect_color',Image,self.callback_image,queue_size=1)

        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    detection = object_detection()
    capture.detect_objects()
