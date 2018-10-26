#!/usr/bin/env python
import rospy
from std_msgs import String
from ras_msgs import RAS_Evidence
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose,PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError




#Node to capture images from the camera and save them to the computer.

class ObjectIdentificationNode:
    def __init__(self):

        self.sound_msgs = {'an_object': 'I see an object','yellow_cube': 'I see a yellow cube','yellow_ball': 'I see a yellow ball',
                            'green_cube':'I see a green cube', 'green_cylinder':'I see a green cylinder','green_hollow_cube':'I see a green hollow cube',
                            'orange_cross':'I see an orange cross', 'patric':'I see Patric',
                            'red_cylinder':'I see a red cylinder','red_hollow_cube':'I see a red hollow cube','red_ball':'I see a red ball',
                            'blue_cube':'I see a blue cube', 'blue_triangle':'I see a blue triangle',
                            'pruple_cross':'I see a purple cross', 'purple_star': 'I see a purple star'}

        self.sound_msg = String()
        sound_msg.data = self.sound_msgs['an_object']
        self.sound_pub = rospy.Publisher('/espeak/string',String,queue_size = 1)
        self.evidence_msg = RAS_Evidence()
        self.evidence_pub = rospy.Publisher('/evidence',RAS_Evidence,queue_size = 1)

        self.bridge = CvBridge()

    def evaluate_image(self,image):
        #TODO: Load the trained neural network model and set according to the result the right values for the evidence message and sound message
        return result

    def callback_image(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)


        result = self.evaluate_image(cv_image)
        #print(clipped_image)


    def identify_object(self):
        rospy.init_node('identify_object')
        rospy.Subscriber('object_detection/clipped_image',Image,self.callback_image,queue_size=1)


        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    identification = ObjectIdentificationNode()
    identification.identify_object()
