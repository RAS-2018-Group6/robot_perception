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

        #possible event [0] and sound [1] messages
        self.result_msgs = {0:['an_object','I see an object'], 1:['yellow_cube','I see a yellow cube'], 2:['yellow_ball','I see a yellow ball'],
                            3:['green_cube','I see a green cube'], 4:['green_cylinder','I see a green cylinder'],5:['green_hollow_cube','I see a green hollow cube'],
                            6:['orange_cross','I see an orange cross'], 7:['patric','I see Patric'],
                            8:['red_cylinder','I see a red cylinder'],9:['red_hollow_cube','I see a red hollow cube'], 10: ['red_ball','I see a red ball'],
                            11:['blue_cube','I see a blue cube'], 12:['blue_triangle','I see a blue triangle'],
                            13:['pruple_cross','I see a purple cross'], 14:['purple_star','I see a purple star']}

        self.sound_msg = String()
        self.sound_msg.data = self.result_msgs[0][1]
        self.sound_pub = rospy.Publisher('/espeak/string',String,queue_size = 1)
        self.evidence_msg = RAS_Evidence()
        self.evidence_pub = rospy.Publisher('/evidence',RAS_Evidence,queue_size = 1)

        self.bridge = CvBridge()

    def evaluate_image(self,image):
        #TODO: Load the trained neural network model and set according to the result the right values for the evidence message and sound message
        result = None
        return result

    def callback_image(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)


        result = self.evaluate_image(cv_image)

        if result is not None:
            self.sound_msg.data = self.result_msgs[result][1]
            self.sound_pub.publish(sel.sound_msg)
            #Set up evidence msg 




    def identify_object(self):
        rospy.init_node('identify_object')
        rospy.Subscriber('object_detection/clipped_image',Image,self.callback_image,queue_size=1)


        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    identification = ObjectIdentificationNode()
    identification.identify_object()
