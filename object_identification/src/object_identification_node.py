#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose,PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError




#Node to capture images from the camera and save them to the computer.

class ObjectIdentificationNode:
    def __init__(self):
        
        self.pos_pub = rospy.Publisher('/found_object',PointStamped,queue_size = 1)
        self.clipped_image_pub = rospy.Publisher('/object_detection/detected_image',Image, queue_size = 1) #Publishes the clipped image to the object clasification

        self.bridge = CvBridge()


    def callback_image(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)


        found, positions, clipped_image = self.scanImage(cv_image)
        #print(clipped_image)

        if found:
            try:
                self.clipped_image_pub.publish(self.bridge.cv2_to_imgmsg(clipped_image, "bgr8"))

            except CvBridgeError as e:
                print(e)

            # Calculate the 3D position
            if self.pc_bool:
                if positions:
                    pose_3D = self.calculate3DPositions(positions)
                    #rospy.loginfo(pose_3D)
                    if(pose_3D):
                        pose = PointStamped()
                        time = self.tf_listener.getLatestCommonTime("/map","/camera_link")
                        pose.header.frame_id = 'camera_link'
                        pose.point.x = pose_3D[2] #Robot X = Camera Z
                        pose.point.y = -pose_3D[0] #Robot Y = Camera X
                        pose.point.z = 0 #Robot Z = Camera Y, set to zero because we do not want to have height, assume straight downward projection on the x,y plane

                        pose_in_map = self.tf_listener.transformPoint("/map",pose)
                        pose_in_map.point.z = 0
                        rospy.loginfo(pose_in_map)
                        self.pos_pub.publish(pose_in_map)
                        rospy.loginfo('Published')

            else:
                rospy.loginfo('No PointCloud data available')



    def identify_object(self):
        rospy.init_node('identify_object')
        rospy.Subscriber('/camera/rgb/image_rect_color',Image,self.callback_image,queue_size=1)


        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    identification = ObjectIdentificationNode()
    identification.identify_object()
