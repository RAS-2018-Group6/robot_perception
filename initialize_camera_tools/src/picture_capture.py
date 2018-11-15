#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np


#Node to capture images from the camera and save them to the computer.

class picture_caputre:
    def __init__(self):
        self.take_image = True
        self.idx = 0

    # Transforms an 1D-BGR array (3*height*width) to an 3D-RGB array (height,width,3)
    def transform_to_image(self,np_array,height,width):
        img = np.zeros((height,width,3))
        img[:,:,0] = np.reshape(np_array[2::3],(height,width))
        img[:,:,1] = np.reshape(np_array[1::3],(height,width))
        img[:,:,2] = np.reshape(np_array[0::3],(height,width))
        return img



    def callback_image(self,msg):
        # Only take one image

        if self.take_image:
            # Convert received image data to CV2 format
            np_image_arr = np.fromstring(msg.data, np.uint8)
            rospy.loginfo(np_image_arr.shape)
            rospy.loginfo(msg.height)
            rospy.loginfo(msg.width)
            np_image = self.transform_to_image(np_image_arr,msg.height,msg.width)
            rospy.loginfo(np_image.shape)
            save_link = '/home/ras16/maze_images/test_'+str(self.idx)+'.png'
            self.idx += 1
            cv2.imwrite(save_link,np_image)
            self.take_image = True

    def save_images(self):
        rospy.init_node('save_images')
        rospy.Subscriber('/camera/rgb/image_rect_color',Image,self.callback_image,queue_size=1)

        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    capture = picture_caputre()
    capture.save_images()
