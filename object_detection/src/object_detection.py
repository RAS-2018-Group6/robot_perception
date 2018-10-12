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
        self.lower = {'red':(0, 169, 84), 'green':(37, 150, 60), 'blue':(80, 114, 60), 'yellow':(17, 150, 115), 'orange':(5, 190, 130), 'purple':(100,32,81)}
        self.upper = {'red':(10,255,175), 'green':(70,255,190), 'blue':(110,255,170), 'yellow':(25,255,230), 'orange':(18,255,215), 'purple':(180,150,185)}

        self.colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255), 'purple':(210,255,128)}


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
        blurred_image = cv2.GaussianBlur(image,(11,11),0)
        hsv = cv2.cvtColor(blurred_image,cv2.COLOR_BGR2HSV)
        for key,value in self.upper.items():
            # Mask the image and use open/close to have smoother conturs
            kernel = np.ones((9,9),np.uint8)
            mask = cv2.inRange(hsv,self.lower[key],self.upper[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours and initialize center (x,y) of the object
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None

            if len(cnts)>0:
                # Find biggest contur:
                c = max(cnts,key = cv2.contourArea)
                x,y,w,h = cv2.boundingRect(c)
                # Only draw bounding boxes that have a goodsize
                area = w * h
                if area >= 4000 and area <= 35000:
                    #print('Found: ' + key )
                    cv2.rectangle(image,(x,y),(x+w,y+h),colors[key],2)
        cv2.imshow('display'image)
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
