#!/usr/bin/env python
import rospy
from rospy.numpy_msg import numpy_msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose,PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import struct
import tf



#Node to capture images from the camera and save them to the computer.

class object_detection_save:
    def __init__(self):


        self.lower = {'red':(0, 169, 84), 'green':(37, 150, 60), 'blue':(80, 114, 60), 'yellow':(17, 150, 115), 'orange':(5, 190, 130), 'purple':(100,32,81)}
        self.upper = {'red':(10,255,175), 'green':(70,255,190), 'blue':(110,255,170), 'yellow':(25,255,230), 'orange':(18,255,215), 'purple':(180,150,185)}

        self.colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255), 'purple':(210,255,128)}
        self.counter = 0

        self.bridge = CvBridge()

    # Transforms an 1D-BGR array (3*height*width) to an 3D-RGB array (height,width,3)
    def transform_to_image(self,np_array,height,width):
        img = np.zeros((height,width,3))
        img[:,:,0] = np.reshape(np_array[2::3],(height,width))
        img[:,:,1] = np.reshape(np_array[1::3],(height,width))
        img[:,:,2] = np.reshape(np_array[0::3],(height,width))
        return img


    def scanImage(self,np_array):
        ### TODO: Find way to extract position(Distance,Bearing) from Depth measurement
        ### TODO: Use tf to display the object in roboter frame
        image = np_array
        found = False
        #rospy.loginfo(image.shape)
        #rospy.loginfo(image)
        blurred_image = cv2.GaussianBlur(image,(11,11),0)
        blurred_image = blurred_image.astype(np.uint8)
        #rospy.loginfo(blurred_image.shape)
        #rospy.loginfo(blurred_image)
        hsv = cv2.cvtColor(blurred_image,cv2.COLOR_BGR2HSV)
        positions = []
        for key,value in self.upper.items():
            if key == 'purple':
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

                    x = x - 20
                    y = y - 20
                    w = w + 40
                    h = h + 40

                    if x < 0 or y < 0:
                        x = 0
                        y = 0

                    if area >= 4000 and area <= 35000:
                        #cv2.rectangle(image,(x,y),(x+w,y+h),self.colors[key],2)
                        found = True
                        clipped_image = image[y:y+h,x:x+w]
                        path = '/home/ras16/dataset/purple_cross/'
                        save_path = path + 'purple_cross_' + str(self.counter) + '.png'
                        self.counter += 1
                        cv2.imwrite(save_path,clipped_image)
                        cv2.rectangle(image,(x,y),(x+w,y+h),self.colors[key],2)
                        #rospy.loginfo(image)

        cv2.imshow('display',image)
        cv2.waitKey(1)

        bounding_image = image

        return found,positions,clipped_image,bounding_image


    def callback_image(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)


        found, positions, clipped_image, bounding_image = self.scanImage(cv_image)
        #print(clipped_image)


            # Calculate the 3D position




    def detect_objects(self):
        rospy.init_node('detect_objects')
        rospy.Subscriber('/camera/rgb/image_rect_color',Image,self.callback_image,queue_size=1)

        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    detection = object_detection_save()
    detection.detect_objects()
