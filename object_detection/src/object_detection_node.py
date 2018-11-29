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
from object_detection.msg import Objects
from std_msgs import Bool



#Node to capture images from the camera and save them to the computer.

class object_detection_node:
    def __init__(self):
        self.pos_msg = Pose()
        self.pos_pub = rospy.Publisher('/found_object',PointStamped,queue_size = 1)
        self.bounding_box_image_pub = rospy.Publisher('/object_detection/detected_image',Image, queue_size = 1) #Publish the image with the bounding box drawn
        #self.clipped_image_pub = rospy.Publisher('/object_detection/clipped_image',Image,queue_size = 1) #Publishes the clipped image to the object clasification
        self.clipped_images_pub = rospy.Publisher('/object_detection/clipped_images',Objects,queue_size = 1) #Publishes the clipped image to the object clasification
        self.object_detected_pub = rospy.Publisher('/wall_detected',Bool,queue_size=1)

        # Previous HSV masks table
        #self.lower = {'red':(0, 169, 84), 'green':(37, 150, 60), 'blue':(80, 114, 60), 'yellow':(17, 150, 115), 'orange':(5, 190, 130), 'purple':(100,32,81)}
        #self.upper = {'red':(10,255,175), 'green':(70,255,190), 'blue':(110,255,170), 'yellow':(25,255,230), 'orange':(18,255,215), 'purple':(180,150,185)}

        # Maze HSV values
        self.lower = {'red':(0, 150, 70), 'green':(40, 130, 50), 'blue':(95, 130, 60), 'yellow':(16, 160, 100), 'orange':(7, 200, 130), 'purple':(120,60,70)}
        self.upper = {'red':(6,255,214), 'green':(80,255,190), 'blue':(101,255,180), 'yellow':(25,250,255), 'orange':(13,255,240), 'purple':(160,150,187)}

        self.colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255), 'purple':(212,255,255)}

        self.tf_listener = tf.TransformListener()
        self.pc_bool = False
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
        object_images = []
        #rospy.loginfo(image.shape)
        #rospy.loginfo(image)
        blurred_image = cv2.GaussianBlur(image,(11,11),0)
        blurred_image = blurred_image.astype(np.uint8)
        #rospy.loginfo(blurred_image.shape)
        #rospy.loginfo(blurred_image)
        hsv = cv2.cvtColor(blurred_image,cv2.COLOR_BGR2HSV)
        positions = []
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
                x = x - 20
                y = y - 20
                w = w + 40
                h = h + 40

                if x < 0:
                    x = 0

                if y < 0:
                    y = 0

                if y + h > 479:
                    h = 479-y

                if x + w > 639:
                    w = 639-x

                if area >= 4000 and area <= 35000:
                    positions.append([x,y,w,h,key])
                    clipped_image = np.copy(image[y:y+h,x:x+w])
                    object_images.append(clipped_image)

                    found = True
                    #rospy.loginfo(image)

        for position in positions:
            x,y,w,h,key = position
            cv2.rectangle(image,(x,y),(x+w,y+h),self.colors[key],2)
        cv2.imshow('display',image)
        cv2.waitKey(1)


        bounding_image = image

        return found,positions,object_images,bounding_image


    def callback_image(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)


        found, positions, object_images, bounding_image = self.scanImage(cv_image)


        if found:
            detected = Bool()
            detected.data = True
            self.object_detected_pub.publish(detected)
            objects_msg = Objects()
            for image in object_images:
                try:
                    objects_msg.data.append(self.bridge.cv2_to_imgmsg(image,"bgr8"))
                except CvBridgeError as e:
                    print(e)
                initial_pose = PointStamped()


            try:
                #self.clipped_images_pub.publish(objects_msg)
                self.bounding_box_image_pub.publish(self.bridge.cv2_to_imgmsg(bounding_image, "bgr8"))

            except CvBridgeError as e:
                print(e)

            # Calculate the 3D position

            if self.pc_bool:
                for position in positions:

                    initial_pose.point.x = float('NaN')
                    initial_pose.point.y = float('NaN')
                    initial_pose.point.z = float('NaN')
                    #print(position)
                    pose_3D = self.calculate3DPositions(position)
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
                        objects_msg.positions.append(pose_in_map)
                        #rospy.loginfo(pose_in_map)
                        #self.pos_pub.publish(pose_in_map)

                    else:
                        pose = PointStamped()
                        pose.header.frame_id = 'map'
                        pose.point.x = float('NaN')
                        pose.point.y = float('NaN')
                        pose.point.z = float('NaN')
                        objects_msg.positions.append(pose)

                self.clipped_images_pub.publish(objects_msg)
                #rospy.loginfo('Published')
                self.pc_bool = False

            else:
                rospy.loginfo('No PointCloud data available')
                self.clipped_images_pub.publish(objects_msg)
                rospy.loginfo('Published with no Position')

    def calculate3DPositions(self,position):
        pose_3D = []
        if self.pc_bool:
            pc_width = self.point_cloud.width
            pc_height = self.point_cloud.height

            # Create acces points
            uv = [[u,v] for u in range(position[0],position[0]+position[2],5) for v in range(position[1],position[1]+position[3],5)]
            #rospy.loginfo(uv)
            pose_3D = [0,0,0]
            counter_x = 0
            counter_y = 0
            counter_z = 0
            for point in pc2.read_points(self.point_cloud, skip_nans = True, uvs = uv):
                if point[0] != 0:
                    pose_3D[0] += point[0]
                    counter_x += 1
                if point[1] != 0:
                    pose_3D[1] += point[1]
                    counter_y += 1
                if point[2] != 0:
                    pose_3D[2] += point[2]
                    counter_z += 1
            if counter_x != 0 and counter_y != 0 and counter_z != 0:
                pose_3D[0] = pose_3D[0]/counter_x
                pose_3D[1] = pose_3D[1]/counter_y
                pose_3D[2] = pose_3D[2]/counter_z

        return pose_3D

    def callback_pointcloud(self,msg):
        #rospy.loginfo(msg._type)
        self.point_cloud = msg

        self.pc_bool = True

    def detect_objects(self):
        rospy.init_node('detect_objects')
        rospy.Subscriber('/camera/rgb/image_rect_color',Image,self.callback_image,queue_size=1)
        rospy.Subscriber('camera/depth_registered/points',PointCloud2,self.callback_pointcloud,queue_size=1)

        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    detection = object_detection_node()
    detection.detect_objects()
