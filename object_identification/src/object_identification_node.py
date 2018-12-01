#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Bool
from ras_msgs.msg import RAS_Evidence
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose,PointStamped, TransformStamped
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
from object_detection.msg import Objects
from object_identification.msg import ObjectList
import math

#Node to capture images from the camera and save them to the computer.

class ObjectIdentificationNode:
    def __init__(self):

        #possible event [0] and sound [1] messages
        self.result_msgs = {0:['yellow_ball','I see a yellow ball','yellow'], 1:['yellow_cube','I see a yellow cube','yellow'],
                            2:['green_cube','I see a green cube','green'], 3:['green_cylinder','I see a green cylinder','green'],4:['green_hollow_cube','I see a green hollow cube','green'],
                            5:['orange_cross','I see an orange cross','orange'], 6:['patric','I see Patric','orange'],
                            7:['red_cylinder','I see a red cylinder','red'],8:['red_hollow_cube','I see a red hollow cube','red'], 9: ['red_ball','I see a red ball','red'],
                            10:['blue_cube','I see a blue cube','blue'], 11:['blue_triangle','I see a blue triangle','blue'],
                            12:['purple_cross','I see a purple cross','purple'], 13:['purple_star','I see a purple star','purple'],
                            14:['other', 'I see something','none']}

        #self.result_msgs = {0:['blue_triangle','I see a blue_triangle'],1:['green_cylinder','I see a green cylinder'],2:['patric','I see Patric'],
        #                    3:['pruple_cross','I see a purple cross'], 4:['red_ball','I see a red_ball'], 5:['yellow_cube','I see a yellow cube'], 6:['an_object','I see an object']}

        self.sound_msg = String()
        self.sound_msg.data = self.result_msgs[14][1]
        self.sound_pub = rospy.Publisher('/espeak/string',String,queue_size = 1)
        self.evidence_msg = RAS_Evidence()
        self.evidence_msg.group_number = 6
        self.evidence_msg.object_id = self.result_msgs[14][0]
        self.transform_msg = TransformStamped()
        self.evidence_msg.object_location = self.transform_msg
        self.evidence_pub = rospy.Publisher('/evidence',RAS_Evidence,queue_size = 1)
        self.known_objects_pub = rospy.Publisher('/known_objects', ObjectList, queue_size = 1)

        self.bridge = CvBridge()

        self.frame_skipper = 0

        self.model = load_model('/home/ras16/networks/simple_network_14.h5')
        self.model.summary()
        self.graph = tf.get_default_graph()

        # Structure: [ID, class_ID, pos_x, pox_y, votes]
        self.object_list = []
        self.object_counter = 0


    def calculate_similarity(self, identification, obj_class):
        if identification == obj_class:
            return 2, 2
        elif self.result_msgs[identification][2] == self.result_msgs[obj_class][2]:
            return 0.5, 1
        else:
            return -1.5, 0

    #Identification result: integer describing the class => self.result_msgs
    #Position of the detected object: [x,y] in the map frame
    def manage_objects(self, identification, position, n_objects):
        if identification == 14:
            return True, None
        # Check if class was determined at all
        # return False, None
        #print(self.object_list)
        #print(identification, position)
        known_object = False
        id = None
        #rospy.loginfo(self.object_counter)
        close_objects = [] # Containing [ID, class_id, distance , votes]
        # Was an object seen before?

        #Iterate over all objects
        for obj in self.object_list:
            #Determine objects close in radius < 20 cm
            distance = math.sqrt(pow((obj[2]-position[0]), 2) + pow((obj[3] - position[1]),2))
            if distance <= 0.2:
                close_objects.append([obj, distance])

        # Check if there are close by objects
        if close_objects:
            #Sort the objects by distance: closest first
            close_objects.sort(key= lambda x: x[1])
            #rospy.loginfo(close_objects)

            #Only evaluate the closest objects depending on the number of objects detected in the frame
            #close_objects_counter = 0
            for object_dist in close_objects:
                #if close_objects_counter == n_objects:
                #    break
                #close_objects_counter += 1
                obj = object_dist[0]
                distance = object_dist[1]
                idx = self.object_list.index(obj)
                #Check if new object was classified at all
                votes, same = self.calculate_similarity(identification, obj[1])
                if votes >= 0:
                    if not known_object and same == 2:
                        known_object = True
                        id = obj[0]
                        self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                        self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                        self.object_list[idx][4] += votes #Change the votes the object has
                        rospy.loginfo("Case 1")
                        continue
                    if not known_object:
                        self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                        self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                        self.object_list[idx][4] += votes #Change the votes the object has
                        rospy.loginfo("Case 2")
                    else:
                        self.object_list[idx][4] -= votes #Change the votes the object has
                        if self.object_list[idx][4] <= 0:
                            self.object_list.remove(obj)
                        rospy.loginfo("Case 3")
                else:
                    self.object_list[idx][4] += votes
                    if self.object_list[idx][4] <= 0:
                        self.object_list.remove(obj)
                    rospy.loginfo("Case 4")

            if not known_object:
                self.object_list.append([self.object_counter, identification, position[0], position[1],3])
                self.object_counter += 1
                id = self.object_list[-1][0]
                rospy.loginfo("Case 5")

        else:
                self.object_list.append([self.object_counter, identification, position[0], position[1],3])
                self.object_counter += 1
                id = self.object_list[-1][0]
                rospy.loginfo("Case 6")




        return known_object, id

    def evaluate_image(self,cv_image):
        result = None
        resized_image = cv2.resize(cv_image,(32,32))
        array = np.asarray(resized_image)
        array = np.expand_dims(array,axis=0)
        array = array/255.

        with self.graph.as_default():
            prediction = self.model.predict(array)
            #rospy.loginfo(prediction)
            max_prob = np.max(prediction)
            if max_prob < 0.5:
                result = 14
            else:
                result = np.argmax(prediction)

        return result

    def callback_image(self,msg):
        #rospy.loginfo("Callback Message received")
        if self.frame_skipper == 0:
            print("\n")
            rospy.loginfo("Frame accepted")
            n_detected_objects = len(msg.positions)
            rospy.loginfo("Known Objects:")
            printed = False
            for object in self.object_list:
                if object[4] > 10:
                    rospy.loginfo("Object with ID: "+str(object[0])+ " and class: " + self.result_msgs[object[1]][0] + " and votes: " +str(object[4]) + " at position: (" + str(object[2]) + ", "+str(object[3]) + ")")
                    printed = True
            if not printed:
                rospy.loginfo("No known objects")
            self.frame_skipper = 0
            image_counter = 0
            for image_message in msg.data:
                self.evidence_msg.image_evidence = image_message
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(image_message, "bgr8")
                except CvBridgeError as e:
                    print(e)

                self.transform_msg.transform.translation.x = msg.positions[image_counter].point.x
                self.transform_msg.transform.translation.y = msg.positions[image_counter].point.y
                self.transform_msg.transform.translation.z = msg.positions[image_counter].point.z
                print("\n")
                rospy.loginfo("Evaluate clipped image")
                result = self.evaluate_image(cv_image)
                image_counter += 1

                if result is not None:
                    rospy.loginfo('Class: '+str(result)+'('+self.result_msgs[result][0]+')')
                    idf = None
                    known_object, idf = self.manage_objects(result,[self.transform_msg.transform.translation.x, self.transform_msg.transform.translation.y],n_detected_objects)
                    if not known_object:
                        # Set the output string for the sound message and publish it.
                        self.sound_msg.data = self.result_msgs[result][1]
                        self.sound_pub.publish(self.sound_msg)
                        #Set up evidence msg and publish it
                        self.evidence_msg.stamp = rospy.get_rostime()
                        self.evidence_msg.object_id = self.result_msgs[result][0]
                        self.evidence_msg.object_location = self.transform_msg
                        self.evidence_pub.publish(self.evidence_msg)
                        rospy.loginfo("New Object with ID: " + str(idf) + " has been seen.")
                    elif idf is None:
                        rospy.loginfo("Object could not be identified")
                    else:
                        rospy.loginfo("Previously seen Object with ID: " + str(idf) + " has been seen again.")
        else:
            self.frame_skipper += 1


    def fuse_objects(self, object_list):
        fused_list = []
        # Sort list by votes
        sorted_list = object_list.sort(key= lambda x: x[4], reverse = True)
        counter = 0
        for obj in sorted_list:
            #Calculate closest objects
            counter += 1
            for next_obj in sorted_list[counter:-1]:
                distance = math.sqrt(pow((obj[2]-next_obj[2]), 2) + pow((obj[3] - next_obj[3]),2))
                if distance <= 0.2:
                    votes, same = self.calculate_similarity(obj[1], next_obj[1])
                    if same:
                        obj[2] = (obj[2] * obj[4] + next_obj[2] * next_obj[4]) / (obj[4] + next_obj[4]) #Average x position based on votes
                        obj[3] = (obj[3] * obj[4] + next_obj[3] * next_obj[4]) / (obj[4] + next_obj[4]) #Average y position based on votes
                        sorted_list.remove(next_obj)
                    else:
                        continue
            fused_list.append(obj)
        return fused_list


    def callback_exploration(self, msg):
        if msg.data == True:
            object_list_msg = ObjectList()
            object_list_msg.header.frame_id = "/map"
            certain_objects = []
            if self.object_list:
                for obj in self.object_list:
		            if obj[4] > 10:
                        certain_objects.append[obj]

                certain_objects = fuse_objects(certain_objects)
                rospy.loginf("Known objects that are published to the map:")
                for obj in certain_objects:
                    rospy.loginfo("Object with ID: "+str(object[0])+ " and class: " + self.result_msgs[object[1]][0] + " and votes: " +str(object[4]) + " at position: (" + str(object[2]) + ", "+str(object[3]) + ")")
    	            pose = PointStamped()
    	            pose.header.frame_id = "/map"
    	            pose.point.x = obj[2]
    	            pose.point.y = obj[3]
    	            object_list_msg.positions.append(pose)
    	            object_list_msg.id.append(obj[0])
    	            object_list_msg.object_class.append(obj[1])
                    self.known_objects_pub.publish(object_list_msg)
            else:
                rospy.loginfo("No objects detected")



    def identify_object(self):
        rospy.init_node('identify_object')

        rospy.Subscriber('/object_detection/clipped_images',Objects,self.callback_image,queue_size = 1)
        rospy.Subscriber('/finished_exploring',Bool, self.callback_exploration, queue_size = 1)




        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    identification = ObjectIdentificationNode()
    identification.identify_object()
