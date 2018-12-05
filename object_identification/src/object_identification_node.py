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

        # Load CNN model
        self.model = load_model('/home/ras16/networks/simple_network_14_all.h5')
        self.model.summary()
        self.graph = tf.get_default_graph()

        # List to keep track of the detected objects. Modified via a voting system to only contain the most reasonable object hypothesises
        # Structure: [ID, class_ID, pos_x, pox_y, votes]
        self.object_list = []
        self.object_counter = 0

    
    
    # Function to calculate the similarity between two objects.
    # Return: votes, degree of similarity (2 == equal, 1 == same color)
    def calculate_similarity(self, identification, obj_class):
        if identification == obj_class:
            return 3, 2
        elif self.result_msgs[identification][2] == self.result_msgs[obj_class][2]:
            return 0.5, 1
        else:
            return -1.5, 0

    #Identification result: integer describing the class => self.result_msgs
    #Position of the detected object: [x,y] in the map frame
    #Performs the voting system to keep only track of the most likely objects.
    def manage_objects(self, identification, position, n_objects):
        # 14 == an_object -> if the class could have been not determined, discard the detection
        # Prevents false positives.
        if identification == 14:
            return True, None, -10
        
        known_object = False
        id = None
        votes_obj = -10
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

            # Iterate over the list of close_objects
            for object_dist in close_objects:
                obj = object_dist[0]
                distance = object_dist[1]
                idx = self.object_list.index(obj)
                #Calculate the similarity between the detected object and the current object we are looking it
                votes, same = self.calculate_similarity(identification, obj[1])
                # If similar (at least same color ,votes >= 0)
                if votes >= 0:
                    if not known_object and same == 2:
                        # The current object has not been associated yet and the current close object has the same class as the detected object
                        known_object = True # Set that object has been associated
                        id = obj[0]
                        votes_obj = obj[4]
                        self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                        self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                        self.object_list[idx][4] += votes #Change the votes the object has
                        rospy.loginfo("Case 1")
                        continue
                    if not known_object:
                        # Detecte object has not been associated yet
                        self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                        self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                        self.object_list[idx][4] += votes #Change the votes the object has
                        rospy.loginfo("Case 2")
                    else:
                        # If associated, reduce hypothesis of other similar objects in the close by environment
                        self.object_list[idx][4] -= votes #Change the votes the object has
                        # If not enough votes --> remove hypothesis
                        if self.object_list[idx][4] <= 0:
                            self.object_list.remove(obj)
                        rospy.loginfo("Case 3")
                else:
                    # Close by object does not have a similar class.
                    self.object_list[idx][4] += votes
                    # If not enough votes --> remove hypothesis
                    if self.object_list[idx][4] <= 0:
                        self.object_list.remove(obj)
                    rospy.loginfo("Case 4")

            if not known_object:
                # Detected object could not be associated at all --> Add new hypothesis
                self.object_list.append([self.object_counter, identification, position[0], position[1],3])
                self.object_counter += 1
                id = self.object_list[-1][0]
                rospy.loginfo("Case 5")

        else:
                # no close by objects --> Add new hypothesis
                self.object_list.append([self.object_counter, identification, position[0], position[1],3])
                self.object_counter += 1
                id = self.object_list[-1][0]
                rospy.loginfo("Case 6")




        return known_object, id, votes_obj

    # Function to evalate the class of the detected image via a CNN
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
            # Only associate a class if at least 50% sure, otherwise drop by setting to an_object. Prevents false_positives
            if max_prob < 0.5:
                result = 14
            else:
                result = np.argmax(prediction)

        return result
    
    
    # Callback function if an object has been detected.
    # msg contains the image data and the estimated position for every detected object in the frame
    def callback_image(self,msg):
        #rospy.loginfo("Callback Message received")
        if self.frame_skipper == 0:
            print("\n")
            rospy.loginfo("Frame accepted")
            n_detected_objects = len(msg.positions)
            rospy.loginfo("Known Objects:")
            printed = False
            # Print out the list of known objects. Only take object with 10 or more votes
            for object in self.object_list:
                if object[4] > 10:
                    rospy.loginfo("Object with ID: "+str(object[0])+ " and class: " + self.result_msgs[object[1]][0] + " and votes: " +str(object[4]) + " at position: (" + str(object[2]) + ", "+str(object[3]) + ")")
                    printed = True
            if not printed:
                rospy.loginfo("No known objects")
            self.frame_skipper = 0
            image_counter = 0
            # Iterate over the detected images
            for image_message in msg.data:
                # Read out image data
                self.evidence_msg.image_evidence = image_message
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(image_message, "bgr8")
                except CvBridgeError as e:
                    print(e)

                # Readout estimated position
                self.transform_msg.transform.translation.x = msg.positions[image_counter].point.x
                self.transform_msg.transform.translation.y = msg.positions[image_counter].point.y
                self.transform_msg.transform.translation.z = msg.positions[image_counter].point.z
                print("\n")
                rospy.loginfo("Evaluate clipped image")
                # Determine class of detected object
                result = self.evaluate_image(cv_image)
                image_counter += 1

                if result is not None:
                    rospy.loginfo('Class: '+str(result)+'('+self.result_msgs[result][0]+')')
                    idf = None
                    # Call the object manager to match the newly detected object to known hypothesies or adds new one
                    known_object, idf, votes_pub = self.manage_objects(result,[self.transform_msg.transform.translation.x, self.transform_msg.transform.translation.y],n_detected_objects)
                    # If a new hypothesis was added
                    if not known_object:
                        # Publish sound and evidence message
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


    # Function to fuse object hypothesises together if they are close to each other in the estimated position and if they have at least the same color.
    # The likelihood of missclassifcations in the shape are very likely due to the movement. Therfore, fuse two estimations together if they are close to eacht other and have the same color.
    def fuse_objects(self, object_list):
        fused_list = []
        # Sort list by votes
        sorted_list = list(object_list)
        # Sort the list of known objects by the number of votes. Object with most votes first (--> most certain object)
        sorted_list.sort(key= lambda x: x[4], reverse = True)
        #print(object_list)
        #print(sorted_list)
        counter = 0
        for obj in sorted_list:
            #Calculate closest objects
            counter += 1
            # Iterate over the remaining objects
            for next_obj in sorted_list[counter:-1]:
                distance = math.sqrt(pow((obj[2]-next_obj[2]), 2) + pow((obj[3] - next_obj[3]),2))
                # Calculate if objects are close
                if distance <= 0.4:
                    # Calculate if objects are similar
                    votes, same = self.calculate_similarity(obj[1], next_obj[1])
                    if same:
                        
                        obj[2] = (obj[2] * obj[4] + next_obj[2] * next_obj[4]) / (obj[4] + next_obj[4]) #Average x position based on votes
                        obj[3] = (obj[3] * obj[4] + next_obj[3] * next_obj[4]) / (obj[4] + next_obj[4]) #Average y position based on votes
                        # Remove object with less votes
                        sorted_list.remove(next_obj)
                    else:
                        continue
            fused_list.append(obj)
        return fused_list

    # Callback function if exploration is done
    # Determines the known objects and publishes them with their estimated position to the map node
    def callback_exploration(self, msg):
        if msg.data == True:
            rospy.loginfo("EXPLORING DONE")
            object_list_msg = ObjectList()
            object_list_msg.header.frame_id = "/map"
            certain_objects = []
            if self.object_list:
                for obj in self.object_list:
                    if obj[4] > 10:
                        certain_objects.append(obj)

                certain_objects = self.fuse_objects(certain_objects)
                rospy.loginfo("Known objects that are published to the map:")
                for obj in certain_objects:
                    rospy.loginfo("Object with ID: "+str(obj[0])+ " and class: " + self.result_msgs[obj[1]][0] + " and votes: " +str(obj[4]) + " at position: (" + str(obj[2]) + ", "+str(obj[3]) + ")")
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
            self.object_list = certain_objects



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
