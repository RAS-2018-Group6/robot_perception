#!/usr/bin/env python
import rospy
from std_msgs.msg import String
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
import math

#Node to capture images from the camera and save them to the computer.

class ObjectIdentificationNode:
    def __init__(self):

        #possible event [0] and sound [1] messages
        self.result_msgs = {0:['yellow_ball','I see a yellow ball'], 1:['yellow_cube','I see a yellow cube'],
                            2:['green_cube','I see a green cube'], 3:['green_cylinder','I see a green cylinder'],4:['green_hollow_cube','I see a green hollow cube'],
                            5:['orange_cross','I see an orange cross'], 6:['patric','I see Patric'],
                            7:['red_cylinder','I see a red cylinder'],8:['red_hollow_cube','I see a red hollow cube'], 9: ['red_ball','I see a red ball'],
                            10:['blue_cube','I see a blue cube'], 11:['blue_triangle','I see a blue triangle'],
                            12:['purple_cross','I see a purple cross'], 13:['purple_star','I see a purple star'],
                            14:['an_object', 'I see an object']}

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

        self.bridge = CvBridge()

        self.frame_skipper = 0

        self.model = load_model('/home/ras16/networks/simple_network.h5')
        self.model.summary()
        self.graph = tf.get_default_graph()

        # Structure: [ID, class_ID, pos_x,pox_y, votes]
        self.object_list = []

    #Identification result: integer describing the class => self.result_msgs
    #Position of the detected object: [x,y] in the map frame
    def manage_objects(self, identification, position):
        # Check if class was determined at all
        return False, 0
        known_object = False
        id = None
        object_counter = len(self.object_list)
        close_objects = [] # Containing [ID, class_id, distance , votes]
        # Was an object seen before?
        if not self.object_list:
            #Case: no object has been seen before
            # Add the object depending on the class it was classified as
            # Known -> 10 votes , unknown -> 3 votes
            if identification != 14:
                self.object_list.append([object_counter, identification, position[0], position[1]],10)
            else:
                self.object_list.append([object_counter, identification, position[0], position[1]],3)

        else:
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

                    for object_dist in close_objects:
                        obj = object_dist[0]
                        distance = object_dist[1]
                        idx = self.object_list[obj]
                        #Check if new object was classified at all
                        if identification != 14:
                            if obj[1] == identification:
                                #Case: A close object has the same class as the detected (and identified object) --> update the object position and increase the votes --> break the for loop
                                # Therefore the first time an close object with the same class was encountered the function breaks and returns
                                self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                                self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                                self.object_list[idx][4] += 1 #Increase the votes the object has
                                known_object = True
                                id = self.object_list[idx][0]
                                break
                            elif obj[1] == 14:
                                # Case: Detected object was identified but the close objects have class: an_object (== unknown class)
                                # Update the position but reduce the votes to change the class at one point
                                # In the case the votes for the object reaches 0, set the class to the one of the identified object and the votes to 3
                                self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                                self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                                self.object_list[idx][4] -= 1 #Decrease the votes the object has
                                if self.object_list[idx][4] == 0:
                                    self.object_list[idx][1] = identification
                                    self.object_list[idx][4] = 3
                                continue

                            else:
                                # Case: An object was detected and identified in close distance to an other object that has a specific class
                                # Do not update the position, but reduce the votes, because the estimate of class might be wrong
                                # If votes reaches 0 set the class to: an_object and the votes to 3
                                self.object_list[idx][4] -= 1 #Decrease the votes the object has
                                # Set Identification to "an_object"
                                if self.object_list[idx][4] == 0:
                                    self.object_list[idx][1] = 14
                                    self.object_list[idx][4] = 3
                                continue
                        else:
                            #Check if closest object is classified
                            if obj[1] == 14:
                                #Case: detected object was not identified and the object is not classified as well
                                # Update the position and increase the votes for the estimated class (unknown class)
                                # When the first object was detected this way break and return from the function
                                self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                                self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                                self.object_list[idx][4] += 1 #Increase the votes the object has
                                known_object = True
                                id = self.object_list[idx][0]
                                break
                            else:
                                #Case: The detected object was not identified, but the object in the list was identified
                                #Update the position, because it still may be the same class, but decrease the number of votes, because the obejct might be missclassified
                                # If the votes reach 0 set the class to an_obejct and the number of votes to 3
                                self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                                self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                                self.object_list[idx][4] -= 1 #Decrease the votes the object has
                                if self.object_list[idx][4] == 0:
                                    self.object_list[idx][1] = 14
                                    self.object_list[idx][4] = 3
                                known_object = True
                                id = self.object_list[idx][0]
                                break

                else:
                    #There is no object close to the newly detected objectself.
                    # Add object to the list depending on the classification
                    # Known -> 10 votes , unknown -> 3 votes
                    if identification != 14:
                        self.object_list.append([object_counter, identification, position[0], position[1]],5)
                    else:
                        self.object_list.append([object_counter, identification, position[0], position[1]],1)


        return known_object, id

    def evaluate_image(self,cv_image):
        #TODO: Load the trained neural network model and set according to the result the right values for the evidence message and sound message
        result = None
        resized_image = cv2.resize(cv_image,(32,32))
        array = np.asarray(resized_image)
        array = np.expand_dims(array,axis=0)
        array = array/255.

        with self.graph.as_default():
            prediction = self.model.predict(array)
            rospy.loginfo(prediction)
            max_prob = np.max(prediction)
            if max_prob < 0.5:
                result = 14
            else:
                result = np.argmax(prediction)

        return result

    def callback_image(self,msg):
        #rospy.loginfo("Callback Message received")
        if self.frame_skipper == 10:
            rospy.loginfo("Frame accepted")
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

                rospy.loginfo("Evaluate clipped image")
                result = self.evaluate_image(cv_image)

                if result is not None:
                    rospy.loginfo('Class: '+str(result)+'('+self.result_msgs[result][0]+')')
                    idf = None
                    known_object, idf = self.manage_objects(result,[self.transform_msg.transform.translation.x, self.transform_msg.transform.translation.y])
                    if not known_object:
                        # Set the output string for the sound message and publish it.
                        self.sound_msg.data = self.result_msgs[result][1]
                        self.sound_pub.publish(self.sound_msg)
                        #Set up evidence msg and publish it
                        self.evidence_msg.stamp = rospy.get_rostime()
                        self.evidence_msg.object_id = self.result_msgs[result][0]
                        self.evidence_msg.object_location = self.transform_msg
                        self.evidence_pub.publish(self.evidence_msg)
                    else:
                        rospy.loginfo("Previously seen Object with ID: " + str(idf) + "has been seen again.")
        else:
            self.frame_skipper += 1


    def identify_object(self):
        rospy.init_node('identify_object')

        rospy.Subscriber('/object_detection/clipped_images',Objects,self.callback_image,queue_size = 1)



        rate = rospy.Rate(10) #HZ
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    identification = ObjectIdentificationNode()
    identification.identify_object()
