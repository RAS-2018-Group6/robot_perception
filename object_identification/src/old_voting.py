# Check if class was determined at all
# return False, None
#print(self.object_list)
#print(identification, position)
known_object = False
id = None
#rospy.loginfo(self.object_counter)
close_objects = [] # Containing [ID, class_id, distance , votes]
# Was an object seen before?
if not self.object_list:
    #Case: no object has been seen before
    # Add the object depending on the class it was classified as
    # Known -> 10 votes , unknown -> 3 votes
    if identification != 14:
        self.object_list.append([self.object_counter, identification, position[0], position[1],10])
        self.object_counter += 1
    else:
        self.object_list.append([self.object_counter, identification, position[0], position[1], 3])
        self.object_counter += 1
    id = self.object_list[-1][0]


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
        #rospy.loginfo(close_objects)

        #Only evaluate the closest objects depending on the number of objects detected in the frame
        close_objects_counter = 0
        for object_dist in close_objects:
            if close_objects_counter == n_objects:
                break
            close_objects_counter += 1
            obj = object_dist[0]
            distance = object_dist[1]
            idx = self.object_list.index(obj)
            #Check if new object was classified at all
            if identification != 14:
                if obj[1] == identification:
                    #Case: A close object has the same class as the detected (and identified object) --> update the object position and increase the votes --> break the for loop
                    # Therefore the first time an close object with the same class was encountered the function breaks and returns
                    self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                    self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                    self.object_list[idx][4] += 4 #Increase the votes the object has
                    known_object = True
                    id = self.object_list[idx][0]
                    rospy.loginfo("Case 1")
                    continue
                elif obj[1] == 14:
                    # Case: Detected object was identified but the close objects have class: an_object (== unknown class)
                    # Update the position but reduce the votes to change the class at one point
                    # In the case the votes for the object reaches 0, set the class to the one of the identified object and the votes to 3
                    self.object_list[idx][2] = (self.object_list[idx][2] + position[0]) / 2 #Average x position
                    self.object_list[idx][3] = (self.object_list[idx][3] + position[1]) / 2 #Average y position
                    self.object_list[idx][4] -= 1 #Decrease the votes the object has
                    if self.object_list[idx][4] == 0:
                        #self.object_list[idx][1] = identification
                        #self.object_list[idx][4] = 3
                        self.object_list.remove(obj)
                    rospy.loginfo("Case 2")
                    continue

                else:
                    # Case: An object was detected and identified in close distance to an other object that has a specific class
                    # Do not update the position, but reduce the votes, because the estimate of class might be wrong
                    # If votes reaches 0 set the class to: an_object and the votes to 3
                    # Add a new object to the list --> Two objects close to each other.
                    self.object_list[idx][4] -= 1 #Decrease the votes the object has
                    # Set Identification to "an_object"
                    if self.object_list[idx][4] == 0:
                        #self.object_list[idx][1] = 14
                        #self.object_list[idx][4] = 3
                        self.object_list.remove(obj)
                    rospy.loginfo("Case 3")
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
                    rospy.loginfo("Case 4")
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
                    rospy.loginfo("Case 5")
                    break

        if not known_object:
            self.object_list.append([self.object_counter, identification, position[0], position[1],5])
            self.object_counter += 1
            id = self.object_list[-1][0]
            rospy.loginfo("Special Case Add New")

    else:
        #There is no object close to the newly detected objectself.
        # Add object to the list depending on the classification
        # Known -> 10 votes , unknown -> 3 votes
        if identification != 14:
            self.object_list.append([self.object_counter, identification, position[0], position[1],5])
            self.object_counter += 1
            rospy.loginfo("Case 6")
        else:
            self.object_list.append([self.object_counter, identification, position[0], position[1],1])
            self.object_counter += 1
            rospy.loginfo("Case 7")


return known_object, id
