#!/usr/bin/env python3

# Copyright (c) 2016 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Newtworking Application

Make Cozmo communicate with other Cozmo robots

'''

import cozmo
import socket
import errno
from socket import error as socket_error

#need to get movement info
from cozmo.util import degrees, distance_mm, speed_mmps


def cozmo_program(robot: cozmo.robot.Robot):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket_error as msg:
        robot.say_text("socket failed" + msg).wait_for_completed()
    ip = "10.0.1.10"
    port = 5000
    
    try:
        s.connect((ip, port))
    except socket_error as msg:
        robot.say_text("socket failed to bind").wait_for_completed()
    cont = True
    
    robot.say_text("ready").wait_for_completed()    
    
    #SET COZMO's NAME
    myName = 'CozmoName'
    
    while cont:
        bytedata = s.recv(4048)
        #NOTE: casting bytedata as a string just returns "b'string'" which was not the goal!  
        #data = str(bytedata)
        #we need to decode the byte data to get the contents of the message
        data = bytedata.decode('utf-8')
        if not data:
            cont = False
            s.close()
            quit()
        else:
            #---------------------------------------------------------
            #This is where you need to adjust the program
            #---------------------------------------------------------
            print(data)
            instructions = data.split(';')
            #make sure that the messsage is intended for this cozmo
            if instructions[0] == myName:
                if len(instructions) == 5:
                    #we know that this is a message involving movement
                    #cast the last two entries as integers
                    instructions[3] = int(instructions[3])
                    instructions[4] = int(instructions[4])
                    #next, we will want to move forward if the x distance is not 0 or turn and move backward
                    if instructions[1] == 'B':
                        #Cozmo should turn 180 degrees
                        robot.turn_in_place(cozmo.util.degrees(180)).wait_for_completed()
                        pass
                    if instructions[3] > 0:
                        #Cozmo should now move forward, if necessary.  we will use a default speed of 150 millimeters per second
                        #this is an action, which means that it cannot do this while also being instructed to perform another
                        #action, like speaking, unless you set the action to be done in parallel
                        robot.drive_straight(cozmo.util.distance_mm(instructions[3]), cozmo.util.speed_mmps(150)).wait_for_completed()
                        
                        #examples of more settings, in case you want to run movement in parallel with another action, like speech
                        #robot.drive_straight(distance_mm(instructions[3]), speed_mmps(150), should_play_anim=False, in_parallel=True)
                        
                        #this function is not in the same category as drive_straight, as far as actions go
                        #rate x time = distance
                        #seconds = instructions[3]//150
                        #robot.drive_wheels(150, 150, None, None, seconds)
                        
                    #next, deal with left/right movement
                    if instructions[2] == 'L':
                        #Cozmo should turn 90 degrees counterclockwise
                        robot.turn_in_place(cozmo.util.degrees(90)).wait_for_completed()
                    elif instructions[2] == 'R':
                        #Cozmo should turn 90 degrees clockwise
                        robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
                    if instructions[4] > 0:
                        #Cozmo should now move forward, if necessary.  we will use a default speed of 150 millimeters per second
                        #this is an action, which means that it cannot do this while also being instructed to perform another
                        #action, like speaking.  
                        robot.drive_straight(cozmo.util.distance_mm(instructions[4]), cozmo.util.speed_mmps(150)).wait_for_completed()
                        
                        #if you want Cozmo to talk and speak, you need to use this other function
                        #rate x time = distance
                        #seconds = instructions[4]//150
                        #robot.drive_wheels(150, 150, None, None, seconds)                        

                    robot.say_text(instructions[0]).wait_for_completed()
                    
                elif len(instructions) == 3:
                    #these messages are of the form "name;headValue;liftValue"
                    #this is where we move the tractor or the head
                    #Cozmo's lift values seem to run from 0 to 1.0
                    
                    '''From Anki's web site:
                    
                    height (cozmo.util.Distance): The height of the lift above the ground. 
                    ratio (float): The ratio from 0.0 to 1.0 that the lift is raised from the ground. 
                    angle (cozmo.util.Angle): The angle of the lift arm relative to the ground.
                    
                    This is helpful:
                    https://www.pydoc.io/pypi/cozmo-1.0.1/autoapi/robot/index.html
                    
                    '''
                    #Cozmo's head angle runs from cozmo.robot.MIN_HEAD_ANGLE to cozmo.robot.MAX_HEAD_ANGLE
                    #we want to make sure that any value included in the instructions does not exceed these values
                    
                    #headangle should be a value between -25 degrees and 44.5 degrees
                    headAngle = float(instructions[1])
                    
                    #this lets us know the maximum and minimum values for this parameter
                    #print("max angle: " + str(cozmo.robot.MAX_HEAD_ANGLE)) #44.5 degrees
                    #print("min angle: " + str(cozmo.robot.MIN_HEAD_ANGLE)) #-25 degrees
                    
                    #make sure that headangle is between 44.5 degrees and - 25 degrees
                    headAngle = max(-25, min(headAngle, 44.5))
                    
                    #set up the action
                    action1 = robot.set_head_angle(degrees(headAngle))
                    print("set action 1")
                    #this lets us know the maximum and minimum values for these parameters
                    #print("max height: " + str(cozmo.robot.MAX_LIFT_HEIGHT)) #92 mm
                    #print("min height: " + str(cozmo.robot.MIN_LIFT_HEIGHT))   #32 mm                 

                    liftArm = float(instructions[2])
                    
                    #for our purposes, the lift arm should be between 0.0 and 1.0
                    liftArm = max(0, min(liftArm, 1.0))
                    
                    #set up the action, and run the two actions in parallel
                    action2 = robot.set_lift_height(liftArm, in_parallel=True)
                    print("set action2")
                    
                    action1.wait_for_completed()
                    action2.wait_for_completed()

                    
                #have Cozmo send a message indicating that it is done.  change the message if you want to 
                #progress through the dance routine
                s.sendall(b"Done")

cozmo.run_program(cozmo_program)
