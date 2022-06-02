###################################################################################
# Choreographic Interface                                                         #
# Author(s): Jordan Kruguer, Lins Derry                                           #
# Version Note: This file version is adapted for use in the HAMS Lightbox Gallery #
#               for the set of projects associated with Curatorial A(i)gents.     #
###################################################################################

## Required Packages and Libraries ##
import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import pose_embedding # CI Module - Author(s): Jordan Kruguer, Lins Derry
import pyautogui, sys
import numpy as np
import time
import math
import os
import pickle
import keyboard
import sys
#import sonification #CI Sonification Module - Author(s): Maximilian Mueller (UNCOMMENT FOR USE)

## Defining import objects for landmark tracking ##
mp_holistic = mp.solutions.holistic # for holistic landmarks
mp_drawing = mp.solutions.drawing_utils # for drawing landmark feedback
embed = pose_embedding.FullBodyPoseEmbedder() # for CI Module - pose embedder 

## Defining global variables - user feedback ##

# BGR code for mp_drawing
lav = (178, 160, 187) 
fuschia = (143, 18, 172)
green = (75, 79, 33)
blue = (144, 59, 42)
maroon = (54, 37, 122)
turquoise = (233, 206, 0)
redOrange = (58, 45, 240)
midnight = (49, 26, 29)
white = (255, 255, 255)
black = (0, 0, 0)

lmColor = lav #default landmark color
connColor = lmColor #cdefault connector color
gesture_text = ''


## Set maps, gesture map, set classifier (LogReg) - (For other available sets contact author(s)) ##
set_maps = {'HAMS-LB': {1:'neutral',2:'circle',3:'handsShoulders',4:'sideT',5:'hips',6:'Vdown',7:'track'}}
classifiers = {'HAMS-LB':'HAMS-LB_LogReg_pose_classifier.pkl'} # New set with universal 'track
gesture_map = {'circle':'refresh','handsShoulders':'zoomIn','sideT':'zoomOut','hips':'scrollUp','Vdown':'scrollDown','track':'track','neutral':'neutral'} # for set HAMS-LB

# Set maps for feedback
gesture_map_text = {'refresh':'REFRESH','zoomIn':'ZOOM IN','zoomOut':'ZOOM OUT','scrollUp':'SCROLL UP','scrollDown':'SCROLL DOWN','track':'TRACK','neutral':'NEUTRAL'} 
lmColor_map = {'refresh':blue,'zoomIn':turquoise,'zoomOut':fuschia,'scrollUp':green,'scrollDown':maroon,'track':turquoise,'neutral':white}
connColor_map = {'refresh':white,'zoomIn':fuschia,'zoomOut':turquoise,'scrollUp':maroon,'scrollDown':green,'track':lav,'neutral':blue}

## Set/Map selection ##
selection = 'HAMS-LB'
class_map = set_maps[str(selection)]
pkl_filename = classifiers[str(selection)]
with open(pkl_filename, 'rb') as file:
    logisticRegr = pickle.load(file)

## Define default state variables ##
which_hand = 'right'
mouse_down = False
click = False
reset_hand = False
buffer_threshold = 5 # frames
# action_list = ['refresh', 'zoomIn', 'zoomOut', 'scrollUp', 'scrollDown'] # exclude neutral and track
action_list = ['refresh', 'zoomIn', 'zoomOut', 'scrollUp', 'scrollDown', 'neutral', 'track'] # exclude track
buffer = {key: 0 for key in action_list} # counts number of frames in a row a particular action has been registered. 

#declaring these to avoid error on start
action = 'neutral' 
execute = False

## audio variable declaration
currentCoords = [0,0]
lastCoords = [0,0]
zoomLevel = 0
cursorX = 0
cursorY = 0
trackingActive = False
lastAction = ''


## Main loop ##
cap = cv2.VideoCapture(0) # open web cam
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        if keyboard.is_pressed("q"):
            print('q pressed. Exiting program...')
            sys.exit()
        
        success, image = cap.read() # capture frame
        start = time.time()

        ## Process input frame ##
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # Flip the image horizontally for a later selfie-view display. Convert the BGR image to RGB.
        image.flags.writeable = False # To improve performance, optionally mark the image as not writeable to pass by reference.
        results = holistic.process(image) # Process the image and detect landmarks using the holistic model.
        image.flags.writeable = True # Draw landmark annotation on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert the image color back so it can be displayed

        ## Pose classification - current frame ##
        output_frame = image.copy()
        pose_landmarks = results.pose_landmarks
        if pose_landmarks is not None:
            # Reshape image to apply embedding and pass to classifier
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
            pose_landmark_embedding = embed(pose_landmarks) # embed using CI module
            pose_landmark_embedding_flattened = pose_landmark_embedding.flatten()
            classification = logisticRegr.predict(pose_landmark_embedding_flattened.astype(float).reshape(1,-1))
            gesture = class_map[classification[0]]
            action = gesture_map[gesture]

            ## SOFT DELETION OF NEUTRAL UNTIL WE DECIDE WHETHER WE ACTUALLY WANT TO GET RID OF IT FROM THE CLASSIFIER ##
            if action == 'neutral':
                action = 'track'

        screenWidth, screenHeight = pyautogui.size() # screen's height/width in pixels - scaling positions from input window

        ## Collect specific landmarks of interest - shoulders for scaling all movements ##
        try:
            x_shoulderR = results.pose_landmarks.landmark[12].x * screenWidth # --> Right Shoulder (x-coor)
            y_shoulderR = results.pose_landmarks.landmark[12].y * screenHeight # --> Right Shoulder (y-coor)
            x_shoulderL = results.pose_landmarks.landmark[11].x * screenWidth # --> Left Shoulder (x-coor)
            y_shoulderL = results.pose_landmarks.landmark[11].y * screenHeight # --> Left Shoulder (y-coor)
            distance_between_shoulders = np.linalg.norm(np.array((x_shoulderR,y_shoulderR))-np.array((x_shoulderL,y_shoulderL)))
        except Exception as e:
            pass
            
        ## Collect specific landmarks of interest - dexterous movements (Using hand landmarks) ##
        try:
            x_wristR = results.left_hand_landmarks.landmark[0].x * screenWidth # --> Right Wrist (x-coor)
            y_wristR = results.left_hand_landmarks.landmark[0].y * screenHeight # --> Right Wrist (y-coor)
            x_thumbR = results.left_hand_landmarks.landmark[2].x * screenWidth # --> Right Thumb Tip (x-coor) #changing thumb from tip to middle (4 to 2)
            y_thumbR = results.left_hand_landmarks.landmark[2].y * screenHeight # --> Right Thumb Tip (y-coor) #changing thumb from tip to middle (4 to 2)
            x_thumbTipR = results.left_hand_landmarks.landmark[4].x * screenWidth # --> Right Thumb Tip (x-coor) 
            y_thumbTipR = results.left_hand_landmarks.landmark[4].y * screenHeight # --> Right Thumb Tip (y-coor) 
            x_indexR = results.left_hand_landmarks.landmark[8].x * screenWidth # --> Right Index Finger Tip (x-coor)
            y_indexR = results.left_hand_landmarks.landmark[8].y * screenHeight # --> Right Index Finger Tip (y-coor)
            x_pinkyR = results.left_hand_landmarks.landmark[20].x * screenWidth # --> Right Pinky Finger Tip (x-coor)
            y_pinkyR = results.left_hand_landmarks.landmark[20].y * screenHeight # --> Right Pinky Finger Tip (y-coor)
        except Exception as e:
            pass
        try:
            x_wristL = results.right_hand_landmarks.landmark[0].x * screenWidth # --> Left Wrist (x-coor) 
            y_wristL = results.right_hand_landmarks.landmark[0].y * screenHeight # --> Left Wrist (y-coor)
            x_thumbL = results.right_hand_landmarks.landmark[2].x * screenWidth # --> Left Thumb Tip (x-coor)  #changing thumb from tip to middle (4 to 2)
            y_thumbL = results.right_hand_landmarks.landmark[2].y * screenHeight # --> Left Thumb Tip (y-coor) #changing thumb from tip to middle (4 to 2)
            x_thumbTipL = results.right_hand_landmarks.landmark[4].x * screenWidth # --> Left Thumb Tip (x-coor)  
            y_thumbTipL = results.right_hand_landmarks.landmark[4].y * screenHeight # --> Left Thumb Tip (y-coor) 
            x_indexL = results.right_hand_landmarks.landmark[8].x * screenWidth # --> Left Index Finger Tip (x-coor)
            y_indexL = results.right_hand_landmarks.landmark[8].y * screenHeight # --> Left Index Finger Tip (y-coor)
            x_pinkyL = results.right_hand_landmarks.landmark[20].x * screenWidth # --> Left Pinky Finger Tip (x-coor)
            y_pinkyL = results.right_hand_landmarks.landmark[20].y * screenHeight # --> RigLeftht Pinky Finger Tip (y-coor)
        except Exception as e:
            pass

        ## Determine handedness (L vs. R) ##
        try:
            x_face_center = results.pose_landmarks.landmark[0].x * screenWidth # --> Nose (x-coor)
            y_face_center = results.pose_landmarks.landmark[0].y * screenHeight # --> Nose (y-coor)
            
            x_earR = results.pose_landmarks.landmark[8].x * screenWidth # --> right ear (x-coor)
            y_earR = results.pose_landmarks.landmark[8].y * screenHeight # --> right ear (y-coor)
            x_earL = results.pose_landmarks.landmark[7].x * screenWidth # --> left ear (x-coor)
            y_earL = results.pose_landmarks.landmark[7].y * screenHeight # --> left ear (y-coor)
        except Exception as e:
            pass

        try:
            distance_between_shoulderR_faceCenter = np.linalg.norm(np.array((x_face_center,y_face_center))-np.array((x_shoulderR,y_shoulderR)))
            distance_between_shoulderL_faceCenter = np.linalg.norm(np.array((x_face_center,y_face_center))-np.array((x_shoulderL,y_shoulderL)))
            if (distance_between_shoulderL_faceCenter>distance_between_shoulderR_faceCenter) and (distance_between_shoulderR_faceCenter<(distance_between_shoulders*0.65)):
                which_hand = 'left' # mirrored
            elif (distance_between_shoulderL_faceCenter<distance_between_shoulderR_faceCenter) and (distance_between_shoulderL_faceCenter<(distance_between_shoulders*0.65)):
                which_hand = 'right' # mirrored
            
            # handedness determined by ears
            # distance_between_shoulderR_earR = np.linalg.norm(np.array((x_earR,y_earR))-np.array((x_shoulderR,y_shoulderR)))
            # distance_between_shoulderL_earL = np.linalg.norm(np.array((x_earL,y_earR))-np.array((x_shoulderL,y_shoulderL)))
            # if (distance_between_shoulderL_earL>distance_between_shoulderR_earR) and (distance_between_shoulderR_earR<(distance_between_shoulders*0.6)):
            #     which_hand = 'left' # mirrored
            # elif (distance_between_shoulderL_earL<distance_between_shoulderR_earR) and (distance_between_shoulderL_earL<(distance_between_shoulders*0.6)):
            #     which_hand = 'right' # mirrored

        except Exception as e:
            pass

        ## Establish additional constraint for zoomin ##
        try:
            distance_between_shoulderR_wristR = np.linalg.norm(np.array((x_wristL,y_wristL))-np.array((x_shoulderR,y_shoulderR)))
            distance_between_shoulderL_wristL = np.linalg.norm(np.array((x_wristR,y_wristR))-np.array((x_shoulderL,y_shoulderL)))
            if (action == 'zoomIn'):
                if (distance_between_shoulderR_wristR<(distance_between_shoulders*0.75)) and (distance_between_shoulderL_wristL<(distance_between_shoulders*0.75)):
                    pass 
                else:
                    action = 'track' # correct for unintended zoom in
        except Exception as e:
            pass
        try:
            if action == 'track':
                if (which_hand == 'right'):
                    handSpan = np.linalg.norm(np.array((x_pinkyR,y_pinkyR))-np.array((x_thumbTipR,y_thumbTipR)))
                else:
                    handSpan = np.linalg.norm(np.array((x_pinkyL,y_pinkyL))-np.array((x_thumbTipL,y_thumbTipL)))
                        
                normalizedHandSpan = handSpan / distance_between_shoulders
                # print("handspan: " + str(normalizedHandSpan))
                if normalizedHandSpan < .2:
                    action = 'neutral' # @MAX --> can we change this to track? Kind of defeats the purpose of what you were doing here but trying to get rid of neutral.
            
        except Exception as e:
            pass
        ## LOG Actions ##
        # if (action == 'track') or (action == 'neutral') or (action == None): 
        # if (action == 'track') or (action == None): # adding neutral to the buffer
        if (action == None):
            gesture_text = ''
            execute = False
            print ('action is none')
            pass
        else:
            if (len(list(set(list(buffer.values())))) == 1): # All values are the same (e.g. == 0) - base case
                buffer[action] += 1
            else:
                stacked_keys = [] 
                for key in buffer.keys():
                    if buffer[key]>0:
                        stacked_keys.append(key)
                if (len(stacked_keys) == 1) and (stacked_keys[0]==action):
                    buffer[action] += 1
                    if buffer[action] > buffer_threshold:
                        if execute == False:
                            execute = True
                            action_to_execute = action
                            lmColor = lmColor_map[action]
                            connColor = connColor_map[action]
                            gesture_text = gesture_map_text[action]
                            buffer = {key: 0 for key in action_list}  # reset dictionary
                            buffer[action] = int(buffer_threshold) + 1 #added so current filled action continues to run
                            lastExecutedAction = action
                    else:
                        execute = False ## added so actions stop when neutral is detected 
                        action_to_execute = 'neutral'
                        gesture_text = ''


                else:
                    buffer = {key: 0 for key in action_list}  # reset dictionary
                    buffer[action] += 1 # add count to new action
                    # gesture_text = ''

        ## EXECUTE Actions ##
        try:
            if execute == True:
                if (action_to_execute == 'refresh'):
                    pyautogui.hotkey('command', 'r')
                    zoomLevel = 0 
                elif (action_to_execute == 'zoomIn'):
                    #pyautogui.hotkey('command', '+')
                    pyautogui.scroll(5)
                    if zoomLevel > -10:
                        zoomLevel -= totalTime
                        audioParam = zoomLevel 
                    # sonification.sendOSCMessage(action, audioParam)
                elif (action_to_execute == 'zoomOut'):
                    #pyautogui.hotkey('command', '-')
                    pyautogui.scroll(-5)
                    if zoomLevel < 10:
                        zoomLevel += totalTime
                        audioParam = zoomLevel 
                    # sonification.sendOSCMessage(action, audioParam)
                elif (action_to_execute == 'scrollUp'):
                    pyautogui.scroll(5)
                    # pyautogui.press('right') #Giulia
                elif (action_to_execute == 'scrollDown'):
                    pyautogui.scroll(-5)
                    # pyautogui.press('left') #Giulia
                elif (action_to_execute == 'track'):
                    trackingActive = True
                    #sonification.sendOSCMessage('trackStart','')
                if (action_to_execute != 'track'):
                    #sonification.sendOSCMessage(action_to_execute,'')
                    trackingActive = False
            else:
                gesture_text = ''
                if (trackingActive == True): 
                    trackingActive = False

        except Exception as e:
            pass
    
        ## Move and Drag Mouse ##
        
        try:
            #if (trackingActive == True): # LET MOUSE GO ALWAYS
            drag_threshold = distance_between_shoulders*0.2
            if (which_hand == 'right'):
                # Mouse displacement y-axis correction
                y_disp = y_indexR-y_wristR

                grab_distance = np.linalg.norm(np.array((x_thumbR,y_thumbR))-np.array((x_indexR,y_indexR)))
                # print(str(grab_distance) + 'GRAB DISTANCE')
                if grab_distance > drag_threshold: # adjust
                    if mouse_down == True:
                        pyautogui.mouseUp()
                        mouse_down = False
                        print('MOUSE UP!')
                        gesture_text = 'TRACK'
                        #sonification.sendMouseDown(mouse_down)
                    pyautogui.moveTo(x_wristR, y_wristR+y_disp) # Move cursor
                    cursorX = x_wristR
                    cursorY = y_wristR
                elif grab_distance < drag_threshold:
                    if mouse_down == False:
                        pyautogui.mouseDown()
                        mouse_down = True
                        print('MOUSE DOWN!')
                        gesture_text = 'DRAG'
                        #sonification.sendMouseDown(mouse_down)
                    pyautogui.dragTo(x_wristR, y_wristR+y_disp,button='left') # Drag cursor
                    cursorX = x_wristR
                    cursorY = y_wristR
            if which_hand == 'left':
                # Mouse displacement y-axis correction
                y_disp = y_indexL-y_wristL

                grab_distance = np.linalg.norm(np.array((x_thumbL,y_thumbL))-np.array((x_indexL,y_indexL)))
                if grab_distance > drag_threshold: # adjust
                    if mouse_down == True:
                        pyautogui.mouseUp()
                        mouse_down = False
                        print('MOUSE UP!')
                        gesture_text = 'TRACK'
                        #sonification.sendMouseDown(mouse_down)
                    pyautogui.moveTo(x_wristL, y_wristL+y_disp) # Move cursor
                    cursorX = x_wristL
                    cursorY = y_wristL
                elif grab_distance < drag_threshold:
                    if mouse_down == False:
                        pyautogui.mouseDown()
                        mouse_down = True
                        print('MOUSE DOWN!')
                        gesture_text = 'DRAG'
                        #sonification.sendMouseDown(mouse_down)
                    pyautogui.dragTo(x_wristL, y_wristL+y_disp,button='left') # Drag cursor
                    cursorX = x_wristR
                    cursorY = y_wristR
            if cursorX is not None:
                currentCoords = [cursorX, cursorY]
                cursorDifference = math.sqrt( ((currentCoords[0]-lastCoords[0])**2)+((currentCoords[1]-lastCoords[1])**2) )
                cursorAccel = cursorDifference
                lastCoords = currentCoords
                #sonification.sendOSCMessage('track', cursorX)
                #sonification.sendXAccelerationOSC(cursorAccel)
            # else: mouse_down = False
        except Exception as e:
            pass
        
        ## Mouse Select ##
        try:
            select_threshold = distance_between_shoulders * 0.15
            if which_hand == 'right':
                select_distance = np.linalg.norm(np.array((x_indexL,y_indexL))-np.array((x_wristR,y_wristR)))
                if select_distance < select_threshold:
                    if click == False:
                        pyautogui.click(button='left')
                        click = True
                        sonification.sendOSCMessage('select','')
                elif select_distance > select_threshold:
                    if click == True:
                        click = False
            if which_hand == 'left':
                select_distance = np.linalg.norm(np.array((x_indexR,y_indexR))-np.array((x_wristL,y_wristL)))
                if select_distance < select_threshold:
                    if click == False:
                        pyautogui.click(button='left')
                        click = True
                        #sonification.sendOSCMessage('select','')
                elif select_distance > select_threshold:
                    if click == True:
                        click = False
        except Exception as e:
            pass        


        ## Determine FPS ##     
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        buffer_threshold = fps # Update buffer threshold

        ## Format User Feedback ... Position feedback bottom right corner (nudged (5, 33) to look just right on MacBook Air) ##
        rad = 4
        thick = 3

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
        if which_hand == 'right':
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
        elif which_hand == 'left':
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        
        capWidth  = cap.get(3) # input frame width
        capHeight = cap.get(4) # input frame height
        scale = 0.5 # Scale feedback 0.23 for Lightbox
        newCapWidth = int(screenWidth * (scale))
        newCapHeight = int(newCapWidth * capHeight / capWidth)

        capX = screenWidth - newCapWidth - 5
        capY = screenHeight - newCapHeight - 33
        #cv2.putText(image, f"FPS: {int(fps)}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2) # provide feedback on current FPS
        
        if mouse_down == True:
            # Mac
            cv2.putText(image, 'DRAG', (5, 150), cv2.FONT_HERSHEY_DUPLEX, 4, midnight, 2, cv2.LINE_4)
            # Lightbox
            # cv2.putText(image, 'DRAG', (5, 75), cv2.FONT_HERSHEY_DUPLEX, 2.5, midnight, 2, cv2.LINE_4)
        elif click == True:
            # Mac
            cv2.putText(image, 'SELECT', (5, 150), cv2.FONT_HERSHEY_DUPLEX, 4, midnight, 2, cv2.LINE_4)
            # Lightbox
            # cv2.putText(image, 'SELECT', (5, 75), cv2.FONT_HERSHEY_DUPLEX, 2.5, midnight, 2, cv2.LINE_4)
        else:
            # Mac
            cv2.putText(image, f'{gesture_text}', (5, 150), cv2.FONT_HERSHEY_DUPLEX, 4, midnight, 2, cv2.LINE_4)
            # Lightbox
            # cv2.putText(image, f'{gesture_text}', (5, 75), cv2.FONT_HERSHEY_DUPLEX, 2.5, midnight, 2, cv2.LINE_4)

        cv2.imshow('Choreographic Interface', cv2.resize(image, (newCapWidth, newCapHeight)))
        cv2.setWindowProperty('Choreographic Interface', cv2.WND_PROP_TOPMOST, 1) # keeps feedback window most front
        cv2.moveWindow('Choreographic Interface', capX,capY) # relocate feedback

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
