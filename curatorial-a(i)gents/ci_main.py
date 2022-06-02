###################################################################################
# Choreographic Interface                                                         #
# Author(s): Jordan Kruguer, Lins Derry                                           #
# Version Note: This file version is adapted for use in the HAMS Lightbox Gallery #
#               for the set of projects associated with Curatorial A(i)gents.     #
###################################################################################

## Required Packages and Libraries ##
from shutil import which
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
#import sonification #CI #sonification Module - Author(s): Maximilian Mueller (UNCOMMENT FOR USE)
import collections
# import statistics
# from statistics import mode
# from scipy.stats import mode
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
max_distance_wrist_index = 0

modalActionList = collections.deque(['neutral']* 10, maxlen=10) #initialize modal list with neutral
modalAction = 'neutral'
modalMouseButtonList = collections.deque(['up']* 10, maxlen=10) #initialize modal list with up
modalMouseState = 'up'
missingLandmarkBuffer = collections.deque([False]* 10, maxlen=10) #initialize modal list with False
clickBuffer = collections.deque([False]* 10, maxlen=10) #initialize modal list with False


#declaring these to avoid error on start
action = 'neutral' 
execute = False
lastExecutedAction = 'neutral'
handChopOrientation = False

## audio variable declaration
currentCoords = [0,0]
lastCoords = [0,0]
zoomLevel = 0
cursorX = 0
cursorY = 0
totalTime = .01


def mode(buffer):
    value = max(buffer, key = buffer.count)
    return value
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

            ## @LINS: SOFT DELETION OF NEUTRAL UNTIL WE DECIDE WHETHER WE ACTUALLY WANT TO GET RID OF IT FROM THE CLASSIFIER ##
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
            if which_hand == 'right':
                missingLandmarkBuffer.append(False)
        except Exception as e:
            if which_hand == 'right':
                missingLandmarkBuffer.append(True)
                clickBuffer.append(False)
            pass
        try:
            y_wristR = results.left_hand_landmarks.landmark[0].y * screenHeight # --> Right Wrist (y-coor)
        except:
            pass
        try:
            x_thumbR = results.left_hand_landmarks.landmark[2].x * screenWidth # --> Right Thumb Tip (x-coor) #changing thumb from tip to middle (4 to 2)
        except:
            pass
        try:
            y_thumbR = results.left_hand_landmarks.landmark[2].y * screenHeight # --> Right Thumb Tip (y-coor) #changing thumb from tip to middle (4 to 2)
        except:
            pass
        try:
            x_thumbTipR = results.left_hand_landmarks.landmark[4].x * screenWidth # --> Right Thumb Tip (x-coor)
        except:
            pass
        try:
            y_thumbTipR = results.left_hand_landmarks.landmark[4].y * screenHeight # --> Right Thumb Tip (y-coor) 
        except:
            pass
        try:
            x_indexR = results.left_hand_landmarks.landmark[8].x * screenWidth # --> Right Index Finger Tip (x-coor)
        except:
            pass
        try:
            y_indexR = results.left_hand_landmarks.landmark[8].y * screenHeight # --> Right Index Finger Tip (y-coor)
        except:
            pass
        try:
            x_pinkyR = results.left_hand_landmarks.landmark[20].x * screenWidth # --> Right Pinky Finger Tip (x-coor)
        except:
            pass
        try:
            y_pinkyR = results.left_hand_landmarks.landmark[20].y * screenHeight # --> Right Pinky Finger Tip (y-coor)
        except:
            pass
        try:
            x_wristL = results.right_hand_landmarks.landmark[0].x * screenWidth # --> Left Wrist (x-coor)
            if which_hand == 'left':
                missingLandmarkBuffer.append(False)
        except:
            if which_hand == 'left':
                missingLandmarkBuffer.append(True)
                clickBuffer.append(False)
            pass
        try:
            y_wristL = results.right_hand_landmarks.landmark[0].y * screenHeight # --> Left Wrist (y-coor)
        except:
            pass
        try:
            x_thumbL = results.right_hand_landmarks.landmark[2].x * screenWidth # --> Left Thumb Tip (x-coor)  #changing thumb from tip to middle (4 to 2)
        except:
            pass
        try:
            y_thumbL = results.right_hand_landmarks.landmark[2].y * screenHeight # --> Left Thumb Tip (y-coor) #changing thumb from tip to middle (4 to 2)
        except:
            pass
        try:
            x_thumbTipL = results.right_hand_landmarks.landmark[4].x * screenWidth # --> Left Thumb Tip (x-coor)  
        except:
            pass
        try:
            y_thumbTipL = results.right_hand_landmarks.landmark[4].y * screenHeight # --> Left Thumb Tip (y-coor) 
        except:
            pass
        try:
            x_indexL = results.right_hand_landmarks.landmark[8].x * screenWidth # --> Left Index Finger Tip (x-coor)
        except:
            pass
        try:
            y_indexL = results.right_hand_landmarks.landmark[8].y * screenHeight # --> Left Index Finger Tip (y-coor)
        except:
            pass
        try:
            x_pinkyL = results.right_hand_landmarks.landmark[20].x * screenWidth # --> Left Pinky Finger Tip (x-coor)
        except:
            pass
        try:
            y_pinkyL = results.right_hand_landmarks.landmark[20].y * screenHeight # --> RigLeftht Pinky Finger Tip (y-coor)
        except:
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
                if normalizedHandSpan < .2:
                    handChopOrientation = True 
                else: handChopOrientation = False
        except Exception as e:
            pass
        
        
        ## LOG Actions ##
        if mode(missingLandmarkBuffer) == True or action == None:
            action = 'neutral' #modal buffer has to be filled with something, so if we can't act, fill it with neutral
        if action == 'track':
            if handChopOrientation == True:
                action = 'neutral'
        modalActionList.append(action)
        # modalAction = mode(modalActionList)
        modalAction = mode(modalActionList)
        #print(modalAction)
        lmColor = lmColor_map[modalAction]
        connColor = connColor_map[modalAction]
        gesture_text = gesture_map_text[modalAction]
        #for the text, when making track active always only set to track if nothing else


        ## EXECUTE Actions ##
        try:
            if (modalAction == 'refresh' and lastExecutedAction != 'refresh'):
                pyautogui.hotkey('command', 'r')
                # pyautogui.hotkey('ctrl', 'r') #for LB
                zoomLevel = 0 
            elif (modalAction == 'zoomIn'):
                #pyautogui.hotkey('command', '+')
                pyautogui.scroll(5)
                if zoomLevel > -10:
                    zoomLevel -= totalTime
                    audioParam = zoomLevel 
            elif (modalAction == 'zoomOut'):
                #pyautogui.hotkey('command', '-')
                pyautogui.scroll(-5)
                if zoomLevel < 10:
                    zoomLevel += totalTime
                    audioParam = zoomLevel 
            elif (modalAction == 'scrollUp'):
                pyautogui.scroll(5)
                # pyautogui.scroll(50) # for LB
                # pyautogui.press('right') #Giulia
            elif (modalAction == 'scrollDown'):
                pyautogui.scroll(-5)
                # pyautogui.scroll(50) # for LB
                # pyautogui.press('left') #Giulia
            '''
            if modalAction == 'track':
                if modalAction != lastExecutedAction:
                    #sonification.sendOSCMessage('trackStart','')
            else:
                #sonification.sendOSCMessage(modalAction,'')
            '''
            #if lastExecutedAction != modalAction:
                #print ('pose changed')
                #print ('modal action is ' + modalAction)
            lastExecutedAction = modalAction
        except Exception as e:
            pass
    
        ## Move and Drag Mouse ##

        ##set handedness vars
        try:

            # Vitruvian adjustment #
            arm_length = distance_between_shoulders*2


            if which_hand == 'left':
                wristX = x_wristL
                wristY = y_wristL
                thumbX = x_thumbL
                thumbY = y_thumbL
                indexX = x_indexL
                indexY = y_indexL

                # For vitruvian reach logic
                shoulderX = x_shoulderL
                shoulderY = y_shoulderL
                max_reach_y = y_shoulderL+arm_length
                min_reach_y = y_shoulderL+arm_length
                if (wristY-shoulderY)>0: # wrist is higher than shoulder  
                    cursorbump_y = ((wristY-shoulderY)/max_reach_y)*max_distance_wrist_index
                else: # wrist is lower than shoulder
                    cursorbump_y = -1*(((wristY-shoulderY)/max_reach_y)*max_distance_wrist_index)
                #reachY = y_shoulderL+arm_length
                #reachX = x_shoulderL+arm_length
                #print(f'Left Hand X Sign: {np.sign(wristX-shoulderX)}')
                #print(f'Left Hand Y Sign: {np.sign(wristY-shoulderY)}')

            else:
                wristX = x_wristR
                wristY = y_wristR
                thumbX = x_thumbR
                thumbY = y_thumbR
                indexX = x_indexR
                indexY = y_indexR


                # For vitruvian reach logic
                shoulderX = x_shoulderR
                shoulderY = y_shoulderR
                max_reach_y = y_shoulderR+arm_length
                min_reach_y = y_shoulderR+arm_length
                if (wristY-shoulderY)>0: # wrist is higher than shoulder  
                    cursorbump_y = ((wristY-shoulderY)/max_reach_y)*max_distance_wrist_index
                else: # wrist is lower than shoulder
                    cursorbump_y = -1*(((wristY-shoulderY)/max_reach_y)*max_distance_wrist_index)
                #reachY = y_shoulderR+arm_length
                #reachX = x_shoulderR+arm_length
                #print(f'Right Hand X Sign: {np.sign(wristX-shoulderX)}')
                #print(f'Right Hand Y Sign: {np.sign(wristY-shoulderY)}')

        except Exception as e:
            #print(e)
            pass

        try:
            # Vitruvian adjustment part 2 #
            if np.abs(indexY-wristY)>max_distance_wrist_index:
                max_distance_wrist_index = np.abs(indexY-wristY) # proxy for hand size, always take largest value found e.g. should get better over time and adjust to person
        except:
            pass
        
        ##end set handedness vars
        # BOOKMARK
        
        if mode(modalMouseButtonList) == 'down':
            if mouse_down == False:
                mouse_down = True
                pyautogui.mouseDown()
                #print('MOUSE DOWN!')
                gesture_text = 'DRAG'
                #sonification.sendMouseDown(mouse_down)
        else:
            if mouse_down == True:
                mouse_down = False
                pyautogui.mouseUp()
                mouse_down = False
                #print('MOUSE UP!')
                #sonification.sendMouseDown(mouse_down)
        
        try:
            if (modalAction == 'track'): # LET MOUSE GO ALWAYS
                drag_threshold = distance_between_shoulders*0.2
                
                # Mouse displacement y-axis correction
                # y_disp = indexY-wristY
                y_disp = 0
                cursorX = wristX
                print(f"CURSOR BUMP: {cursorbump_y}")
                cursorY = wristY+cursorbump_y
                grab_distance = np.linalg.norm(np.array((thumbX,thumbY))-np.array((indexX,indexY)))
                if grab_distance > drag_threshold or handChopOrientation == True: # adjust
                    modalMouseButtonList.append('up')
                elif grab_distance < drag_threshold and modalAction == 'track':
                    modalMouseButtonList.append('down')
                if cursorX is not None and cursorX > 0:
                    currentCoords = [cursorX, cursorY]
                    cursorDifference = math.sqrt( ((currentCoords[0]-lastCoords[0])**2)+((currentCoords[1]-lastCoords[1])**2) )
                    cursorAccel = cursorDifference
                    #print(f'Cursor Acceleration: {cursorAccel}')
                    if cursorAccel > 15:
                        v1 = Vector2(lastCoords[0], lastCoords[1])
                        v2 = Vector2(cursorX, cursorY)
                        # print ('v1 : '+v1)
                        # print ('v2 : '+v2)
                        # print(lerp(v1,v2,1))
                        v3 = lerp(v1, v2, totalTime*.05)
                        pyautogui.moveTo(v3.x,v3.y,0)
                    lastCoords = currentCoords

                    #sonification.sendOSCMessage('track', cursorX)
                    #sonification.sendXAccelerationOSC(cursorAccel)
        except Exception as e:
            #print(e)
            modalMouseButtonList.append('up')
            pass
     

        ## Mouse Select ##
        try:
            if modalAction == 'track':
                select_threshold = distance_between_shoulders * 0.15
                if which_hand == 'right':
                    select_distance = np.linalg.norm(np.array((x_indexL,y_indexL))-np.array((x_wristR,y_wristR)))
                if which_hand == 'left':
                    select_distance = np.linalg.norm(np.array((x_indexR,y_indexR))-np.array((x_wristL,y_wristL)))
                
                if select_distance < select_threshold:
                    # print('select True added to buffer')
                    clickBuffer.append(True)
                elif select_distance > select_threshold:
                    clickBuffer.append(False)
        
                if mode(clickBuffer) == True:
                    if click == False:
                        click = True
                        pyautogui.click(button='left')
                        #sonification.sendOSCMessage('select','')
                        print('clicked!')
                elif click == True:
                    click = False
                    #sonification.sendOSCMessage('deselect','')

                
        except Exception as e:
            # print (e)
            pass       


        ## Determine FPS ##     
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

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
        scale = 0.75 # Scale feedback 0.23 for Lightbox
        newCapWidth = int(screenWidth * (scale))
        newCapHeight = int(newCapWidth * capHeight / capWidth)

        capX = screenWidth - newCapWidth - 5
        capY = screenHeight - newCapHeight - 33
        #cv2.putText(image, f"FPS: {int(fps)}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2) # provide feedback on current FPS
        
        
        cv2.imshow('Choreographic Interface', cv2.resize(image, (newCapWidth, newCapHeight)))
        cv2.setWindowProperty('Choreographic Interface', cv2.WND_PROP_TOPMOST, 1) # keeps feedback window most front
        cv2.moveWindow('Choreographic Interface', capX,capY) # relocate feedback

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()


