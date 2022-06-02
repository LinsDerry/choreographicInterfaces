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
import collections
from statistics import mode
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
#modalActionList = ['neutral'] * 10 #initialize modal list with neutral
modalActionList = collections.deque(['neutral']* 10, maxlen=30) #initialize modal list with neutral
modalAction = 'neutral'
modalMouseButtonList = collections.deque(['up']* 10, maxlen=10) #initialize modal list with up
modalMouseState = 'up'
essentialMouseLandmarkMissing = False
missingLandmarkBuffer = collections.deque([False]* 10, maxlen=10) #initialize modal list with False

#declaring these to avoid error on start
action = 'neutral' 
execute = False
lastExecutedAction = 'neutral'

## audio variable declaration
currentCoords = [0,0]
lastCoords = [0,0]
zoomLevel = 0
cursorX = 0
cursorY = 0
trackingActive = False
totalTime = .01
# x_wristL, x_thumbL, x_thumbTipL, x_indexL, x_pinkyL = 5 * 0
# y_wristL, y_thumbL, y_thumbTipL, y_indexL, y_pinkyL = 5 * 0
# leftLandmarkVars_x = [x_wristL, x_thumbL, x_thumbTipL, x_indexL, x_pinkyL]

class Vector2:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    def __str__(self):
        return f"x:{self.x}, y:{self.y}"
# v1 = Vector2(100, 100)
# v2 = Vector2(200,200)


def lerp(v1, v2, time): #time is value between 0 and 1
    return v1 * (1- time) + v2 * time

#functions
def GetRightHandXLandMarks(landmarkName, landmarkIndex):
            try:
                landmarkName = results.left_hand_landmarks.landmark[landmarkIndex].x * screenWidth
            except Exception as e:
                print(e)
#end functions

########################
### HELPER FUNCTIONS ###
########################

"""
TODO: make this better/cleaner but for now just put them in functions
"""

def getScaler(results):
    x_shoulderR = results.pose_landmarks.landmark[12].x * screenWidth # --> Right Shoulder (x-coor)
    y_shoulderR = results.pose_landmarks.landmark[12].y * screenHeight # --> Right Shoulder (y-coor)
    x_shoulderL = results.pose_landmarks.landmark[11].x * screenWidth # --> Left Shoulder (x-coor)
    y_shoulderL = results.pose_landmarks.landmark[11].y * screenHeight # --> Left Shoulder (y-coor)
    return np.linalg.norm(np.array((x_shoulderR,y_shoulderR))-np.array((x_shoulderL,y_shoulderL)))

def determineHandedness(results):
    x_face_center = results.pose_landmarks.landmark[0].x * screenWidth # --> Nose (x-coor)
    y_face_center = results.pose_landmarks.landmark[0].y * screenHeight # --> Nose (y-coor)
    x_earR = results.pose_landmarks.landmark[8].x * screenWidth # --> right ear (x-coor)
    y_earR = results.pose_landmarks.landmark[8].y * screenHeight # --> right ear (y-coor)
    x_earL = results.pose_landmarks.landmark[7].x * screenWidth # --> left ear (x-coor)
    y_earL = results.pose_landmarks.landmark[7].y * screenHeight # --> left ear (y-coor)
    distance_between_shoulderR_faceCenter = np.linalg.norm(np.array((x_face_center,y_face_center))-np.array((x_shoulderR,y_shoulderR)))
    distance_between_shoulderL_faceCenter = np.linalg.norm(np.array((x_face_center,y_face_center))-np.array((x_shoulderL,y_shoulderL)))
    if (distance_between_shoulderL_faceCenter>distance_between_shoulderR_faceCenter) and (distance_between_shoulderR_faceCenter<(distance_between_shoulders*0.65)):
        return 'left' # mirrored
    elif (distance_between_shoulderL_faceCenter<distance_between_shoulderR_faceCenter) and (distance_between_shoulderL_faceCenter<(distance_between_shoulders*0.65)):
        return 'right' # mirrored
    
    return which_hand

def executeAction(action,action_to_execute,zoomLevel):
    if (action_to_execute == 'refresh' and lastExecutedAction != 'refresh'):
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
    elif (action_to_execute == 'track' or action_to_execute == 'down'):
        trackingActive = True
        #sonification.sendOSCMessage('trackStart','')
    if (action_to_execute != 'track'):
        #sonification.sendOSCMessage(action_to_execute,'')
        trackingActive = False

def moveAndDragMouse(results,which_hand):

    if which_hand == 'left':
        wristX = results.right_hand_landmarks.landmark[0].x * screenWidth # --> Left Wrist (x-coor)
        wristY = results.right_hand_landmarks.landmark[0].y * screenHeight # --> Left Wrist (y-coor)
        thumbX = results.right_hand_landmarks.landmark[2].x * screenWidth # --> Left Thumb Tip (x-coor)  
        thumbY = results.right_hand_landmarks.landmark[2].y * screenHeight # --> Left Thumb Tip (y-coor)
        thumbTipX = results.right_hand_landmarks.landmark[4].x * screenWidth # --> Left Thumb Tip (x-coor) 
        thumbTipY = results.right_hand_landmarks.landmark[4].y * screenHeight # --> Left Thumb Tip (y-coor) 
        pinkyX = results.right_hand_landmarks.landmark[20].x * screenWidth # --> Left Pinky Finger Tip (x-coor)
        pinkyY = results.right_hand_landmarks.landmark[20].y * screenHeight # --> RigLeftht Pinky Finger Tip (y-coor)
        indexX = results.right_hand_landmarks.landmark[8].x * screenWidth # --> Left Index Finger Tip (x-coor)
        indexY = results.right_hand_landmarks.landmark[8].y * screenHeight # --> Left Index Finger Tip (y-coor)
    else:
        wristX = results.left_hand_landmarks.landmark[0].x * screenWidth # --> Right Wrist (x-coor)
        wristY = results.left_hand_landmarks.landmark[0].y * screenHeight # --> Right Wrist (y-coor)
        thumbX = results.left_hand_landmarks.landmark[2].x * screenWidth # --> Right Thumb Tip (x-coor) 
        thumbY = results.left_hand_landmarks.landmark[2].y * screenHeight # --> Right Thumb Tip (y-coor) 
        thumbTipX = results.left_hand_landmarks.landmark[4].x * screenWidth # --> Right Thumb Tip (x-coor) 
        thumbTipY = results.left_hand_landmarks.landmark[4].y * screenHeight # -->Right Thumb Tip (y-coor) 
        pinkyX = results.left_hand_landmarks.landmark[20].x * screenWidth # --> Right Pinky Finger Tip (x-coor)
        pinkyY = results.left_hand_landmarks.landmark[20].y * screenHeight # --> Right Pinky Finger Tip (y-coor)
        indexX = results.left_hand_landmarks.landmark[8].x * screenWidth # --> Right Index Finger Tip (x-coor)
        indexY = results.left_hand_landmarks.landmark[8].y * screenHeight # --> Right Index Finger Tip (y-coor)


    if action == 'track':
        handSpan = np.linalg.norm(np.array((pinkyX,pinkyY))-np.array((thumbTipX,thumbTipY)))
       
        normalizedHandSpan = handSpan / distance_between_shoulders
        # print("handspan: " + str(normalizedHandSpan))
        if normalizedHandSpan < .2:
            handChopOrientation = True # @MAX --> can we change this to track? Kind of defeats the purpose of what you were doing here but trying to get rid of neutral.
        else: handChopOrientation = False

    if mode(modalMouseButtonList)=='down':
        if mouse_down == False:
            mouse_down = True
            pyautogui.mouseDown()
            gesture_text = 'DRAG'
            #sonification.sendMouseDown(mouse_down)
    else:
        if mouse_down == True:
            mouse_down = False
            pyautogui.mouseUp()
            mouse_down = False
            #sonification.sendMouseDown(mouse_down)
    try:

        if (mode(missingLandmarkBuffer) == False and modalAction == 'track'): # LET MOUSE GO ALWAYS
            
            drag_threshold = distance_between_shoulders*0.2
            # Mouse displacement y-axis correction
            # y_disp = indexY-wristY
            y_disp = 0

            grab_distance = np.linalg.norm(np.array((thumbX,thumbY))-np.array((indexX,indexY)))
            if grab_distance > drag_threshold or handChopOrientation == True: # adjust
                modalMouseButtonList.append('up')
                if mouse_down == False:
                    # pyautogui.moveTo(wristX, wristY+y_disp) # Move cursor
                    cursorX = wristX
                    cursorY = wristY
            elif grab_distance < drag_threshold and modalAction == 'track':
                modalMouseButtonList.append('down')
                if mouse_down:
                    # pyautogui.moveTo(wristX, wristY+y_disp) # Move cursor, not drag to, because we are already holding the mouse down 
                    cursorX = wristX
                    cursorY = wristY
            if cursorX is not None and cursorX > 0:
                currentCoords = [cursorX, cursorY]
                cursorDifference = math.sqrt( ((currentCoords[0]-lastCoords[0])**2)+((currentCoords[1]-lastCoords[1])**2) )
                cursorAccel = cursorDifference
                print(cursorAccel)
                if cursorAccel > 15:
                    v1 = Vector2(lastCoords[0], lastCoords[1])
                    v2 = Vector2(cursorX, cursorY)
                    print ('v1 : '+v1)
                    print ('v2 : '+v2)
                    print(lerp(v1,v2,1))
                    v3 = lerp(v1, v2, totalTime*.1)
                    pyautogui.moveTo(v3.x,v3.y, .1)
                    lastCoords = currentCoords

                #sonification.sendOSCMessage('track', cursorX)
                #sonification.sendXAccelerationOSC(cursorAccel)
            # else: mouse_down = False
    except Exception as e:
        #print(e)
        modalMouseButtonList.append('up')

def zoomInCheck(results):

    distance_between_shoulderR_wristR = np.linalg.norm(np.array((results.right_hand_landmarks.landmark[0].x*screenWidth,results.right_hand_landmarks.landmark[0].y*screenHeight))-np.array((results.pose_landmarks.landmark[12].x*screenWidth,results.pose_landmarks.landmark[12].y * screenHeight)))
    distance_between_shoulderL_wristL = np.linalg.norm(np.array((results.left_hand_landmarks.landmark[0].x * screenWidth,results.left_hand_landmarks.landmark[0].y * screenHeight))-np.array((results.pose_landmarks.landmark[11].x * screenWidth,results.pose_landmarks.landmark[11].y * screenHeight)))
    if (action == 'zoomIn'):
        if (distance_between_shoulderR_wristR<(distance_between_shoulders*0.75)) and (distance_between_shoulderL_wristL<(distance_between_shoulders*0.75)):
            return 'zoomIn' 
        else:
            return 'track' # correct for unintended zoom in



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
            distance_between_shoulders = getScaler(results)
        except Exception as e:
            pass
            
        ## Determine handedness (L vs. R) ##
        try:
            which_hand = determineHandedness(results)
        except Exception as e:
            pass
        
        ## Establish additional constraint for zoomin ##
        try:
            action = zoomInCheck(results)
        except Exception as e:
            pass



        ## LOG Actions ##
      
        modalActionList.append(action)
        modalAction = mode(modalActionList)
        #print ('modal action is ' + modalAction)

        if (action == None):
            action = 'neutral' # modal approach means the buffer should always be filled with something so dont continue
        else:
            execute = True
            action_to_execute = modalAction
            lmColor = lmColor_map[modalAction]
            connColor = connColor_map[modalAction]
            gesture_text = gesture_map_text[modalAction]
 
        ## EXECUTE Actions ##
        if execute == True:
            executeAction(action,action_to_execute,zoomLevel)
        else:
            gesture_text = ''
            if (trackingActive == True): 
                trackingActive = False
        if lastExecutedAction != action_to_execute:
            print ('pose changed')
            print ('modal action is ' + modalAction)
        lastExecutedAction = action_to_execute
                
        ## Move and Drag Mouse ##
        try:
            moveAndDragMouse(results,which_hand)
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