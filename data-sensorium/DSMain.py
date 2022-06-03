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
set_maps = {'DS':{1:'cycle',2:'drought',3:'earthquake',4:'flood',5:'movement',6:'neutral',7:'storm',8:'volcano',9:'wildfire',10:'xTemp'}} 

classifiers = {'DS':'032522_LogReg_pose_classifier_DataSensorium.pkl'} # New set with universal 'track

# Set maps for feedback
#gesture_map_text = {'refresh':'REFRESH','zoomIn':'ZOOM IN','zoomOut':'ZOOM OUT','scrollUp':'SCROLL UP','scrollDown':'SCROLL DOWN','track':'TRACK','neutral':'NEUTRAL'} 
#lmColor_map = {'refresh':blue,'zoomIn':turquoise,'zoomOut':fuschia,'scrollUp':green,'scrollDown':maroon,'track':turquoise,'neutral':white}
#connColor_map = {'refresh':white,'zoomIn':fuschia,'zoomOut':turquoise,'scrollUp':maroon,'scrollDown':green,'track':lav,'neutral':blue}

## Set/Map selection ##
selection = 'DS'
class_map = set_maps[str(selection)]
pkl_filename = classifiers[str(selection)]
with open(pkl_filename, 'rb') as file:
    logisticRegr = pickle.load(file)


#declaring these to avoid error on start
action = 'neutral' 

## audio variable declaration
currentCoords = [0,0]
lastCoords = [0,0]
zoomLevel = 0
cursorX = 0
cursorY = 0
totalTime = .01

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

def lerp(v1, v2, time): #time is value between 0 and 1
    return v1 * (1- time) + v2 * time
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
            action = gesture


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
        except Exception as e:
            pass
        try:
            y_wristR = results.left_hand_landmarks.landmark[0].y * screenHeight # --> Right Wrist (y-coor)
        except:
            pass
        try:
            x_wristL = results.right_hand_landmarks.landmark[0].x * screenWidth # --> Left Wrist (x-coor)
        except:
            pass
        try:
            y_wristL = results.right_hand_landmarks.landmark[0].y * screenHeight # --> Left Wrist (y-coor)
        except:
            pass




        
   
        #lmColor = lmColor_map[action]
        #connColor = connColor_map[action]
        gesture_text = action #gesture_map_text[action]
        #for the text, when making track active always only set to track if nothing else
    

        ## Determine FPS ##     
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        ## Format User Feedback ... Position feedback bottom right corner (nudged (5, 33) to look just right on MacBook Air) ##
        rad = 4
        thick = 3

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
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