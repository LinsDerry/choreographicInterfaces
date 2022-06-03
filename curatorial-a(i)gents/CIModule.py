"""
Choreograhic Interface Module
By: Lins Derry, Jordan Kruguer, Maximilian Mueller 
Metalab @ Harvard
"""

import cv2
import mediapipe as mp
import math
import csv
import os
import pandas as pd
import numpy as np
import collections
import poseEmbedding # TODO - include here in this module maybe
import pickle
from shutil import which
from mediapipe.python.solutions import pose as mp_pose
import pyautogui, sys
import time
import keyboard
#import sonification #CI #sonification Module - Author(s): Maximilian Mueller (UNCOMMENT FOR USE)

def printsomething(string):
    print(string)
## THOUGHT --> dictionary to add elements/landmarks we care about
class ChoreographicInterface:
    """
    Estimates pose, hand, and face points on a human body using the mediapipe library.
    Uses landmarks to produce choreographic interface gestures and actions. 
    """

    def __init__(self,detectionCon=0.5,trackingCon=0.5,scaleBar=0.0,self.holisticLandmarks=None,
                 whichHand='right',mouseDown=False,resetHand=False,maxDistanceWristIndex=0.0,modalMouseButtonList=collections.deque(['up']*10,maxlen=10),modalMouseState='up',clickBuffer=collections.deque([False]*10,maxlen=10),handChopOrientation=False,
                 action='neutral',execute=False,lastExecutedAction='neutral',modalActionList=collections.deque(['neutral']*10,maxlen=10),modalAction='neutral',missingLandmarkBuffer=collections.deque([False]* 10, maxlen=10),logReg=None,
                 landmarksOfInterest={}):

        """
        :params CIDetector
        :param min_detection_confidence: TODO-DESCRIBE
        :param min_tracking_confidence: TODO-DESCRIBE
        :param scaleBar: Distance between shoulders used as scale bar
        """

        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.scaleBar = scaleBar

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHolsitic = mp.solutions.holistic
        self.holistic = self.mpHolsitic.Holistic(min_detection_confidence=self.detectionCon,
                                                 min_tracking_confidence=self.trackingCon)
        self.holisticLandmarks = holisticLandmarks

        """
        :params CIHands
        :param : TODO-DESCRIBE
        """
        
        self.whichHand = whichHand
        self.mouseDown = mouseDown
        self.resetHand = resetHand
        self.maxDistanceWristIndex = maxDistanceWristIndex
        self.modalMouseButtonList = modalMouseButtonList
        self.modalMouseState = modalMouseState
        self.clickBuffer = clickBuffer
        self.handChopOrientation = handChopOrientation

        """
        :params CIPose
        :param : TODO-DESCRIBE
        """
        self.action = action
        self.execute = execute
        self.lastExecutedAction = lastExecutedAction
        self.modalActionList = modalActionList
        self.modalAction = modalAction
        self.missingLandmarkBuffer = missingLandmarkBuffer
        self.logReg = logReg

        """
        :params CIPose
        :param : TODO-DESCRIBE
        """
        self.landmarksOfInterest

    def findHolisticLandmarks(self, img):
        """
        Process image and find holistic landmarks in an Image.
        :param img: Image to find the holistic landmarks in.
        :return: Processed image and holistic landmarks
        """
        img = cv2.cvtColor(cv2.flip(img,1),cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = self.holistic.process(img) 
        self.holisticLandmarks = results
        img.flags.writeable = True 
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return img 
    
    def updateLandmarksOfInterest(self):
        # TODO FINISH
        results = self.holisticLandmarks
        if results is not None:
            try:
                self.landmarksOfInterest['rightWrist'] = results.left_hand_landmarks.landmark[0]
                self.landmarksOfInterest['leftWrist'] = results.right_hand_landmarks.landmark[0]
                self.missingLandmarkBuffer.append(False)
            except Exception as e:
                self.missingLandmarkBuffer.append(True)
                self.clickBuffer.append(False)
            self.landmarksOfInterest['rightThumb'] = results.left_hand_landmarks.landmark[2]
            self.landmarksOfInterest['leftThumb'] = results.right_hand_landmarks.landmark[2]
            self.landmarksOfInterest['rightIndex'] = results.left_hand_landmarks.landmark[8]
            self.landmarksOfInterest['leftIndex'] = results.right_hand_landmarks.landmark[8]
            self.landmarksOfInterest['rightShoulder'] = results.pose_landmarks.landmark[12]
            self.landmarksOfInterest['leftShoulder'] = results.pose_landmarks.landmark[11]


    def updateScaleBar(self,screenWidth,screenHeight):
        results = self.holisticLandmarks
        if results is not None:
            rs_x = self.landmarksOfInterest['rightShoulder'].x * screenWidth 
            rs_y = self.landmarksOfInterest['rightShoulder'].y * screenHeight 
            ls_x = self.landmarksOfInterest['leftShoulder'].x * screenWidth 
            ls_y = self.landmarksOfInterest['leftShoulder'].y * screenHeight 
            self.scaleBar = np.linalg.norm(np.array((rs_x,rs_y))-np.array((ls_x,ls_y)))

    def determineHandedness(self,screenWidth,screenHeight):
        
        results = self.holisticLandmarks
        if results is not None:

            fc_x = results.pose_landmarks.landmark[0].x * screenWidth 
            fc_y = results.pose_landmarks.landmark[0].y * screenHeight 
            re_x = results.pose_landmarks.landmark[8].x * screenWidth 
            re_y = results.pose_landmarks.landmark[8].y * screenHeight 
            le_x = results.pose_landmarks.landmark[7].x * screenWidth 
            le_y = results.pose_landmarks.landmark[7].y * screenHeight 

            # TODO make this abstracted so we don't have to repeat this
            rs_x = results.pose_landmarks.landmark[12].x * screenWidth 
            rs_y = results.pose_landmarks.landmark[12].y * screenHeight 
            ls_x = results.pose_landmarks.landmark[11].x * screenWidth 
            ls_y = results.pose_landmarks.landmark[11].y * screenHeight

            dist1 = np.linalg.norm(np.array((fc_x,fc_y))-np.array((rs_x,rs_y)))
            dist2 = np.linalg.norm(np.array((fc_x,fc_y))-np.array((ls_x,ls_y)))
    
            if (dist2>dist1) and (dist1<(self.scaleBar*0.65)):
                self.whichHand = 'left' # mirrored
            elif (dist2<dist1) and (dist2<(self.scaleBar*0.65)):
                self.whichHand = 'right' # mirrored
    
    def determineHandOrientation(self,screenWidth,screenHeight):
        
        results = self.holisticLandmarks
        if results is not None:

            pr_x = results.left_hand_landmarks.landmark[20].x * screenWidth
            pr_y = results.left_hand_landmarks.landmark[20].y * screenHeight 
            ttr_x = results.left_hand_landmarks.landmark[4].x * screenWidth
            ttr_y = results.left_hand_landmarks.landmark[4].y * screenHeight
     
            pl_x = results.right_hand_landmarks.landmark[20].x * screenWidth 
            pl_y = results.right_hand_landmarks.landmark[20].y * screenHeight 
            ttl_x = results.right_hand_landmarks.landmark[4].x * screenWidth  
            ttl_y = results.right_hand_landmarks.landmark[4].y * screenHeight 
       
            if self.action == 'track':
                if (self.whichHand == 'right'):
                    handSpan = np.linalg.norm(np.array((pr_x,pr_y))-np.array((ttr_x,ttr_y)))
                else:
                    handSpan = np.linalg.norm(np.array((pl_x,pl_y))-np.array((ttl_x,ttl_y)))
                # TODO Review this logic
                normalizedHandSpan = handSpan / self.scaleBar
                if normalizedHandSpan < .2:
                    self.handChopOrientation = True 
                else: 
                    self.handChopOrientation = False

    def initPoseClasssifier(self,fileName,setMaps):
        with open(fileName, 'rb') as file:
            self.logReg = pickle.load(file)
        return setMaps[fileName]

    def classifyPose(self,img,classMap,gestureMap):
        outputFrame = img.copy() # maybe delete
        poseLandmarks = self.holisticLandmarks.pose_landmarks
        if poseLandmarks is not None:
            frameHeight, frameWidth = outputFrame.shape[0], outputFrame.shape[1]
            poseLandmarks = np.array([[lmk.x*frameWidth,lmk.y*frameHeight,lmk.z*frameWidth] for lmk in poseLandmarks.landmark], dtype=np.float32)
            assert poseLandmarks.shape == (33, 3), 'Unexpected pose landmarks shape: {}'.format(poseLandmarks.shape)
            embed = poseEmbedding.FullBodyPoseEmbedder() # TODO change
            poseLandmarkEmbedding = embed(poseLandmarks)
            poseLandmarkEmbeddingFlattened = poseLandmarkEmbedding.flatten()
            classification = self.logReg.predict(poseLandmarkEmbeddingFlattened.astype(float).reshape(1,-1))
            gesture = class_map[classification[0]]
            self.action = gesture_map[gesture]


    def logAction(self):

        if mode(self.missingLandmarkBuffer) == True or self.action == None:
            self.action = 'neutral' # TODO check this
        if self.action == 'track':
            if self.handChopOrientation == True:
                self.action = 'neutral' # TODO check this
        self.modalActionList.append(self.action)
        # modalAction = mode(modalActionList)
        self.modalAction = mode(self.modalActionList)
        return self.modalAction # TODO review

    def executeAction(self):
        if (self.modalAction == 'refresh' and self.lastExecutedAction != 'refresh'):
            pyautogui.hotkey('command', 'r')
            # pyautogui.hotkey('ctrl', 'r') #for LB
            zoomLevel = 0 
        elif (self.modalAction == 'zoomIn'):
            #pyautogui.hotkey('command', '+')
            pyautogui.scroll(5)
            if zoomLevel > -10:
                zoomLevel -= totalTime
                audioParam = zoomLevel 
        elif (self.modalAction == 'zoomOut'):
            #pyautogui.hotkey('command', '-')
            pyautogui.scroll(-5)
            if zoomLevel < 10:
                zoomLevel += totalTime
                audioParam = zoomLevel 
        elif (self.modalAction == 'scrollUp'):
            pyautogui.scroll(5)
            # pyautogui.scroll(50) # for LB
            # pyautogui.press('right') #Giulia
        elif (self.modalAction == 'scrollDown'):
            pyautogui.scroll(-5)
            # pyautogui.scroll(50) # for LB
            # pyautogui.press('left') #Giulia

    # TODO FINISH
    def mouseSelect(self):
        
        if self.modalAction == 'track':
            select_threshold = self.scaleBar*0.15
            if self.whichHand == 'right':
                select_distance = np.linalg.norm(np.array((x_indexL,y_indexL))-np.array((x_wristR,y_wristR)))
            if self.whichHand == 'left':
                select_distance = np.linalg.norm(np.array((x_indexR,y_indexR))-np.array((x_wristL,y_wristL)))
            
            if select_distance < select_threshold:
                # print('select True added to buffer')
                self.clickBuffer.append(True)
            elif select_distance > select_threshold:
                self.clickBuffer.append(False)
    
            if mode(self.clickBuffer) == True:
                if click == False:
                    click = True
                    pyautogui.click(button='left')
                    #sonification.sendOSCMessage('select','')
                    print('clicked!')
            elif click == True:
                click = False
                #sonification.sendOSCMessage('deselect','')


    def moveDrafMouse(self):
       
        # Vitruvian adjustment #
        armLength = self.scaleBar*2

    




