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
