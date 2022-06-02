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
import poseEmbedding # TODO - include here in this module


## THOUGHT --> dictionary to add elements/landmarks we care about
class CIDetector:
    """
    Estimates pose, hand, and face points on a human body using the mediapipe library.
    """

    def __init__(self,detectionCon=0.5,trackingCon=0.5,scaleBar=0.0,self.holisticLandmarks=None):

        """
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
        return self.holisticLandmarks, img # maybe not return holistic landmarks
    
    def updateScaleBar(self,screenWidth,screenHeight):
        results = self.holisticLandmarks
        if results is not None:
            rs_x = results.pose_landmarks.landmark[12].x * screenWidth 
            rs_y = results.pose_landmarks.landmark[12].y * screenHeight 
            ls_x = results.pose_landmarks.landmark[11].x * screenWidth 
            ls_y = results.pose_landmarks.landmark[11].y * screenHeight 
            self.scaleBar = np.linalg.norm(np.array((rs_x,rs_y))-np.array((ls_x,ls_y)))

    

class CIHands(CIDetector):
    """
    Estimates hand points on human body - inherits from CIDetector
    """

    def __init__(self,whichHand='right',mouseDown=False,resetHand=False,maxDistanceWristIndex=0.0,modalMouseButtonList=collections.deque(['up']*10,maxlen=10),modalMouseState='up',clickBuffer=collections.deque([False]*10,maxlen=10),handChopOrientation=False):

       
        """
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

    def determineHandedness(self,screenWidth,screenHeight):
        
        results = self.holisticLandmarks
        if results is not None:

            fc_x x_face_center = results.pose_landmarks.landmark[0].x * screenWidth 
            fc_y y_face_center = results.pose_landmarks.landmark[0].y * screenHeight 
            re_x x_earR = results.pose_landmarks.landmark[8].x * screenWidth 
            re_y y_earR = results.pose_landmarks.landmark[8].y * screenHeight 
            le_x x_earL = results.pose_landmarks.landmark[7].x * screenWidth 
            le_y y_earL = results.pose_landmarks.landmark[7].y * screenHeight 

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
    
    def determineHandOrientation(self,action,screenWidth,screenHeight):
        
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
       
            if action == 'track':
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

class CIPose(CIDetector):
    """
    Estimates pose points on human body - inherits from CIDetector
    """

    def __init__(self,action='neutral',execute=False,lastExecutedAction='neutral',modalActionList=collections.deque(['neutral']*10,maxlen=10),modalAction='neutral',missingLandmarkBuffer=collections.deque([False]* 10, maxlen=10),logReg=None): 
        
        """
        :param : TODO-DESCRIBE
        """

        self.action = action
        self.execute = execute
        self.lastExecutedAction = lastExecutedAction
        self.modalActionList = modalActionList
        self.modalAction = modalAction
        self.missingLandmarkBuffer = missingLandmarkBuffer
        self.logReg = logReg

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
            action = gesture_map[gesture]
            return action


    def logAction(self,action):

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

        

    












