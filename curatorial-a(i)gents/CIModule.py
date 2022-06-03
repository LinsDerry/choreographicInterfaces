"""
Choreograhic Interface Module
By: Lins Derry, Jordan Kruguer, Maximilian Mueller 
Metalab @ Harvard
"""

## TODO - FIX DEQUE --> 

import cv2
import mediapipe as mp
import math
import csv
import os
import numpy as np
import collections
import poseEmbedding # TODO - include here in this module maybe
import pickle
from shutil import which
from mediapipe.python.solutions import pose as mp_pose
import pyautogui, sys
import time
import keyboard
from scipy.stats import mode
import sonification #CI #sonification Module - Author(s): Maximilian Mueller (UNCOMMENT FOR USE)

# TODO - fold this into CI class 
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

class ChoreographicInterface:
    """
    Estimates pose, hand, and face points on a human body using the mediapipe library.
    Uses landmarks to produce choreographic interface gestures and actions. 
    """

    def __init__(self,detectionCon=0.5,trackingCon=0.5,scaleBar=0.0,holisticLandmarks=None,
                currentCoords = np.nan, lastCoords = [1,1],
                 whichHand='right',mouseDown=False,resetHand=False,maxDistanceWristIndex=0.0,modalMouseButtonList=collections.deque(['up']*5,maxlen=5),modalMouseState='up',clickBuffer=collections.deque([False]*10,maxlen=10),handChopOrientation=False,
                 action='neutral',execute=False,lastExecutedAction='track',modalActionList=collections.deque(['neutral']*10,maxlen=10),modalAction='neutral',missingLandmarkBuffer=collections.deque([False]* 10, maxlen=10),logReg=None,
                 landmarksOfInterest={'rightWrist':np.nan,'leftWrist':np.nan,'rightThumb':np.nan,'leftThumb':np.nan,'rightIndex':np.nan,'leftIndex':np.nan,'rightShoulder':np.nan,'leftShoulder':np.nan,'faceCenter':np.nan,'rightEar':np.nan,'leftEar':np.nan}):

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
        self.landmarksOfInterest = landmarksOfInterest

        self.currentCoords = currentCoords
        self.lastCoords = lastCoords

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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        return img 
    
    def updateLandmarksOfInterest(self):
        # TODO FINISH
        results = self.holisticLandmarks
        if results is not None:
            try:
                self.landmarksOfInterest['rightWrist'] = results.left_hand_landmarks.landmark[0]
                if self.whichHand == 'right':
                    self.missingLandmarkBuffer.append(False)
            except Exception as e:
                if self.whichHand == 'left':
                    self.missingLandmarkBuffer.append(True)
                    self.clickBuffer.append(False)
            try:
                self.landmarksOfInterest['leftWrist'] = results.right_hand_landmarks.landmark[0]
                if self.whichHand == 'left':
                    self.missingLandmarkBuffer.append(False)
            except Exception as e:
                if self.whichHand == 'left':
                    self.missingLandmarkBuffer.append(True)
                    self.clickBuffer.append(False)
            try:
                self.landmarksOfInterest['rightThumb'] = results.left_hand_landmarks.landmark[2]
            except Exception as e:
                pass
            try:
                self.landmarksOfInterest['leftThumb'] = results.right_hand_landmarks.landmark[2]
            except Exception as e:
                pass
            try:
                self.landmarksOfInterest['rightIndex'] = results.left_hand_landmarks.landmark[8]
            except Exception as e:
                pass
            try:
                self.landmarksOfInterest['leftIndex'] = results.right_hand_landmarks.landmark[8]
            except Exception as e:
                pass
            try:
                self.landmarksOfInterest['rightShoulder'] = results.pose_landmarks.landmark[12]
            except Exception as e:
                pass
            try:
                self.landmarksOfInterest['leftShoulder'] = results.pose_landmarks.landmark[11]
            except Exception as e:
                pass
            try:
                self.landmarksOfInterest['faceCenter'] = results.pose_landmarks.landmark[0]
            except Exception as e:
                pass
            try:
                self.landmarksOfInterest['rightEar'] = results.pose_landmarks.landmark[8]
            except Exception as e:
                pass
            try:
                self.landmarksOfInterest['leftEar'] = results.pose_landmarks.landmark[7]
            except Exception as e:
                pass

    def updateScaleBar(self,screenWidth,screenHeight):
        results = self.holisticLandmarks
        if results is not None:
            try:
                rs_x = self.landmarksOfInterest['rightShoulder'].x * screenWidth 
                rs_y = self.landmarksOfInterest['rightShoulder'].y * screenHeight 
                ls_x = self.landmarksOfInterest['leftShoulder'].x * screenWidth 
                ls_y = self.landmarksOfInterest['leftShoulder'].y * screenHeight 
                self.scaleBar = np.linalg.norm(np.array((rs_x,rs_y))-np.array((ls_x,ls_y)))
            except Exception as e:
                pass

    def determineHandedness(self,screenWidth,screenHeight):
        
        results = self.holisticLandmarks
        if results is not None:
            try:
                fc_x = self.landmarksOfInterest['faceCenter'].x * screenWidth 
                fc_y = self.landmarksOfInterest['faceCenter'].y * screenHeight 
                re_x = self.landmarksOfInterest['rightEar'].x * screenWidth 
                re_y = self.landmarksOfInterest['rightEar'].y * screenHeight 
                le_x = self.landmarksOfInterest['leftEar'].x * screenWidth 
                le_y = self.landmarksOfInterest['leftEar'].y * screenHeight 

                # TODO make this abstracted so we don't have to repeat this
                rs_x = self.landmarksOfInterest['rightShoulder'].x * screenWidth 
                rs_y = self.landmarksOfInterest['rightShoulder'].y * screenHeight 
                ls_x = self.landmarksOfInterest['leftShoulder'].x * screenWidth 
                ls_y = self.landmarksOfInterest['leftShoulder'].y * screenHeight

                dist1 = np.linalg.norm(np.array((fc_x,fc_y))-np.array((rs_x,rs_y)))
                dist2 = np.linalg.norm(np.array((fc_x,fc_y))-np.array((ls_x,ls_y)))
        
                if (dist2>dist1) and (dist1<(self.scaleBar*0.65)):
                    self.whichHand = 'left' # mirrored
                elif (dist2<dist1) and (dist2<(self.scaleBar*0.65)):
                    self.whichHand = 'right' # mirrored
            except Exception as e:
                pass
    
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

    def initPoseClasssifier(self,fileName):
        with open(fileName, 'rb') as file:
            self.logReg = pickle.load(file)

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
            gesture = classMap[classification[0]]
            self.action = gestureMap[gesture]
            return self.action

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

    def executeAction(self): # TODO: check why these are not returning values. 
        try:
          
            if (self.modalAction == 'refresh' and self.lastExecutedAction != 'refresh'):
                #pyautogui.hotkey('command', 'r')
                pyautogui.hotkey('ctrl', 'r') #for LB
                self.modalAction[0][0] = 'REFRESHED WORKED'
                sonification.sendOSCMessage('refresh','')
                return 'REFRESH!'
            elif (self.modalAction[0][0] == 'zoomIn'):
                #pyautogui.hotkey('command', '+')
                pyautogui.scroll(5)
                sonification.sendOSCMessage('zoomIn','')
                return 'ZOOMIN!'
            elif (self.modalAction[0][0] == 'zoomOut'):
                #pyautogui.hotkey('command', '-')
                pyautogui.scroll(-5)
                sonification.sendOSCMessage('zoomOut','')
                return 'ZOOMOUT!'
            elif (self.modalAction[0][0] == 'scrollUp'):
                pyautogui.scroll(5)
                # pyautogui.scroll(50) # for LB
                # pyautogui.press('right') #Giulia
                sonification.sendOSCMessage('scrollUp','')
                return 'SCROLLUP!'
            elif (self.modalAction[0][0] == 'scrollDown'):
                pyautogui.scroll(-5)
                # pyautogui.scroll(50) # for LB
                # pyautogui.press('left') #Giulia
                sonification.sendOSCMessage('scrollDown','')
                return 'SCROLLDOWN!'
            # print(f"{} = {}")
            if self.modalAction[0][0] != self.lastExecutedAction:
                sonification.sendOSCMessage('changed', '')
                print('ACTION CHANGED')

                if (self.modalAction[0][0] == 'track'):
                    sonification.sendOSCMessage('trackStart','')

            self.lastExecutedAction = self.modalAction[0][0]

        except Exception as e:
            return e

    # TODO FINISH
    def mouseSelect(self): # currently not in use...
        
        if self.modalAction == 'track':
            select_threshold = self.scaleBar*0.15
            if self.whichHand == 'right':
                select_distance = np.linalg.norm(np.array((self.landmarksOfInterest['leftIndex'].x * screenWidth ,self.landmarksOfInterest['leftIndex'].y * screenHeight))-np.array((self.landmarksOfInterest['rightWrist'].x * screenWidth ,self.landmarksOfInterest['rightWrist'].y * screenHeight)))
            if self.whichHand == 'left':
                select_distance = np.linalg.norm(np.array((self.landmarksOfInterest['rightIndex'].x * screenWidth ,self.landmarksOfInterest['rightIndex'].y * screenHeight))-np.array((self.landmarksOfInterest['leftWrist'].x * screenWidth ,self.landmarksOfInterest['leftWrist'].y * screenHeight)))
            if select_distance < select_threshold:
                self.clickBuffer.append(True)
            elif select_distance > select_threshold:
                self.clickBuffer.append(False)

            if mode(self.clickBuffer) == True: # TODO make sure it doesn't rapid fire
                    pyautogui.click(button='left')
                    #sonification.sendOSCMessage('select','')
                    print('clicked!')
            #elif click == True:
            #    click = False
                #sonification.sendOSCMessage('deselect','')
            return select_threshold, select_distance

    def moveDragMouse(self,screenWidth,screenHeight):

        try:

            
            if (self.modalAction[0][0]=='track'):

                # DRAG AND MOVE 
                
                #armLength = self.scaleBar*2 # Vitruvian adjustment #
                drag_threshold = self.scaleBar*0.15

                # TODO make landmarks of interest a nested dictionary with 'left' and 'right'keys so we dont have to duplicate here
                if self.whichHand == 'right':
                    wristX = self.landmarksOfInterest['rightWrist'].x*screenWidth
                    wristY = self.landmarksOfInterest['rightWrist'].y*screenHeight
                    thumbX = self.landmarksOfInterest['rightThumb'].x*screenWidth
                    thumbY = self.landmarksOfInterest['rightThumb'].y*screenHeight
                    indexX =  self.landmarksOfInterest['rightIndex'].x*screenWidth
                    indexY = self.landmarksOfInterest['rightIndex'].y*screenHeight
                    #shoulderX = self.landmarksOfInterest['rightShoulder'].x * screenWidth
                    #shoulderY = self.landmarksOfInterest['rightShoulder'].y * screenHeight
                else:
                    wristX = self.landmarksOfInterest['leftWrist'].x*screenWidth
                    wristY = self.landmarksOfInterest['leftWrist'].y*screenHeight
                    thumbX = self.landmarksOfInterest['leftThumb'].x*screenWidth
                    thumbY = self.landmarksOfInterest['leftThumb'].y*screenHeight
                    indexX =  self.landmarksOfInterest['leftIndex'].x*screenWidth
                    indexY = self.landmarksOfInterest['leftIndex'].y*screenHeight
                    #shoulderX = self.landmarksOfInterest['leftShoulder'].x * screenWidth
                    #shoulderY = self.landmarksOfInterest['leftShoulder'].y * screenHeight
                
                '''
                # reach logic
                screenDistanceTop = screenHeight-shoulderY # distance to top extreme of screen
                screenDistanceBottom = shoulderY # distance to top extreme of screen
                screenDistanceRight = screenWidth-shoulderX # distance to right extreeme of screen
                screenDistanceLeft = shoulderX # distance to right extreeme of screen
                reachAmountY = (wristY-shoulderY)/armLength # pos if hand above shoulder
                reachAmountX = (wristX-shoulderX)/armLength # pos if hand right of shoulder

                if (reachAmountY>0) and (reachAmountX>0):
                    posY = screenDistanceTop*reachAmountY
                    posX = screenDistanceRight*reachAmountX
                elif (reachAmountY<0) and (reachAmountX>0):
                    posY = screenDistanceTop*-1*reachAmountY
                    posX = screenDistanceRight*reachAmountX
                elif (reachAmountY>0) and (reachAmountX<0):
                    posY = screenDistanceTop*-1*reachAmountY
                    posX = screenDistanceRight*reachAmountX
                '''

                
                
                grab_distance = np.linalg.norm(np.array((thumbX,thumbY))-np.array((indexX,indexY)))
                if grab_distance > drag_threshold: # adjust
                    self.modalMouseButtonList.append('up')
                elif grab_distance < drag_threshold:
                    self.modalMouseButtonList.append('down')

                if mode(self.modalMouseButtonList)[0][0] == 'up': # adjust
                    if self.mouseDown == True:
                        #pyautogui.mouseUp()
                        print('MOUSEUP!')
                        self.mouseDown = False
                        sonification.sendMouseUp()
                    pyautogui.moveTo(wristX, wristY,0) # Move cursor
                elif mode(self.modalMouseButtonList)[0][0] == 'down':
                    if self.mouseDown == False:
                        #pyautogui.mouseDown()
                        pyautogui.click()
                        print('MOUSEDOWN!')
                        self.mouseDown = True
                        sonification.sendMouseDown()
                #pyautogui.moveTo(wristX, wristY)
                    #pyautogui.dragTo(wristX, wristY,0,button='left') # Drag cursor

                if wristX is not None and wristX > 0:
                    self.currentCoords = [wristX, wristY]#change to final
                    cursorDifference = math.sqrt( ((self.currentCoords[0]-self.lastCoords[0])**2)+((self.currentCoords[1]-self.lastCoords[1])**2) )
                    cursorAccel = cursorDifference
                    self.lastCoords = self.currentCoords
                    sonification.sendOSCMessage('track', wristX)
                    sonification.sendXAccelerationOSC(cursorAccel)

                '''
                
                if self.whichHand == 'right':
                    cursorX = self.landmarksOfInterest['rightWrist'].x*screenWidth
                    cursorY = self.landmarksOfInterest['rightWrist'].x*screenHeight
                else:
                    cursorX = self.landmarksOfInterest['leftWrist'].x*screenWidth
                    cursorY = self.landmarksOfInterest['leftWrist'].x*screenHeight
                '''
                
        except Exception as e:
            #print(e)
            pass
        
            


               



        

       