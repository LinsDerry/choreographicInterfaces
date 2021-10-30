import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import time

import pyautogui, sys
import numpy as np
import math

import os
import pose_module # RENAME
import sonification
import threading#I may have added this - if so feel free to remove
import _thread#I may have added this - if so feel free to remove
import random
t0 = time.time()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

lav = (187, 160, 178) # RGB for mp_drawing
pink = (172, 18, 143) # RGB for mp_drawing
red = (58, 45, 240) # BGR for cv2 text
white = (0, 0, 0)
black = (0, 0, 0)

# Folder with pose class CSVs. 
pose_samples_folder = 'poses_csvs_out'

# Initialize embedder.
pose_embedder = pose_module.FullBodyPoseEmbedder()

# Initialize classifier.
pose_classifier = pose_module.PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

cap = cv2.VideoCapture(0)

t1 = time.time()
print(f"Took {t1-t0} seconds for setup")

action = 'neutral'
verbosePrinting = False
previousAction = 0
actionCount = 0
cursorX = 0
prevCursorX = 0
def lerp(v, d):
    return v[0] * (1 - d) + v[1] * d


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        

        t0 = time.time()
        success, image = cap.read()
        
        # if success is False:#trying to get rid of NoneType object has no attribute landmark error
        #     print('image is null')
        #     continue
        
        start = time.time()


        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False

        # Process the image and detect the holistic
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        t1 = time.time()
        if verbosePrinting:
            print(f"Took {t1-t0} seconds to get frame and classifiy using holistic model")
        
        #----------------#
        # DRAW LANDMARKS #
        #----------------#
        lmColor = pink
        connColor = lav
        rad = 9
        thick = 4

        # t0 = time.time()
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=(rad // 2)),
                                 mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=(rad // 2)),
                                 mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),
                                 mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
        # t1 = time.time()
        # print(f"Took {t1-t0} seconds to draw landmarks")
        

        #---------------#
        # CLASSIFY POSE #
        #---------------#
        t0 = time.time()
        output_frame = image.copy()
        pose_landmarks = results.pose_landmarks
        if pose_landmarks is not None:
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
            t1 = time.time()
            if verbosePrinting:
                print(f"Took {t1-t0} seconds to format the pose landmarks")

            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)
            action = max(pose_classification, key=pose_classification.get)
            # cv2.putText(image, f'Pose: {action}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2) # provide feedback on current pose
            t1 = time.time()
            if verbosePrinting:
                print(f"Took {t1-t0} seconds to classify the pose")


        #-----------------#
        # RETRIEVE LOI(S) #
        #-----------------#

        


        # ##########################
        # ## SANDBOX: EDIT BELOW ##
        # ##########################

        # # getting screen's height in pixels, getting screen's width in pixels 
        screenWidth, screenHeight = pyautogui.size()

        textColor = white

        try:
            # Anchor on right index finger --> landmark 8
            x = results.left_hand_landmarks.landmark[8].x * screenWidth # scaled
            y = results.left_hand_landmarks.landmark[8].y * screenHeight # scaled
            z = results.left_hand_landmarks.landmark[8].z

            # Move cursor
            pyautogui.moveTo(x, y)
            if x is not None:

                cursorX = x
                cursorAccel = abs(cursorX - prevCursorX)
                prevCursorX = cursorX
                #print (cursorAccel)
                sonification.sendXAccelerationOSC(cursorAccel)
                sonification.sendOSCMessage('track', x)
                #cursorParam = str(cursorX) + ' ' + str(cursorAccel)


            # if action == 'track':
            #     print (x)
            #     sonification.sendOSCMessage(action, x)

        except Exception as e:
            if verbosePrinting:
                print(e)


         # TEMPLATE START
        try:
            if (action=='refresh') and (pose_classification[action] == 10): #  Strong Confidence:
                #pyautogui.PAUSE = 2
                #print('REFRESH')
                pyautogui.hotkey('command', 'r') # refresh
                #pyautogui.scroll(-5) #scroll down
                #pyautogui.PAUSE = 0
        except Exception as e:
            if verbosePrinting:
                print(e)
         #TEMPLATE END

#         try:
#             if (action=='select') and (pose_classification[action] == 10): #  Stron Confidence:
#                 print('SELECT')
#                 #pyautogui.PAUSE = 2
#                 pyautogui.click(button='left') #click
#                 #pyautogui.PAUSE = 0
#         except Exception as e:
#             print(e)

        # try:
        #     if (action=='zoomIn') and (pose_classification[action] == 10): #& pose_classification[action] == 10: #  Strong Confidence:
        #         #pyautogui.PAUSE = 2
        #         pyautogui.hotkey('command', '+') #zoom in
        #         #pyautogui.PAUSE = 0
        # except Exception as e:
        #     print(e)

        # try:
        #     if (action=='zoomOut') and (pose_classification[action] == 10): #& pose_classification[action] == 10: #  Strong Confidence:
        #         #pyautogui.PAUSE = 2
        #         pyautogui.hotkey('command', '-') #zoom out
        #         #pyautogui.PAUSE = 0
        # except Exception as e:
        #     print(e)
             
        try:
            if (action=='scrollUp') and (pose_classification[action] == 10): #  Stron Confidence:
                #pyautogui.PAUSE = 2
                pyautogui.scroll(5)
                #pyautogui.PAUSE = 0
        except Exception as e:
            print(e)
             
        try:
            if (action=='scrollDn') and (pose_classification[action] == 10): #  Stron Confidence:
                #pyautogui.PAUSE = 2
                pyautogui.scroll(-5)
                #pyautogui.PAUSE = 0
        except Exception as e:
            print(e)

        
        try:
            # Use pinky and thumb as select --> landmark 20, landmark 4
            x_thumb = results.right_hand_landmarks.landmark[4].x * screenWidth # scaled
            y_thumb = results.right_hand_landmarks.landmark[4].y * screenHeight # scaled

            x_pinky = results.right_hand_landmarks.landmark[20].x * screenWidth # scaled
            y_pinky = results.right_hand_landmarks.landmark[20].y * screenHeight # scaled

            distance = ((((x_thumb - x_pinky)**2)+((y_thumb-y_pinky)**2))**0.5)

            #print(f"Distance between pinky and thumb: {distance}")
            if distance<20: # adjust
                #pyautogui.click(button='left')
                pyautogui.mouseDown()
                cv2.putText(image, 'select', (25,140), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2)

            elif distance>20:
                pyautogui.mouseUp()
                cv2.putText(image, 'unSelect', (25,140), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2)
                

        except Exception as e:
            if verbosePrinting:
                print(e)

    
        # try:
        #     # Use wrist-z as zoom
        #     wrist = results.right_hand_landmarks.landmark[0].z * screenWidth # scaled
    
        #     if wrist < 0:
        #         pyautogui.hotkey('command', '-')
        #         cv2.putText(image, f'Command: Zoom In', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        #     elif wrist > 0:
        #         pyautogui.hotkey('command', '+')
        #         cv2.putText(image, f'Command: Zoom Out', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        # except Exception as e:
        #     print(e)
        # '''

        # #################
        # ## SANDBOX END ##
        # #################


        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        if verbosePrinting:
            print("FPS: ", fps)

        action_stack = fps

        # Feedback in black & white
        #image = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY)

        # Get width and height of camera capture
        capWidth  = cap.get(3) 
        capHeight = cap.get(4)

        # Scale feedback
        scale = 0.5
        newCapWidth = int(screenWidth * (scale))
        newCapHeight = int(newCapWidth * capHeight / capWidth)

        # Position feedback bottom right corner (nuudged (5, 33) to look just right on MacBook Air)
        capX = screenWidth - newCapWidth - 5
        capY = screenHeight - newCapHeight - 33

        #counter-based pose switch detector for denoising; untested
        if action == previousAction:
            actionCount += 1
            previousAction = action
            if verbosePrinting:
                print (actionCount, action)
            if actionCount == 3 and action != 'track':
                sonification.sendOSCMessage(action, '')
                sonification.sendOSCMessage('changed', '')
                #_thread.start_new_thread(sonification.playPoseSound,(action,))#starting a coroutine so that playing sound doesnt hold up execution
                print ('pose switched! to ' + action)
            elif actionCount > 3:
                if action == 'zoomOut':
                    zoomOutTime += totalTime
                    audioParam = zoomOutTime 
                    sonification.sendOSCMessage(action, audioParam)
                elif action == 'zoomIn':
                    zoomInTime += totalTime
                    audioParam = zoomInTime 
                    sonification.sendOSCMessage(action, audioParam)
        else:
            actionCount = 0
            previousAction = action
            zoomOutTime = 0
            zoomInTime = 0

        #original working pose switch 
        # if action != previousAction:
        #     previousAction = action
        #     sonification.playPoseSound(action)
        #     print ('pose switched! to ' + action)
        
        cv2.putText(image, f'FPS: {int(fps)}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2) # provide feedback on current FPS
        cv2.putText(image, f'{action}', (25,65), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2) # provide feedback on current pose
        cv2.imshow('Choreographic Interface', cv2.resize(image, (newCapWidth, newCapHeight)))
        cv2.setWindowProperty('Choreographic Interface', cv2.WND_PROP_TOPMOST, 1) # keeps feedback window most front
        cv2.moveWindow('Choreographic Interface', capX,capY) # relocate feedback

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

