import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import time

import pyautogui, sys
import numpy as np
import math

import os
import module # RENAME


# Multiprocessing
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Pool
import sys
# Multiprocessing 

# Multithreading
from threading import Thread
# Multithreading


def init_pose_classifier(pose_samples_folder):

    # Initialize embedder.
    pose_embedder = module.FullBodyPoseEmbedder()

    # Initialize classifier.
    pose_classifier = module.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    return pose_classifier


def classify_pose(image, results):
                
                # Copy image object (maybe delete?)
                output_frame = image.copy()

                # Retrieve pose landmarks from holistic model
                pose_landmarks = results.pose_landmarks

                # Check for detection
                if pose_landmarks is not None:

                    # Get image shape
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]

                    # Unpack pose landmarks. Reformat image size. 
                    pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)
                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                    

                    # Classify the image and return the pose
                    pose_classification = pose_classifier(pose_landmarks)
                    pose = max(pose_classification, key=pose_classification.get)
                    #result_queue.put(pose)
                    print(pose)
                    return pose


def tracker(results, screenHeight, screenWidth):
    if results.left_hand_landmarks is not None:
        x = results.left_hand_landmarks.landmark[8].x * screenWidth # scaled
        y = results.left_hand_landmarks.landmark[8].y * screenHeight # scaled
        
        try:
            pyautogui.moveTo(x, y)
        except Exception as e:
            print(e)


def pose_classification_process(image):
    return None



if __name__=='__main__':
    # capture video
    cap = cv2.VideoCapture(0)

    # set up mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Start pose classifier
    pose_classifier = init_pose_classifier('poses_csvs_out')


    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            
            # Retrieve frame
            success, image = cap.read()

            # Start framerate tracker
            start = time.time()
           
            # Flip image horizontally for feedback display. Convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False

            # Process the image and detect the holistic landmarks
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True

            # Convert image back to BGR for deedback display.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


            #---------------------------------------------------------------------------------------------
            #result_queue = Queue()

            # Classify pose

            #p1 = Process(target=classify_pose, args=(image, results))
            #p1.start()
            #print(result_queue.get())
            #p1.join()
            #thread1 = Thread(target=classify_pose, args=(image, results))
            #thread1.start()
            #pose = thread1.join()

            # Retrieve output screen dimensions
            screenWidth, screenHeight = pyautogui.size() # retrieve screen height, width(pixels)
            tracker(results, screenHeight, screenWidth)
            #thread2 = Thread(target=tracker, args=(results, screenHeight, screenWidth))
            #thread2.start()


            #pose = classify_pose(image, results)
            #cv2.putText(image, f'Pose: {pose}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2) # provide feedback on current pose

            

            # Move mouse (e.g. tracker)
            #p2 = Process(result_queue, target = tracker(results, screenHeight, screenWidth))
            #p2.start()
            #tracker(results, screenHeight, screenWidth)
           # print(result_queue.get())
            

            # try:
            #     # Anchor on right index finger --> landmark 8
            #     x = results.left_hand_landmarks.landmark[8].x * screenWidth # scaled
            #     y = results.left_hand_landmarks.landmark[8].y * screenHeight # scaled
            #     z = results.left_hand_landmarks.landmark[8].z

            #     # Move cursor
            #     pyautogui.moveTo(x, y)

            # except Exception as e:
            #     print(e)

            # ##########################
            # ## SANDBOX: EDIT BELOW ##
            # ##########################

            


            # # TEMPLATE START
            # try:
            #     if (action=='refresh') and (pose_classification[action] == 10): #  Strong Confidence:
            #         #pyautogui.PAUSE = 2
            #         print('REFRESH')
            #         pyautogui.hotkey('command', 'r') # refresh
            #         #pyautogui.scroll(-5) #scroll down
            #         #cv2.putText(image, f'Command: {action}', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #         #pyautogui.PAUSE = 0
            # except Exception as e:
            #     print(e)
            # #TEMPLATE END

            # try:
            #     if (action=='select') and (pose_classification[action] == 10): #  Stron Confidence:
            #         print('SELECT')
                    
            #         #pyautogui.PAUSE = 2
            #         pyautogui.click(button='left') #click
            #         #cv2.putText(image, f'Command: {action}', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #         #pyautogui.PAUSE = 0
            # except Exception as e:
            #     print(e)

            # try:
            #     if (action=='zoomIn') and (pose_classification[action] == 10): #& pose_classification[action] == 10: #  Strong Confidence:
            #         #pyautogui.PAUSE = 2
            #         #pyautogui.hotkey('command', '+') #zoom in
            #         pyautogui.scroll(5)
            #         #cv2.putText(image, f'Command: {action}', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #         #pyautogui.PAUSE = 0
            # except Exception as e:
            #     print(e)

            # try:
            #     if (action=='zoomOut') and (pose_classification[action] == 10): #& pose_classification[action] == 10: #  Strong Confidence:
            #         #pyautogui.PAUSE = 2
            #         #pyautogui.hotkey('command', '-') #zoom out
            #         pyautogui.scroll(-5) #scroll down
            #         #cv2.putText(image, f'Command: {action}', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #         #pyautogui.PAUSE = 0
            # except Exception as e:
            #     print(e)

            # '''
            # try:
            #     # Use pinky and thumb as select --> landmark 20, landmark 4
            #     x_thumb = results.right_hand_landmarks.landmark[4].x * screenWidth # scaled
            #     y_thumb = results.right_hand_landmarks.landmark[4].y * screenHeight # scaled

            #     x_pinky = results.right_hand_landmarks.landmark[20].x * screenWidth # scaled
            #     y_pinky = results.right_hand_landmarks.landmark[20].y * screenHeight # scaled

            #     distance = ((((x_thumb - x_pinky)**2)+((y_thumb-y_pinky)**2))**0.5)

            #     #print(f"Distance between pinky and thumb: {distance}")
            #     if distance<25: # adjust
            #         print('Click!')
            #         pyautogui.click(button='left')
            #         cv2.putText(image, f'Command: click', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)


            #     distance_old = distance


            # except Exception as e:
            #     print(e)

        
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
            print("FPS: ", fps)

            #action_stack = fps

            cv2.putText(image, f'FPS: {int(fps)}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2) # provide feedback on current FPS
            cv2.imshow('MediaPipe Holistic', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()




