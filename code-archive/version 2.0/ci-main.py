# Author(s): Jordan Kruguer, Lins Derry

import multiprocessing
import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import pose_module 

import os
import math
import time
import numpy as np
import pyautogui, sys

# Keras (UNCOMMENT IF YOU ARE USING KERAS)
#import tensorflow.keras as tf
# Keras

# Multiprocessing
import multiprocessing as mproc
# Multiprocessing 

# Multithreading
import threading
# Multithreading

# Ray 
#import psutil
#import ray
#num_cpus = psutil.cpu_count(logical=False)
#ray.init(num_cpus=num_cpus)
#ray.init()
# Ray

########################################
# POSE CLASSIFICATION HELPER FUNCTIONS #
########################################

#@ray.remote
def init_pose_classifier(pose_samples_folder):
    # Initialize embedder.
    pose_embedder = pose_module.FullBodyPoseEmbedder()

    # Initialize classifier.
    pose_classifier = pose_module.PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

    return pose_classifier

##@ray.remote
def classify_pose(results, pose_classifier, image):
    
    # Retrieve pose landmarks from holistic model
    pose_landmarks = results.pose_landmarks

    # Copy image object (maybe delete?)
    output_frame = image.copy()

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
        pose_action(pose)
        return pose

def pose_action(pose):

    if pose == 'refresh':
        pyautogui.hotkey('command', 'r') # refresh
    elif pose == 'zoomIn':
        pyautogui.hotkey('command', '+') #zoom in
    elif pose == 'zoomOut':
        pyautogui.hotkey('command', '-') #zoom out
    elif pose == 'scrollUp':
        pyautogui.scroll(5)
    elif pose == 'scrollDn': 
        pyautogui.scroll(-5)
     
    # Alternative zoom
    #wrist = results.right_hand_landmarks.landmark[0].z * screenWidth # scaled
    #if wrist < 0:
    #    pyautogui.hotkey('command', '-')
    #elif wrist > 0:
    #    pyautogui.hotkey('command', '+')


def init_pose_classifier_tm():

    # read .txt file to get labels
    labels_path = "./keras/labels.txt" 
    labelsfile = open(labels_path, 'r')

    # initialize classes and read in lines until there are no more
    classes = []
    line = labelsfile.readline()
    while line:
        # retrieve just class name and append to classes
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
    # close label file
    labelsfile.close()

    # load the teachable machine model
    model_path = './keras/model.h5' 
    model = tf.models.load_model(model_path, compile=False)

    return classes, model

def classify_pose_tm(model, image):

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    frame = cv2.flip(image, 1)

    frameWidth = 1280
    frameHeight = 720
    
    # crop to square for use with TM model
    margin = int(((frameWidth-frameHeight)/2))
    square_frame = frame[0:frameHeight, margin:margin + frameHeight]
    # resize to 224x224 for use with TM model
    resized_img = cv2.resize(square_frame, (224, 224))
    # convert image color to go to model
    model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # turn the image into a numpy array
    image_array = np.asarray(model_img)
    # normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # load the image into the array
    data[0] = normalized_image_array

    # run the prediction
    predictions = model.predict(data)
    max_value = max(list(predictions[0]))
    max_index = list(predictions[0]).index(max_value)
    print(max_index)
    pose_dict = {0:'neutral',1:'track',2:'select',3:'scrlDn',4:'scrlUp',5:'start'} # UPDATE THIS
    
    pose = pose_dict[max_index-1]
    print(pose)
    return pose

def pose_action_tm(pose):
    pass # COPY pose_action format but with tm classes @ Lins

    



######################################
# MOUSE INTERACTION HELPER FUNCTIONS #
######################################


def handle_mouse(mouse_status, results, creenHeight, screenWidth): # ACTUALLY RIGHT HAND

    mouse_down = mouse_status 

    if results.left_hand_landmarks is not None:
        x = results.left_hand_landmarks.landmark[8].x * screenWidth # scaled
        y = results.left_hand_landmarks.landmark[8].y * screenHeight # scaled

        # Use pinky and thumb as select --> landmark 20, landmark 4
        x_thumb_r = results.left_hand_landmarks.landmark[4].x * screenWidth # scaled
        y_thumb_r = results.left_hand_landmarks.landmark[4].y * screenHeight # scaled
        x_ring_r = results.left_hand_landmarks.landmark[16].x * screenWidth # scaled
        y_ring_r = results.left_hand_landmarks.landmark[16].y * screenHeight # scaled

        distance_r = ((((x_thumb_r - x_ring_r)**2)+((y_thumb_r-y_ring_r)**2))**0.5)
        print(distance_r)

        if distance_r < 35:
            pyautogui.moveTo(x, y)

        
    if results.right_hand_landmarks is not None: # ACTUALLY LEFT HAND
        
        # Use pinky and thumb as select --> landmark 20, landmark 4
        x_thumb = results.right_hand_landmarks.landmark[4].x * screenWidth # scaled
        y_thumb = results.right_hand_landmarks.landmark[4].y * screenHeight # scaled
        x_ring = results.right_hand_landmarks.landmark[16].x * screenWidth # scaled
        y_ring = results.right_hand_landmarks.landmark[16].y * screenHeight # scaled

        distance_l = ((((x_thumb - x_ring)**2)+((y_thumb-y_ring)**2))**0.5)
        #print(distance_l)

        if distance_l < 35: # adjust
            pyautogui.mouseDown()
            cv2.putText(image, 'Select', (25,140), cv2.FONT_HERSHEY_TRIPLEX, 1.75, red, 2)
            mouse_down = True
            print('MOUSE UP')
            return mouse_down

        elif (distance_l > 35) and (mouse_down == True):
            print('MOUSE DOWN')
            pyautogui.mouseUp()
            mouse_down = False
            return mouse_down

def feedback():

    lmColor = pink
    connColor = lav
    rad = 9
    thick = 4

    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=(rad // 2)),
    mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=(rad // 2)),
    mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),
    mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))

    # Get width and height of camera capture
    capWidth  = cap.get(3) 
    capHeight = cap.get(4)

    # Scale feedback
    scale = 0.25
    newCapWidth = int(screenWidth * (scale))
    newCapHeight = int(newCapWidth * capHeight / capWidth)

    # Position feedback bottom right corner (nuudged (5, 33) to look just right on MacBook Air)
    capX = screenWidth - newCapWidth - 5
    capY = screenHeight - newCapHeight - 33

    #cv2.putText(image, f'Interactive gesture: {pose}', (25,70), cv2.FONT_HERSHEY_TRIPLEX, 1.75, red, 3) # provide feedback on current pose
    cv2.imshow('Choreographic Interface', cv2.resize(image, (newCapWidth, newCapHeight)))
    cv2.setWindowProperty('Choreographic Interface', cv2.WND_PROP_TOPMOST, 1) # keeps feedback window most front
    cv2.moveWindow('Choreographic Interface', capX,capY) # relocate feedback


#############
# MAIN LOOP #
#############

if __name__=='__main__':

    mouse_status = False # e.g. mouse starts up
    lav = (187, 160, 178) # RGB for mp_drawing
    pink = (172, 18, 143) # RGB for mp_drawing
    red = (58, 45, 240) # BGR for cv2 text
    white = (0, 0, 0)


    # For multiprocessing
    #pool = mproc.Pool(mproc.cpu_count())   

    # capture video
    cap = cv2.VideoCapture(0)

    # Start pose classifier (if you are using MediaPipe)
    pose_classifier = init_pose_classifier('poses_csvs_out')

    # If you want to use keras model (uncomment this)
    #classes, model = init_pose_classifier_tm()

    # set up mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
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

            # Convert image back to BGR for deedback display.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Retrieve output screen dimensions
            screenWidth, screenHeight = pyautogui.size()

            #############################################
            # POSE CLASSIFICATION OPTIONS --> MediaPipe #
            #############################################
            
            # OPTION #1: DIRECT CALL (MediaPipe)
            pose = classify_pose(results,pose_classifier,image)

            # OPTION #2: MULTITHREADING
            #thread1 = threading.Thread(target=classify_pose, args=(pose_classifier,image,))
            #thread1.start()

            # OPTION #3: RAY (Incomplete)
            #classify_pose.remote(pose_classifier,image)

            # OPTION #4: Teachable Machine
            #classify_pose_tm(model,image)

            ############################################
            # MOUSE INTERACTIONS OPTIONS --> MediaPipe # 
            ############################################

            # OPTION #1: DIRECT CALL
            mouse_status = handle_mouse(mouse_status, results, screenHeight, screenWidth)
            
            # OPTION #2: MULTITHREADING
            # thread2 = threading.Thread(target=classify_pose, args=(pose_classifier,image,))
            #thread2.start()
            #mouse_status = thread2.join()

            # OPTION #3: RAY (Incomplete)
            #mouse_status = ray.get(handle_mouse.remote(mouse_status, results, screenHeight, screenWidth))

            # Draw landmarks
            #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            print("FPS: ", fps)

            ##########################
            # HANDLE VISUAL FEEDBACK #
            ##########################
            feedback()
            

            if cv2.waitKey(5) & 0xFF == 27:
                break

    
    cap.release()
