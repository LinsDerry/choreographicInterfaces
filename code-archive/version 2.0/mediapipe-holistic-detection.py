import cv2
import mediapipe as mp
import time

import pyautogui, sys
import numpy as np
import math

import tensorflow.keras as tf


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        success, image = cap.read()
       
        start = time.time()


        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and detect the holistic
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



        #print(results.pose_landmarks)
        #break

        #mp_drawing.draw_landmarks(
        #    image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #mp_drawing.draw_landmarks(
        #    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)




        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('MediaPipe Holistic', image)


        ##########################
        ## Move mouse with hand ##
        ##########################

        # getting screen's height in pixels, getting screen's width in pixels 
        screenWidth, screenHeight = pyautogui.size()

        try:
            # Anchor on right index finger --> landmark 8
            x = results.left_hand_landmarks.landmark[8].x * screenWidth # scaled
            y = results.left_hand_landmarks.landmark[8].y * screenHeight # scaled
           #z = results.left_hand_landmarks.landmark[8].z

            # Move cursor
            pyautogui.moveTo(x, y)

        except Exception as e:
            print(e)

        # scroll 
        #try:
        #    pyautogui.scroll(z)
        #except Exception as e:
        #    print(e)


        '''
        try:
            # Use pinky and thumb as select --> landmark 20, landmark 4
            x_thumb = results.left_hand_landmarks.landmark[4].x * screenWidth # scaled
            y_thumb = results.left_hand_landmarks.landmark[4].y * screenHeight # scaled

            x_pinky = results.left_hand_landmarks.landmark[20].x * screenWidth # scaled
            y_pinky = results.left_hand_landmarks.landmark[20].y * screenHeight # scaled

            distance = ((((x_thumb - x_pinky)**2)+((y_thumb-y_pinky)**2))**0.5)

            #print(f"Distance between pinky and thumb: {distance}")
            if distance<25: # adjust
                print('Click!')
                pyautogui.click(clicks=2)

            distance_old = distance
                


        except Exception as e:
            print(e)

        '''

        ########################
        # Pose detection (Old) #
        ########################

        # load the teachable machine model

        # read .txt file to get labels
        labels_path = './labels.txt' 
        # open input file label.txt
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

        model_path = './keras_model.h5' 
        model = tf.models.load_model(model_path, compile=False)


        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        frame = cv2.flip(image, 1)

        frameWidth = screenWidth
        frameHeight = screenHeight

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



        # confidence threshold is 90%.
        conf_threshold = 95
        confidence = []

        for i in range(0, len(classes)):
            
            # scale prediction confidence to % and apppend to 1-D list
            confidence.append(int(predictions[0][i]*100))

            if confidence[i] > conf_threshold:
                threshold_class = classes[i]
                print(f'Class: {threshold_class}\n Confidence {confidence[i]}\n')
                #cv2.putText(image, f'Keras Class: {threshold_class}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)




        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()