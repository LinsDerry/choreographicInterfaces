##################
# Jordan Kruguer #
# Github:        #
##################

import multiprocessing
import numpy as np
import cv2
import tensorflow.keras as tf
import pyttsx3
import math
# use matplotlib if cv2.imshow() doesn't work
import matplotlib.pyplot as plt

import imutils # for detector
import dlib 


# Cursor control
import time
#from pynput.mouse import Button, Controller
#mouse = Controller()
import pyautogui, sys


# this process is purely for text-to-speech so it doesn't hang processor
def speak(speakQ, ):
    # initialize text-to-speech object
    engine = pyttsx3.init()
    # can adjust volume if you'd like
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume)  # add number here
    # initialize last_msg to be empty
    last_msg = ""
    # keeps program running forever until ctrl+c or window is closed
    while True:
        msg = speakQ.get()
        # clear out msg queue to get most recent msg
        while not speakQ.empty():
            msg = speakQ.get()
        # if most recent msg is different from previous msg
        # and if it's not "Background"
        if msg != last_msg and msg != "Background":
            last_msg = msg
            # text-to-speech say class name from labels.txt
            engine.say(msg)
            engine.runAndWait()
        if msg == "Background":
            last_msg = ""


# main line code
# if statement to circumvent issue in windows
if __name__ == '__main__':

    # read .txt file to get labels
    labels_path = "/Users/jordankruguer/Documents/repositories/metalab/curatorial_agents/teachable_machine/Teachable-Machine-Object-Detection/converted_keras_most_recent/labels.txt" # UPDATE TO CORRECT RELATIVE PATH
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

    # load the teachable machine model
    model_path = '/Users/jordankruguer/Documents/repositories/metalab/curatorial_agents/teachable_machine/Teachable-Machine-Object-Detection/converted_keras_most_recent/keras_model.h5' # UPDATE TO CORRECT RELATIVE PATH
    model = tf.models.load_model(model_path, compile=False)
 


    # initialize webcam video object
    cap = cv2.VideoCapture(0)

    # width & height of webcam video in pixels -> adjust to your size
    # adjust values if you see black bars on the sides of capture window
    frameWidth = 1280
    frameHeight = 720

    # set width and height in pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    # enable auto gain
    cap.set(cv2.CAP_PROP_GAIN, 0)


    detector = dlib.simple_object_detector("detector_april.svm")

    #We can look at the HOG filter we learned.  It should look like a face.  Neat!
    win_det = dlib.image_window() # detector
    win_det.set_image(detector) # detector

    win = dlib.image_window() # detector

    # creating a queue to share data to speech process
    speakQ = multiprocessing.Queue()

    # creating speech process to not hang processor
    p1 = multiprocessing.Process(target=speak, args=(speakQ, ), daemon="True")

    # starting process 1 - speech
    p1.start()

    # keeps program running forever until ctrl+c or window is closed
    while True:

        # disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model.
        # We are inputting 1x 224x224 pixel RGB image.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # capture image
        check, frame = cap.read()
        # mirror image - mirrored by default in Teachable Machine
        # depending upon your computer/webcam, you may have to flip the video
        frame = cv2.flip(frame, 1)

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
        #print(predictions)

        
        ## Hand tracking ##

        frame = imutils.resize(frame, width=800)

        rects = detector(frame)

        # getting screen's height in pixels, getting screen's width in pixels 
        screenWidth, screenHeight = pyautogui.size()

        input_x = 800 # dlib window (abstract) TODO
        input_y = 500 # dlib window (abstract) TODO

        scaling_factor = input_x/screenWidth


        for k, d in enumerate(rects):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            x = (d.right()-d.left())/2 + d.left()
            y = (d.bottom()-d.top())/2 + d.top()

            print(f'position of centroid {x} {y}')

            output_x = x/scaling_factor
            output_y = y/scaling_factor

            pyautogui.moveTo(output_x, output_y)
            
	 

        win.clear_overlay()
        win.set_image(frame)
        win.add_overlay(rects)
        ## Hand tracking ##
        



        # confidence threshold is 90%.
        conf_threshold = 95
        confidence = []
        conf_label = ""
        threshold_class = ""
        # create blach border at bottom for labels
        per_line = 2  # number of classes per line of text
        bordered_frame = cv2.copyMakeBorder(
            square_frame,
            top=0,
            bottom=30 + 15*math.ceil(len(classes)/per_line),
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        # for each one of the classes
        for i in range(0, len(classes)):
            # scale prediction confidence to % and apppend to 1-D list
            confidence.append(int(predictions[0][i]*100))
            # put text per line based on number of classes per line
            if (i != 0 and not i % per_line):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil(i/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # append classes and confidences to text for label
            conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            # prints last line
            if (i == (len(classes)-1)):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil((i+1)/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""

            # if above confidence threshold, send to queue
            if confidence[i] > conf_threshold:
                speakQ.put(classes[i])
                threshold_class = classes[i]
                
    
                #if i == 0: # neutral
                    #continue # do nothing

               	#if i == 1: # track
                    #continue # do nothing

                if i == 2: # select
               	    pyautogui.click(button='left')
                    #continue # do nothing

                elif i == 3: # scrlDn
                    #pyautogui.scroll(-5)
                    pyautogui.hotkey('command', '-')
                    continue

                elif i == 4: # scrlUp
                	   #pyautogui.scroll(5)
                	   pyautogui.hotkey('command', '+')
                	   continue

                elif i == 5: # start
                    pyautogui.hotkey('command', 'r')
                    #pyautogui.move(0, 10)
                    #pyautogui.move(10, 0)
                    #pyautogui.move(0, -10)
                    #pyautogui.move(-10, 0)
                    continue

                else:
                    continue
                
                


        # add label class above confidence threshold
        cv2.putText(
            img=bordered_frame,
            text=threshold_class,
            org=(int(0), int(frameHeight+20)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255)
        )

        # original video feed implementation
        cv2.imshow("Capturing", bordered_frame)
        cv2.waitKey(10)

        # # if the above implementation doesn't work properly
        # # comment out two lines above and use the lines below
        # # will also need to import matplotlib at the top
        # plt_frame = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(plt_frame)
        # plt.draw()
        # plt.pause(.001)

    # terminate process 1
    p1.terminate()
