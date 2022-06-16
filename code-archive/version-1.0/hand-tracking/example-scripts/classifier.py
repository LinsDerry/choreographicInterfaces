from imutils import face_utils
import numpy as np
import imutils
import argparse
import dlib
import cv2





# Video capture source
cap = cv2.VideoCapture(0)

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector = dlib.simple_object_detector("detector.svm")


#We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

win = dlib.image_window()

while True:

    ret, image = cap.read()
    image = imutils.resize(image, width=800)

    rects = detector(image)

    for k, d in enumerate(rects):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        #shape = predictor(image, d)
        #shape = face_utils.shape_to_np(shape)

        #(x, y, w, h) = face_utils.rect_to_bb(rect)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


    win.clear_overlay()
    win.set_image(image)
    win.add_overlay(rects)



      