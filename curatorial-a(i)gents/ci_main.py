import CIModuleMOD as ci
import cv2
import pyautogui
import time

## VIZ STUFF TODO: put in module - make sure pose specific color is on
# BGR code for mp_drawing
lav = (178, 160, 187) 
fuschia = (143, 18, 172)
green = (75, 79, 33)
blue = (144, 59, 42)
maroon = (54, 37, 122)
turquoise = (233, 206, 0)
redOrange = (58, 45, 240)
midnight = (49, 26, 29)
white = (255, 255, 255)
black = (0, 0, 0)
lmColor = lav #default landmark color
connColor = lmColor #cdefault connector color
gesture_text = ''
lmColor_map = {'refresh':blue,'zoomIn':turquoise,'zoomOut':fuschia,'scrollUp':green,'scrollDown':maroon,'track':turquoise,'neutral':white}
connColor_map = {'refresh':white,'zoomIn':fuschia,'zoomOut':turquoise,'scrollUp':maroon,'scrollDown':green,'track':lav,'neutral':blue}
rad = 2
thick = 2


settings = 'lightbox' # or lightbox
if settings == 'mac':
    settings = [5,150,4]
elif settings == 'lightbox':
    settings = [5,75,2.5] 

selection = 'HAMS-LB'

# no neutral
set_maps = {'HAMS-LB':{1:'circle',2:'handsShoulders',3:'sideT',4:'hips',5:'Vdown',6:'track'}}
classifiers = {'HAMS-LB':'031322_LogReg_pose_classifier.pkl'} 
gesture_map = {'circle':'refresh','handsShoulders':'zoomIn','sideT':'zoomOut','hips':'scrollUp','Vdown':'scrollDown','track':'track'} # for set HAMS-LB
gesture_map_text = {'refresh':'REFRESH','zoomIn':'ZOOM IN','zoomOut':'ZOOM OUT','scrollUp':'SCROLL UP','scrollDown':'SCROLL DOWN','track':'TRACK'} 
action = 'track'

# neutral
#set_maps = {'HAMS-LB': {1:'neutral',2:'circle',3:'handsShoulders',4:'sideT',5:'hips',6:'Vdown',7:'track'}}
#classifiers = {'HAMS-LB':'HAMS-LB_LogReg_pose_classifier.pkl'} # New set with universal 'track
#gesture_map = {'circle':'refresh','handsShoulders':'zoomIn','sideT':'zoomOut','hips':'scrollUp','Vdown':'scrollDown','track':'track','neutral':'neutral'} # for set HAMS-LB
#gesture_map_text = {'refresh':'REFRESH','zoomIn':'ZOOM IN','zoomOut':'ZOOM OUT','scrollUp':'SCROLL UP','scrollDown':'SCROLL DOWN','track':'TRACK','neutral':'NEUTRAL'} 
action = 'neutral'

# important stuff
class_map = set_maps[str(selection)]
pkl_filename = classifiers[str(selection)]

## audio variable declaration --> TODO included in CI module
totalTime = .01

lb_ci = ci.ChoreographicInterface()

# inir classifier
lb_ci.initPoseClasssifier(pkl_filename)

cap = cv2.VideoCapture(0) # open web cam
while cap.isOpened():

    try:

        success, image = cap.read() # capture frame
        screenWidth, screenHeight = pyautogui.size() # screen's height/width in pixels - scaling positions from input window
        start = time.time() # FPS time keeping

        # process image, get holistic model, update landmarks of interest
        image = lb_ci.findHolisticLandmarks(image)
        lb_ci.updateLandmarksOfInterest()
        #print(lb_ci.landmarksOfInterest)

        # classify pose - TODO: maybe keep extra constraint on zoomIn add if needed during testing
        action = lb_ci.classifyPose(image,class_map,gesture_map)
        lb_ci.logAction()
        #print(lb_ci.modalAction[0][0])
        try:
            gesture_text = gesture_map_text[lb_ci.modalAction[0][0]]
            lmColor = lmColor_map[lb_ci.modalAction[0][0]]
            connColor = connColor_map[lb_ci.modalAction[0][0]]
        except Exception as e:
            pass

        # execute action
        executedAction = lb_ci.executeAction()
        #if executedAction != None:
        #    print(executedAction)

        # update scale bar
        lb_ci.updateScaleBar(screenWidth,screenHeight)
        #print(lb_ci.scaleBar)

        # determine which hand to use - TODO change this so ear and face center are in the landmarks dictionary
        lb_ci.determineHandedness(screenWidth,screenHeight)
        #print(lb_ci.whichHand)

        # TODO prevent runaway buffer. list that gets very large.... e.g. if list bigger than look back start shedding one 

        # TODO: temp --> add click later 

        # Mouse movement and drag
        #lb_ci.determineHandOrientation(screenWidth,screenHeight)
        lb_ci.moveDragMouse(screenWidth,screenHeight)

        
        #print(f"screen height: {screenHeight}")
        #print(f"screen width {screenWidth}")
        #print(f"wristx: {lb_ci.landmarksOfInterest['rightWrist'].x*screenWidth}")
        #print(f"wristx: {lb_ci.landmarksOfInterest['rightWrist'].y*screenHeight}")

    
        #print(lb_ci.clickBuffer)

        # VIZ - TODO: add to module
        capWidth  = cap.get(3) # input frame width
        capHeight = cap.get(4) # input frame height
        scale = 0.23 # Scale feedback 0.23 for Lightbox .75 for mc, TODO addo config
        newCapWidth = int(screenWidth * (scale))
        newCapHeight = int(newCapWidth * capHeight / capWidth)
        capX = screenWidth - newCapWidth - 5
        capY = screenHeight - newCapHeight - 33
        lb_ci.mpDraw.draw_landmarks(image, lb_ci.holisticLandmarks.pose_landmarks, lb_ci.mpHolsitic.POSE_CONNECTIONS, lb_ci.mpDraw.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),lb_ci.mpDraw.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
        if lb_ci.whichHand == 'right':
                lb_ci.mpDraw.draw_landmarks(image, lb_ci.holisticLandmarks.left_hand_landmarks, lb_ci.mpHolsitic.HAND_CONNECTIONS, lb_ci.mpDraw.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),lb_ci.mpDraw.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
        elif lb_ci.whichHand == 'left':
            lb_ci.mpDraw.draw_landmarks(image, lb_ci.holisticLandmarks.right_hand_landmarks, lb_ci.mpHolsitic.HAND_CONNECTIONS, lb_ci.mpDraw.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),lb_ci.mpDraw.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
        if lb_ci.mouseDown == True:
                cv2.putText(image, 'SELECT',  (settings[0], settings[1]), cv2.FONT_HERSHEY_DUPLEX, settings[2], midnight, 2, cv2.LINE_4)
        #elif click == True:
        #    cv2.putText(image, 'SELECT', (settings[0], settings[1]), cv2.FONT_HERSHEY_DUPLEX, settings[2], midnight, 2, cv2.LINE_4)
        else:
            cv2.putText(image, f'{gesture_text}', (settings[0], settings[1]), cv2.FONT_HERSHEY_DUPLEX, settings[2], midnight, 2, cv2.LINE_4)
        cv2.imshow('Choreographic Interface', cv2.resize(image, (newCapWidth, newCapHeight)))
        cv2.setWindowProperty('Choreographic Interface', cv2.WND_PROP_TOPMOST, 1) # keeps feedback window most front
        cv2.moveWindow('Choreographic Interface', capX,capY) # relocate feedback
        if cv2.waitKey(5) & 0xFF == 27:
            break 
        
        time.sleep(1)
        
    except Exception as e:
        #print(e)
        pass

cap.release()