import CIModule as ci
import cv2
import pyautogui
import time

## VIZ STUFF TODO: put in module - make sure pose specific color is on
# BGR code for mp_drawing
lav = (178, 160, 187) 
fuschia = (143, 18, 172)
green = (230, 255, 150)
blue = (144, 59, 42)
red = (10, 75, 255)
turquoise = (233, 206, 0)
white = (255, 255, 255)
black = (0, 0, 0)
lmColor = lav #default landmark color
connColor = lmColor #cdefault connector color
gesture_text = ''
lmColor_map = {'refresh':blue,'zoomIn':turquoise,'zoomOut':fuschia,'scrollUp':green,'scrollDown':red,'track':turquoise,'right':white,'left':white}
connColor_map = {'refresh':white,'zoomIn':fuschia,'zoomOut':turquoise,'scrollUp':red,'scrollDown':green,'track':lav,'right':blue,'left':blue}
rad = 4 # 2 for LB
thick = 4 # 2 for LB

# settings for cv2 visual feedback
# settings[0] = x coordinate for beginning text
# settings[1] = y coordinate for beginning text
# settings[2] = text size
#settings[3] = scale of feedback window (e.g., 0.4 is 40% screen width)
settings = 'mac' # mac or lightbox
if settings == 'mac':
    settings = [5,150,4,0.4]
    # settings = [1000,150,4,0.4] #for video recording only
elif settings == 'lightbox':
    settings = [5,75,2.5,0.23]

selection = 'HAMS-LB'

# no neutral, with right hip and left hip for "This Recommendation System is Broken"
#set_maps = {'HAMS-LB':{1:'circle',2:'handsShoulders',3:'sideT',4:'hips',5:'Vdown',6:'track',7:'hipL',8:'hipR'}}
#classifiers = {'HAMS-LB':'011322_LogReg_pose_classifier_hipsincluded.pkl'} 
#gesture_map = {'circle':'refresh','handsShoulders':'zoomIn','sideT':'zoomOut','hips':'scrollUp','Vdown':'scrollDown','track':'track','hipL':'left','hipR':'right'} # for set HAMS-LB
#gesture_map_text = {'refresh':'REFRESH','zoomIn':'ZOOM IN','zoomOut':'ZOOM OUT','scrollUp':'SCROLL UP','scrollDown':'SCROLL DOWN','track':'TRACK','left':'LEFT','right':'RIGHT'} 
#action = 'track'

# no neutral
set_maps = {'HAMS-LB':{1:'circle',2:'handsShoulders',3:'sideT',4:'hips',5:'Vdown',6:'track'}}
classifiers = {'HAMS-LB':'./pose-set-pkl-files/031322_LogReg_pose_classifier.pkl'} 
gesture_map = {'circle':'refresh','handsShoulders':'zoomIn','sideT':'zoomOut','hips':'scrollUp','Vdown':'scrollDown','track':'track'} # for set HAMS-LB
gesture_map_text = {'refresh':'REFRESH','zoomIn':'ZOOM IN','zoomOut':'ZOOM OUT','scrollUp':'SCROLL UP','scrollDown':'SCROLL DOWN','track':'TRACK'} 
action = 'track'

# important stuff
class_map = set_maps[str(selection)]
pkl_filename = classifiers[str(selection)]

## audio variable declaration --> TODO included in CI module
currentCoords = [0,0]
lastCoords = [0,0]
zoomLevel = 0
cursorX = 0
cursorY = 0
totalTime = .01


lb_ci = ci.ChoreographicInterface()

# inir classifier
lb_ci.initPoseClasssifier(pkl_filename)

cap = cv2.VideoCapture(0) # open web cam
while cap.isOpened():


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

    # TODO: temp --> add click later 

    # Mouse movement and drag
    #lb_ci.determineHandOrientation(screenWidth,screenHeight)
    
    lb_ci.moveDragMouse(screenWidth,screenHeight)
 
    #print(lb_ci.clickBuffer)

    # VIZ - TODO: add to module
    capWidth  = cap.get(3) # input frame width
    capHeight = cap.get(4) # input frame height
    scale = settings[3] # Scale feedback
    newCapWidth = int(screenWidth * (scale))
    newCapHeight = int(newCapWidth * capHeight / capWidth)
    # bottom right default
    capX = screenWidth - newCapWidth - 5
    capY = screenHeight - newCapHeight - 33

    # upper right for "This Rec." and "AIxquisite Corpse"
    # capX = screenWidth - newCapWidth - 5
    # capY = 5

    # #bottom left for "A Flitting Atlas"
    # capX = 5
    # capY = screenHeight - newCapHeight - 33

    lb_ci.mpDraw.draw_landmarks(image, lb_ci.holisticLandmarks.pose_landmarks, lb_ci.mpHolsitic.POSE_CONNECTIONS, lb_ci.mpDraw.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),lb_ci.mpDraw.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
    if lb_ci.whichHand == 'right':
            lb_ci.mpDraw.draw_landmarks(image, lb_ci.holisticLandmarks.left_hand_landmarks, lb_ci.mpHolsitic.HAND_CONNECTIONS, lb_ci.mpDraw.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),lb_ci.mpDraw.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
    elif lb_ci.whichHand == 'left':
        lb_ci.mpDraw.draw_landmarks(image, lb_ci.holisticLandmarks.right_hand_landmarks, lb_ci.mpHolsitic.HAND_CONNECTIONS, lb_ci.mpDraw.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),lb_ci.mpDraw.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
    if lb_ci.mouseDown == True:
            cv2.putText(image, 'SELECT',  (settings[0], settings[1]), cv2.FONT_HERSHEY_DUPLEX, settings[2], black, 2, cv2.LINE_4)
    #elif click == True:
    #    cv2.putText(image, 'SELECT', (settings[0], settings[1]), cv2.FONT_HERSHEY_DUPLEX, settings[2], black, 2, cv2.LINE_4)
    else:
        cv2.putText(image, f'{gesture_text}', (settings[0], settings[1]), cv2.FONT_HERSHEY_DUPLEX, settings[2], black, 2, cv2.LINE_4)
        # cv2.putText(image, 'TRACK', (settings[0], settings[1]), cv2.FONT_HERSHEY_DUPLEX, settings[2], black, 2, cv2.LINE_4) #for projects only using track and select
    cv2.imshow('Choreographic Interface', cv2.resize(image, (newCapWidth, newCapHeight)))
    cv2.setWindowProperty('Choreographic Interface', cv2.WND_PROP_TOPMOST, 1) # keeps feedback window most front
    cv2.moveWindow('Choreographic Interface', capX, capY) # relocate feedback
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # print(f"Elapsed Time: {time.time()-start}")

cap.release()