
## Required Packages and Libraries ##
# import statistics
# from statistics import mode
# from scipy.stats import mode
## Defining import objects for landmark tracking ##
mp_holistic = mp.solutions.holistic # for holistic landmarks
mp_drawing = mp.solutions.drawing_utils # for drawing landmark feedback


## Defining global variables - user feedback ##

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


## audio variable declaration - TODO Max make sonification class or subclass in CI module
currentCoords = [0,0]
lastCoords = [0,0]
zoomLevel = 0
cursorX = 0
cursorY = 0
totalTime = .01

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
def mode(buffer):
    value = max(buffer, key = buffer.count)
    return value
import myfunction as mf
## Main loop ##
cap = cv2.VideoCapture(0) # open web cam
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        
        success, image = cap.read() # capture frame
        start = time.time()

        
        # process image, get holistic model
        import ChoreographicInterface as CI
        lbci = CI()

        lbci.findHolisticLandmarks(image)

        lbci.holisticLandmarks.landmakr

    lbds = CI()

    lbds.add_util(csv,hotkey)

    

        # classify pose

        screenWidth, screenHeight = pyautogui.size() # screen's height/width in pixels - scaling positions from input window

        # update scale bar
            
        
        ## Collect specific landmarks of interest - dexterous movements (Using hand landmarks) ##        


        ## Determine handedness (L vs. R) ##
       

        ## Establish additional constraint for zoomin ##
        try:
            distance_between_shoulderR_wristR = np.linalg.norm(np.array((x_wristL,y_wristL))-np.array((x_shoulderR,y_shoulderR)))
            distance_between_shoulderL_wristL = np.linalg.norm(np.array((x_wristR,y_wristR))-np.array((x_shoulderL,y_shoulderL)))
            if (action == 'zoomIn'):
                if (distance_between_shoulderR_wristR<(distance_between_shoulders*0.75)) and (distance_between_shoulderL_wristL<(distance_between_shoulders*0.75)):
                    pass 
                else:
                    action = 'track' # correct for unintended zoom in
        except Exception as e:
            pass


        # check hand chop orientation
        
        
        ## LOG Actions - modalAction ##
        lmColor = lmColor_map[modalAction]
        connColor = connColor_map[modalAction]
        gesture_text = gesture_map_text[modalAction]
        #for the text, when making track active always only set to track if nothing else


        ## EXECUTE Actions ##
        try:
            '''
            if modalAction == 'track':
                if modalAction != lastExecutedAction:
                    #sonification.sendOSCMessage('trackStart','')
            else:
                #sonification.sendOSCMessage(modalAction,'')
            '''
            #if lastExecutedAction != modalAction:
                #print ('pose changed')
                #print ('modal action is ' + modalAction)
            lastExecutedAction = modalAction
        except Exception as e:
            pass
    
        ## Move and Drag Mouse ##

        ##set handedness vars
        try:

            if which_hand == 'left':
                wristX = x_wristL
                wristY = y_wristL
                thumbX = x_thumbL
                thumbY = y_thumbL
                indexX = x_indexL
                indexY = y_indexL

                # For vitruvian reach logic
                shoulderX = x_shoulderL
                shoulderY = y_shoulderL
                max_reach_y = y_shoulderL+arm_length
                min_reach_y = y_shoulderL+arm_length
                if (wristY-shoulderY)>0: # wrist is higher than shoulder  
                    cursorbump_y = ((wristY-shoulderY)/max_reach_y)*max_distance_wrist_index
                else: # wrist is lower than shoulder
                    cursorbump_y = -1*(((wristY-shoulderY)/max_reach_y)*max_distance_wrist_index)
                #reachY = y_shoulderL+arm_length
                #reachX = x_shoulderL+arm_length
                #print(f'Left Hand X Sign: {np.sign(wristX-shoulderX)}')
                #print(f'Left Hand Y Sign: {np.sign(wristY-shoulderY)}')

            else:
                wristX = x_wristR
                wristY = y_wristR
                thumbX = x_thumbR
                thumbY = y_thumbR
                indexX = x_indexR
                indexY = y_indexR


                # For vitruvian reach logic
                shoulderX = x_shoulderR
                shoulderY = y_shoulderR
                max_reach_y = y_shoulderR+arm_length
                min_reach_y = y_shoulderR+arm_length
                if (wristY-shoulderY)>0: # wrist is higher than shoulder  
                    cursorbump_y = ((wristY-shoulderY)/max_reach_y)*max_distance_wrist_index
                else: # wrist is lower than shoulder
                    cursorbump_y = -1*(((wristY-shoulderY)/max_reach_y)*max_distance_wrist_index)
                #reachY = y_shoulderR+arm_length
                #reachX = x_shoulderR+arm_length
                #print(f'Right Hand X Sign: {np.sign(wristX-shoulderX)}')
                #print(f'Right Hand Y Sign: {np.sign(wristY-shoulderY)}')

        except Exception as e:
            #print(e)
            pass

        try:
            # Vitruvian adjustment part 2 #
            if np.abs(indexY-wristY)>max_distance_wrist_index:
                max_distance_wrist_index = np.abs(indexY-wristY) # proxy for hand size, always take largest value found e.g. should get better over time and adjust to person
        except:
            pass
        
        ##end set handedness vars
        
        if mode(modalMouseButtonList) == 'down':
            if mouse_down == False:
                mouse_down = True
                pyautogui.mouseDown()
                #print('MOUSE DOWN!')
                gesture_text = 'DRAG'
                #sonification.sendMouseDown(mouse_down)
        else:
            if mouse_down == True:
                mouse_down = False
                pyautogui.mouseUp()
                mouse_down = False
                #print('MOUSE UP!')
                #sonification.sendMouseDown(mouse_down)
        
        try:
            if (modalAction == 'track'): # LET MOUSE GO ALWAYS
                drag_threshold = distance_between_shoulders*0.2
                
                # Mouse displacement y-axis correction
                # y_disp = indexY-wristY
                y_disp = 0
                cursorX = wristX
                print(f"CURSOR BUMP: {cursorbump_y}")
                cursorY = wristY+cursorbump_y
                grab_distance = np.linalg.norm(np.array((thumbX,thumbY))-np.array((indexX,indexY)))
                if grab_distance > drag_threshold or handChopOrientation == True: # adjust
                    modalMouseButtonList.append('up')
                elif grab_distance < drag_threshold and modalAction == 'track':
                    modalMouseButtonList.append('down')
                if cursorX is not None and cursorX > 0:
                    currentCoords = [cursorX, cursorY]
                    cursorDifference = math.sqrt( ((currentCoords[0]-lastCoords[0])**2)+((currentCoords[1]-lastCoords[1])**2) )
                    cursorAccel = cursorDifference
                    #print(f'Cursor Acceleration: {cursorAccel}')
                    if cursorAccel > 15:
                        v1 = Vector2(lastCoords[0], lastCoords[1])
                        v2 = Vector2(cursorX, cursorY)
                        # print ('v1 : '+v1)
                        # print ('v2 : '+v2)
                        # print(lerp(v1,v2,1))
                        v3 = lerp(v1, v2, totalTime*.05)
                        pyautogui.moveTo(v3.x,v3.y,0)
                    lastCoords = currentCoords

                    #sonification.sendOSCMessage('track', cursorX)
                    #sonification.sendXAccelerationOSC(cursorAccel)
        except Exception as e:
            #print(e)
            modalMouseButtonList.append('up')
            pass
     

        
        # mouse select #


        ## Determine FPS ##     
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        ## Format User Feedback ... Position feedback bottom right corner (nudged (5, 33) to look just right on MacBook Air) ##
        rad = 4
        thick = 3

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
        if which_hand == 'right':
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
        elif which_hand == 'left':
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad),mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad))
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        
        capWidth  = cap.get(3) # input frame width
        capHeight = cap.get(4) # input frame height
        scale = 0.75 # Scale feedback 0.23 for Lightbox
        newCapWidth = int(screenWidth * (scale))
        newCapHeight = int(newCapWidth * capHeight / capWidth)

        capX = screenWidth - newCapWidth - 5
        capY = screenHeight - newCapHeight - 33
        #cv2.putText(image, f"FPS: {int(fps)}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2) # provide feedback on current FPS
        
        if mouse_down == True:
            # Mac
            cv2.putText(image, 'DRAG', (5, 150), cv2.FONT_HERSHEY_DUPLEX, 4, midnight, 2, cv2.LINE_4)
            # Lightbox
            # cv2.putText(image, 'DRAG', (5, 75), cv2.FONT_HERSHEY_DUPLEX, 2.5, midnight, 2, cv2.LINE_4)
        elif click == True:
            # Mac
            cv2.putText(image, 'SELECT', (5, 150), cv2.FONT_HERSHEY_DUPLEX, 4, midnight, 2, cv2.LINE_4)
            # Lightbox
            # cv2.putText(image, 'SELECT', (5, 75), cv2.FONT_HERSHEY_DUPLEX, 2.5, midnight, 2, cv2.LINE_4)
        else:
            # Mac
            cv2.putText(image, f'{gesture_text}', (5, 150), cv2.FONT_HERSHEY_DUPLEX, 4, midnight, 2, cv2.LINE_4)
            # Lightbox
            # cv2.putText(image, f'{gesture_text}', (5, 75), cv2.FONT_HERSHEY_DUPLEX, 2.5, midnight, 2, cv2.LINE_4)

        cv2.imshow('Choreographic Interface', cv2.resize(image, (newCapWidth, newCapHeight)))
        cv2.setWindowProperty('Choreographic Interface', cv2.WND_PROP_TOPMOST, 1) # keeps feedback window most front
        cv2.moveWindow('Choreographic Interface', capX,capY) # relocate feedback

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()


