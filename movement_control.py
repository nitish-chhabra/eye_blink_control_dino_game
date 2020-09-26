import pyautogui
import keyboard
from face_contour import detect_landmarks, eye_aspect_ratio
from constants import FaceTrackingConstants

import cv2 
import numpy as np 
import dlib 
from imutils import face_utils
from scipy.spatial import distance as dist
import time

# initialize the frame counters and the total number of blinks
EYE_AR_THRESH = FaceTrackingConstants.EYE_AR_THRESH #0.27
EYEBROW_DIST_THRESH = FaceTrackingConstants.EYEBROW_DIST_THRESH #0.1
MAR_THRESH = FaceTrackingConstants.MAR_THRESH
CONSECUTIVE_FRAMES = FaceTrackingConstants.CONSECUTIVE_FRAMES #2

BLINK_COUNTER = FaceTrackingConstants.BLINK_COUNTER #0
BROW_COUNTER = FaceTrackingConstants.BROW_COUNTER #0
MAR_COUNTER = FaceTrackingConstants.MAR_COUNTER #0
TOTAL_BLINKS = FaceTrackingConstants.TOTAL_BLINKS #0

cap = cv2.VideoCapture(0) 
# We initialise detector of dlib 
face_detector = dlib.get_frontal_face_detector() 
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

pyautogui.PAUSE = 0.0

# Start the main program 
while(cap.isOpened()): 
    got_frame, frame = cap.read() 

    if got_frame:
        start_time = time.time() 
        # function_frame, decisions, BLINK_COUNTER, BROW_COUNTER, MAR_COUNTER, TOTAL_BLINKS = detect_landmarks(frame, face_detector, landmark_predictor, BLINK_COUNTER, BROW_COUNTER, TOTAL_BLINKS, EYE_AR_THRESH, CONSECUTIVE_FRAMES, EYEBROW_DIST_THRESH, MAR_COUNTER, MAR_THRESH, draw=1)
        decisions, BLINK_COUNTER, BROW_COUNTER, MAR_COUNTER, TOTAL_BLINKS = detect_landmarks(frame, face_detector, landmark_predictor, BLINK_COUNTER, BROW_COUNTER, TOTAL_BLINKS, EYE_AR_THRESH, CONSECUTIVE_FRAMES, EYEBROW_DIST_THRESH, MAR_COUNTER, MAR_THRESH, draw=0)
        print(decisions)

        try:
            if decisions["mouth_opened"] == 1:
                print("Pressing space")
                pyautogui.keyUp('down')
                pyautogui.press('space')

            else:
                if decisions["browed"]:
                    pyautogui.keyDown('down')
                elif decisions["browed"] == 0:
                    pyautogui.keyUp('down')
                else:
                    pass
        except Exception as e:
            print(e)

        # end_time = time.time()
        # fps = round(1/(end_time-start_time),2)
        # cv2.putText(function_frame, "FPS: {}".format(fps), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.imshow("Frame", function_frame) 
        # key = cv2.waitKey(1) 
        # if key == 27: 
        #     break # press esc the frame is destroyed

        if keyboard.is_pressed('esc'):  # if key 'q' is pressed 
            break  # finishing the loop
  
cv2.destroyAllWindows()
cap.release()                      