import pyautogui
import keyboard
from face_contour import detect_landmarks, eye_aspect_ratio
from constants import FaceTrackingConstants

import cv2 
import numpy as np 
import dlib 
from imutils import face_utils
from scipy.spatial import distance as dist

# initialize the frame counters and the total number of blinks
EYE_AR_THRESH = FaceTrackingConstants.EYE_AR_THRESH #0.27
EYEBROW_DIST_THRESH = FaceTrackingConstants.EYEBROW_DIST_THRESH #0.1
EYE_AR_CONSEC_FRAMES = FaceTrackingConstants.EYE_AR_CONSEC_FRAMES #2

BLINK_COUNTER = FaceTrackingConstants.BLINK_COUNTER #0
BROW_COUNTER = FaceTrackingConstants.BROW_COUNTER #0
TOTAL = FaceTrackingConstants.TOTAL #0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0) 
# We initialise detector of dlib 
face_detector = dlib.get_frontal_face_detector() 
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# Start the main program 
while(cap.isOpened()): 
    got_frame, frame = cap.read() 

    if got_frame:

        decisions, BLINK_COUNTER, BROW_COUNTER, TOTAL = detect_landmarks(frame, face_detector, landmark_predictor, BLINK_COUNTER, BROW_COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, EYEBROW_DIST_THRESH, draw=0)

        if len(decisions.keys())>0:
            if decisions["blinked"] == 1:
                pyautogui.press('space')
             
            elif decisions["browed"] == 1:
                pyautogui.press('down')
            else:
                pass
        else: 
            pass

        # cv2.imshow("Frame", decisions) 
        if keyboard.is_pressed('esc'):  # if key 'q' is pressed 
            break  # finishing the loop
  
cv2.destroyAllWindows()
cap.release()