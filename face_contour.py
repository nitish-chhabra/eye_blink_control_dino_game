# We Import the necessary packages needed 
import cv2 
import numpy as np 
import dlib 
from imutils import face_utils
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 2
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0) 
# We initialise detector of dlib 
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
  
# Start the main program 
while True: 
    _, frame = cap.read() 
# We actually Convert to grayscale conversion 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = detector(gray) 
    for face in faces: 
    # The face landmarks code begins from here 
        x1 = face.left() 
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom() 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1) 
        landmarks = predictor(gray, face) 

        shape = face_utils.shape_to_np(landmarks)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        print(leftEAR , rightEAR)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0

            # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

       # We are then accesing the landmark points  
        for n in range(0, 68): 
            x = landmarks.part(n).x 
            y = landmarks.part(n).y 
            cv2.circle(frame, (x, y), 2, (0, 255, 0), 3) 
    iter_frame = cv2.resize(frame,(1920,1080),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    cv2.imshow("Frame", iter_frame) 
    key = cv2.waitKey(1) 
    if key == 27: 
        break # press esc the frame is destroyed