# We Import the necessary packages needed 
import cv2 
import numpy as np 
import dlib 
from imutils import face_utils
from scipy.spatial import distance as dist
from constants import FaceTrackingConstants
import time

def eye_aspect_ratio(eye):
    # print("Entered EAR")
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

def eye_to_eyebrow_distance(eye, eyebrow):
    # print("Entered Eye to Eyebrow")
    # compute distance of 3rd and 4th point in eyebrow
    # with 2nd and 3rd point in the eye
    A = dist.euclidean(eye[1], eyebrow[2])
    B = dist.euclidean(eye[2], eyebrow[3])
    # compute the eye aspect ratio
    eye_to_eyebrow_distance = (A + B) / (2.0)
    # return the eye to eyebrow distance
    return eye_to_eyebrow_distance

def detect_landmarks(input_frame, face_detector, landmark_predictor, BLINK_COUNTER, BROW_COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, EYEBROW_DIST_THRESH, draw = 1):
    # print("Entered detect landmarks")

    function_frame = input_frame.copy()

    gray = cv2.cvtColor(function_frame, cv2.COLOR_BGR2GRAY) 
    faces = face_detector(gray)
    decisions = {}
    for face in faces: 
    # The face landmarks code begins from here 
        x1 = face.left() 
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom() 

        if draw:
            cv2.rectangle(function_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        else:
            #Don't draw anything
            pass 

        landmarks = landmark_predictor(gray, face) 

        shape = face_utils.shape_to_np(landmarks)

        (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        (rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

        # Helper Information on how to use the above dictionary:
        # Face utils helps in extracting various face features without need to remember id locations for each feature
        # This dictionary gives guidance on which all Landmarks can be detected
        # FACIAL_LANDMARKS_68_IDXS = OrderedDict([
        # ("mouth", (48, 68)),
        # ("inner_mouth", (60, 68)),
        # ("right_eyebrow", (17, 22)),
        # ("left_eyebrow", (22, 27)),
        # ("right_eye", (36, 42)),
        # ("left_eye", (42, 48)),
        # ("nose", (27, 36)),
        # ("jaw", (0, 17))
        # ]) 

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # extract the left and right eyebrow coordinates, then use the
		# coordinates to compute the eye to eyebrow distance for both eyes
        leftEyeBrow = shape[lebStart:lebEnd]
        rightEyeBrow = shape[rebStart:rebEnd]
        leftEyeToEyebrow = eye_to_eyebrow_distance(leftEye, leftEyeBrow)
        rightEyeToEyebrow = eye_to_eyebrow_distance(rightEye, rightEyeBrow)
        # average the eye aspect ratio together for both eyes
        ete_dist = (leftEyeToEyebrow + rightEyeToEyebrow) / 2.0
 
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        if draw:
            cv2.drawContours(function_frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(function_frame, [rightEyeHull], -1, (0, 255, 0), 1)
        else:
            #Don't draw anything
            pass
        
        decisions = {"blinked" : 0, "browed" : 0}
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            BLINK_COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                decisions["blinked"] = 1
            # reset the eye frame counter
            BLINK_COUNTER = 0
        
        if ete_dist > EYEBROW_DIST_THRESH:
            BROW_COUNTER += 1
            if BROW_COUNTER >= EYE_AR_CONSEC_FRAMES:
                decisions["browed"] = 1
            else:
                pass
        else:
            BROW_COUNTER = 0


        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        if draw:
            cv2.putText(function_frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(function_frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(function_frame, "BD: {:.2f}".format(ete_dist), (420, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(function_frame, "In Blink: {:.2f}".format(decisions["blinked"]), (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(function_frame, "In Brow: {:.2f}".format(decisions["browed"]), (200, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # We are then accesing the landmark points  
            for n in range(0, 68): 
                x = landmarks.part(n).x 
                y = landmarks.part(n).y 
                cv2.circle(function_frame, (x, y), 2, (0, 255, 0), 2)

        else:
            #Don't draw anything
            pass

    if draw:
        return function_frame, BLINK_COUNTER, BROW_COUNTER , TOTAL
    else:
        return decisions, BLINK_COUNTER, BROW_COUNTER, TOTAL

if __name__ == "__main__":

    # initialize the frame counters and the total number of blinks
    EYE_AR_THRESH = FaceTrackingConstants.EYE_AR_THRESH #0.27
    EYEBROW_DIST_THRESH = FaceTrackingConstants.EYEBROW_DIST_THRESH #0.1
    EYE_AR_CONSEC_FRAMES = FaceTrackingConstants.EYE_AR_CONSEC_FRAMES #2

    BLINK_COUNTER = FaceTrackingConstants.BLINK_COUNTER #0
    BROW_COUNTER = FaceTrackingConstants.BROW_COUNTER #0
    TOTAL = FaceTrackingConstants.TOTAL #0

    cap = cv2.VideoCapture(0) 
    # We initialise detector of dlib 
    face_detector = dlib.get_frontal_face_detector() 
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

    # Start the main program 
    while(cap.isOpened()): 
        got_frame, frame = cap.read() 

        if got_frame:
            start_time = time.time()
            output_frame, BLINK_COUNTER, BROW_COUNTER, TOTAL = detect_landmarks(frame, face_detector, landmark_predictor, BLINK_COUNTER, BROW_COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, EYEBROW_DIST_THRESH)
            end_time = time.time()
            fps = round(1/(end_time-start_time),2)
            cv2.putText(output_frame, "FPS: {}".format(fps), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # output_frame = cv2.resize(output_frame,(1920,1080),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            stack = np.hstack([frame,output_frame])
            cv2.imshow("Frame", stack) 
            key = cv2.waitKey(1) 
            if key == 27: 
                break # press esc the frame is destroyed
            
    cap.release()
    cv2.destroyAllWindows()
