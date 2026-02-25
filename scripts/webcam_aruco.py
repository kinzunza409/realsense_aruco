import numpy as np
import cv2 as cv

# aruco detection initializations
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
aruco_params = cv.aruco.DetectorParameters()
aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # aruco detection
    corners, ids, rejected = aruco_detector.detectMarkers(frame)

    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
    else:
        cv.putText(frame, 'NO MARKERS DETECTED!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


    
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()