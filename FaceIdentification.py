'''
OPENCV FACE AND EYE CLASSIFIER
'''

import cv2

vid_cap = cv2.VideoCapture(0)
# Importing the self-taught face and eye classfier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, webcam = vid_cap.read()
    width = int(vid_cap.get(3))
    height = int(vid_cap.get(4))
    # Requires a grayscale image to perform the classification
    gray = cv2.cvtColor(webcam, cv2.COLOR_BGR2GRAY)
    # By scaling the faces in a scale pyramid fashion, we are able to be detected
    # Recommended scale values: 1.05(more reliable)-1.4(faster detection)
    # Min neighbours recommended 3-6. Higher val gives higher quality but less detections
    faces = face_cascade.detectMultiScale(gray, 1.3, 6)
    for (fx, fy, fw, fh) in faces:
        # Draw the face identifying rectangles around the face on the webcam frame (BLUE)
        cv2.rectangle(webcam, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 7)
        # ROI stands for the region of interest for the eyes
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        # Need to translate the area of interest for the eyes on the webcam frame
        roi_color = webcam[fy:fy + fh, fx:fx + fw]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 6)
        for (ex, ey, ew, eh) in eyes:
            # Draw the eye identifying rectangles around the eyes on the frame ROI (RED)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 5)

        font = cv2.FONT_HERSHEY_SIMPLEX
        webcam = cv2.putText(webcam, 'FACE AND EYE DETECTION', (10, height-10), font, 4, (0, 165, 255), 10, cv2.LINE_AA)

    cv2.imshow('Webcam', webcam)

    if cv2.waitKey(1) == ord('q'):
        break

vid_cap.release()
cv2.destroyAllWindows()