import cv2
import time
import numpy as np 

import hand_tracking_module as htm


wCam, hCam = 640, 480 #can use 1280, 720



cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(detectionCon=0.7)



while True:
    success, img = cap.read()

    img = detector.findHands(img)

    lmlist = detector.findPosition(img, draw= False)
    if lmlist is not None:
        print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]

        cv2.circle(img, (x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(30,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)

    cv2.imshow("IMG", img)
    cv2.waitKey(1)
