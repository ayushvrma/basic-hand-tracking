import cv2
import time
import numpy as np 

import hand_tracking_module as htm

import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


wCam, hCam = 640, 480 #can use 1280, 720



cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]


while True:
    success, img = cap.read()

    img = detector.findHands(img)

    lmlist = detector.findPosition(img, draw= False)
    if lmlist is not None:
        #print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length  = math.hypot(x2-x1,y2-y1)
        # print(length)

        #Hand Range was from 20 to 230 approx
        #Volume Range -65 to 0, so we need to convert these 2 values

        vol = np.interp(length,[20,230],[minVol,maxVol]) #for normalisation

        print(length, vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(30,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)

    cv2.imshow("IMG", img)
    cv2.waitKey(1)
