import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx
mouse = Controller()

app = wx.App(False)
(sx, sy) = wx.GetDisplaySize()
(camx, camy) = (320, 240)

lowerBound = np.array([33, 80, 40])
upperBound = np.array([102, 255, 255])

cam = cv2.VideoCapture(0)

kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))
pinchFlag = 0
openx, openy, openw, openh = (0, 0, 0, 0)

mLocOld = np.array([0, 0])
mouseLoc = np.array([0, 0])
DampingFactor = 2  # value should be greater than 1
# mLocOld = mLocOld+(targetLoc-mLocOld)/DampingFactor
while True:
    ret, img = cam.read()
    img = cv2.resize(img, (340, 220))

    # convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create the Mask
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    # morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    maskFinal = maskClose
    _, conts, _ = cv2.findContours(
        maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if(len(conts) == 2):
        if(pinchFlag == 1):
            pinchFlag = 0
            mouse.release(Button.left)
        x1, y1, w1, h1 = cv2.boundingRect(conts[0])
        x2, y2, w2, h2 = cv2.boundingRect(conts[1])
        cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
        cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
        cx1 = x1+w1/2
        cy1 = y1+h1/2
        cx2 = x2+w2/2
        cy2 = y2+h2/2
        cx = (cx1+cx2)/2
        cy = (cy1+cy2)/2
        cv2.line(img, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
        # mouseLoc = (sx-(cx*sx/camx), cy*sy/camy)
        mouseLoc = mLocOld+((cx, cy)-mLocOld)/DampingFactor
        mouse.position = (sx-(mouseLoc[0]*sx/camx), mouseLoc[1]*sy/camy)
        while mouse.position != (sx-(mouseLoc[0]*sx/camx), mouseLoc[1]*sy/camy):
            pass
        mLocOld = mouseLoc
        openx, openy, openw, openh = cv2.boundingRect(
            np.array([[[x1, y1], [x1+w1, y1+h1], [x2, y2], [x2+w2, y2+h2]]]))
        cv2.rectangle(img, (openx, openy),
                      (openx+openw, openy+openh), (255, 0, 0), 2)

    elif(len(conts) == 1):
        x, y, w, h = cv2.boundingRect(conts[0])
        if(pinchFlag == 0):
            if(abs((w*h-openw*openh)*100/(w*h)) < 30):
                pinchFlag = 1
                mouse.press(Button.left)
                openx, openy, openw, openh = (0, 0, 0, 0)

        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cx = x+w/2
            cy = y+h/2
            cv2.circle(img, (cx, cy), (w+h)/4, (0, 0, 255), 2)
            mouseLoc = mLocOld+((cx, cy)-mLocOld)/DampingFactor
            mouse.position = (
                sx-(mouseLoc[0]*sx/camx), mouseLoc[1]*sy/camy)
            while mouse.position != (sx-(mouseLoc[0]*sx/camx), mouseLoc[1]*sy/camy):
                pass
            mLocOld = mouseLoc
    cv2.imshow("cam", img)
    cv2.waitKey(5)
