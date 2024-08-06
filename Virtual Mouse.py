import cv2
import pyautogui as pg
import mediapipe as mp
import hand_tracking_module as htm
import math
import numpy as np

detector=htm.HandDetetor(track_conf=0.8)
cap=cv2.VideoCapture(0)

w_scr,h_scr=pg.size()
frame_red=100
px,py,cx,cy=0,0,0,0
smoothening=5

w_cam=640
h_cam=480

cap.set(3,w_cam)
cap.set(4,h_cam)

while cap.isOpened():
    success,frame=cap.read()
    # flip_frame=cv2.flip(frame,1)

    frame=detector.hands_detection(frame)
    landmarks=detector.hand_landmark(frame,0)
    ## Building a rectangle so that it interpolates with the mouse
    cv2.rectangle(frame,(frame_red,frame_red),
                  (w_cam-frame_red,h_cam-frame_red),(255,0,0),5)

    if len(landmarks)!=0:
        x8,y8=landmarks[8][1],landmarks[8][2] # landmarks of tip of index finger
        x12,y12=landmarks[12][1],landmarks[12][2] # landmarks of tip of middle finger


        cv2.circle(frame,(x8,y8),10,(255,0,0),cv2.FILLED) # Making circle on tip of index finger
        cv2.circle(frame,(x12,y12),10,(255,0,0),cv2.FILLED) # Making circle on tip of middle finger 
        
        if y12 <= y8 :
            midx,midy=(x8+x12)/2 , (y8+y12)/2
            finger_dist=math.hypot(x8-x12,y8-y12) # Distance between two fingers

            cv2.line(frame,(x8,y8),(x12,y12),(0,0,255),5)
            cv2.circle(frame,(int(midx),int(midy)),
                       10,(255,0,255),cv2.FILLED)

            if finger_dist< 50:
                cv2.circle(frame,(int(midx),int(midy)),
                       7,(0,255,0),cv2.FILLED)
                mouse_x,mouse_y=pg.position()
                pg.leftClick(mouse_x,mouse_y)

        elif y8<y12:
            x_interp=np.interp(x8,(frame_red,w_cam-frame_red),(0,w_scr))
            y_interp=np.interp(y8,(frame_red,h_cam-frame_red),(0,h_scr))

            # Smoothening the values
            cx=px + (x_interp - px)/smoothening
            cy=py + (y_interp - py)/smoothening

            pg.moveTo(w_scr-cx,cy)

    px,py=cx,cy

    cv2.imshow('Virtual mouse',frame)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()