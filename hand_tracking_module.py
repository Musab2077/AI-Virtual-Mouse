import cv2
import mediapipe as mp
import time
import cvzone
# from mediapipe import 

HANDS=mp.solutions.hands.Hands

class HandDetetor(HANDS):
    def __init__(self,mode=False,max_hands=2,detection_conf=0.5,track_conf=0.5):

        super().__init__()
        self.mp_hands=mp.solutions.hands
        self.mp_draw=mp.solutions.drawing_utils
        self.hands=self.mp_hands.Hands()

    def hands_detection(self,frame,draw=True):
        img_RGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks:
            for self.hand_landmarks in self.results.multi_hand_landmarks:
                    if draw:
                        self.mp_draw.draw_landmarks(frame,self.hand_landmarks,
                                                     self.mp_hands.HAND_CONNECTIONS)
        return frame
    
    def hand_landmark(self,frame,draw=True):
        land_mark_list=[]
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for id, hand_landmark in enumerate(hand_landmarks.landmark):
                    h,w,c=frame.shape
                    cx,cy=round(w*hand_landmark.x),round(h*hand_landmark.y)
                    land_mark_list.append([id,cx,cy])
                    if draw:
                        cv2.circle(frame,(cx,cy),5,
                                (255,0,255),cv2.FILLED)
        return land_mark_list

def main():

    cap=cv2.VideoCapture(0)
    c_time=0
    p_time=0

    while cap.isOpened():
        success,frame=cap.read()
        detector=HandDetetor()
        
        c_time=time.time()
        fps=1/(c_time-p_time)
        p_time=c_time

        frame=detector.hands_detection(frame)
        landmarks=detector.hand_landmark(frame,False)

        if len(landmarks)!=0:
            print(landmarks[4])
        cvzone.putTextRect(frame,f'{int(fps)}',(50,100))

        cv2.imshow('Image',frame)

        if cv2.waitKey(1) & 0XFF==ord('q'):
            break

if __name__=='__main__':
    main()
