import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    success , img  = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for handLMS in result.multi_hand_landmarks:
            for id ,lm in enumerate(handLMS.landmark):
                # print(id , lm)
                h,w,c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                print(id , cx , cy)
                # if id==0:
                cv2.putText(img , f'({cx} , {cy})' , (cx,cy) , cv2.FONT_HERSHEY_COMPLEX , 0.5 , (200,0,100) , 1 )

            mpDraw.draw_landmarks(img , handLMS , mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    # cv2.putText(img , str(round(fps)) , (10,70) , cv2.FONT_HERSHEY_COMPLEX , 3 , (255,0,255) , 3)



    cv2.imshow("Image" , img)
    cv2.waitKey(1)