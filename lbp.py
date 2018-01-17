import numpy as np
import cv2
#from matplotlib import pyplot as plt
from PIL import Image
import sys
from timeit import default_timer as timer
from time import gmtime, strftime


#cap = cv2.VideoCapture("rtsp://192.168.1.199/Streaming/Channels/101")
cap = cv2.VideoCapture("http://admin:admin12345@192.168.1.199/Streaming/Channels/2/picture")
#cap = cv2.VideoCapture(0)

#cap.set(3, 640) #WIDTH
#cap.set(4, 480) #HEIGHT

#face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')

#face_cascade = cv2.CascadeClassifier('cascades/lbpcascade_frontalface.xml')
#face_cascade = cv2.CascadeClassifier('data/data/hogcascade_pedestrians.xml')

#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#face_cascade = cv2.CascadeClassifier('data/data/haarcascades/haarcascade_profileface.xml')
a = 1
t = 0
start = timer()

while(True):
    try:
        end = timer()
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print(time())
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = frame;
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        #print(len(faces))
        # Display the resulting frame       
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = frame[y:y+h, x:x+w]
            print(x,y,w,h)
            #cv2.imwrite('faces/f_'+str(len(faces))+'_'+strftime('%d%m%y_%H%M%S',gmtime())+'.jpg',frame ) 
        


           # eyes = eye_cascade.detectMultiScale(roi_gray)
           # for (ex,ey,ew,eh) in eyes:
           #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


            if w>50 and h>50 and a ==1 and end - start > 0.5:
                crop_img=gray[y:y+h, x:x+w]
                #cv2.imshow('face',crop_img) 
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #break
                start = end
                #cv2.imwrite('faces/face1'+str(t)+'.jpg',crop_img); t=t+1;
                cv2.imwrite('lbpfaces/f_'+str(len(faces))+'_'+strftime('%d%m%y_%H%M%S',gmtime())+'.jpg',crop_img ) 
        cv2.imshow('LBP',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception:
        print("error")
        try:
            #cap = cv2.VideoCapture("rtsp://192.168.1.199/Streaming/Channels/101")
            cap = cv2.VideoCapture(0)
        except Exception:
            print("cant open rtsp")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


