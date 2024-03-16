import numpy as np
import cv2 as cv
harr_casscade = cv.CascadeClassifier('har_face.xml')
people = ["EVANS","HEMSWORTH","PRABHAS","RDJ","VIJAY"]
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
img = cv.imread(r'D:\practice\t.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('person',gray)
faces_rect = harr_casscade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3)
for (x,y,z,w) in faces_rect:
    faces_roi = gray[y:y+w,x:x+z]
    label,confidence = face_recognizer.predict(faces_roi)
    print(label)
    print (confidence)
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=3)
    cv.rectangle(img,(x,y),(x+z,y+w),(0,255,0),thickness=2)
cv.imshow('detected',img)
cv.waitKey(0)