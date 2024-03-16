import os
import cv2 as cv
import numpy as np
people = ["EVANS","HEMSWORTH","PRABHAS","RDJ","VIJAY"]
harr_cascade = cv.CascadeClassifier('har_face.xml')
dir = r'D:\ML\photos'
features = []
labels = []
def create_train():
    for person in people:
        path = os.path.join(dir,person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect = harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
            for (x,y,z,w) in faces_rect:
                faces_roi = gray[y:y+w,x:x+z]
                features.append(faces_roi)
                labels.append(label)
create_train()
print(len(features))
print(len(labels))
features = np.array(features,dtype='object')
labels = np.array(labels)
face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.train(features,labels)
np.save('features.npy',features)
np.save('labels.npy',labels)
face_recogniser.save('face_trained.yml')