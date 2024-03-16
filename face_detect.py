import cv2 as cv
img = cv.imread(r'D:\ML\photos\HEMSWORTH\11.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
harr_cascade = cv.CascadeClassifier('har_face.xml')
face_rect = harr_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=1)
print(len(face_rect))
for (x,y,a,b) in face_rect:
    cv.rectangle(gray,(x,y),(x+a,y+b),(0,255,0),2)
cv.imshow('mogan',gray)
cv.waitKey(0)