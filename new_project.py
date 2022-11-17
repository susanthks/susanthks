import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
path='project/images'
images=[]
classnames=[]
mylist=os.listdir(path)
for cl in mylist:
    currentimage=cv2.imread(f'{path}/{cl}')
    images.append(currentimage)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)
def findencodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
def markattendance(name):
    with open('project/attendance.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

encodelistknownfaces=findencodings(images)
print(encodelistknownfaces)
cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgsmall=cv2.resize(img,(0,0),None,0.25,0.25)
    faceInFrame=face_recognition.face_locations(imgsmall)
    faceEncode=face_recognition.face_encodings(imgsmall,faceInFrame)
    # print(faceEncode)
    for faceencode,faceloc in zip(faceEncode,faceInFrame):
        matches=face_recognition.compare_faces(encodelistknownfaces,faceencode)
        faceDistance=face_recognition.face_distance(encodelistknownfaces,faceencode)
        print('matches:',matches)
        # print('face distance:',faceDistance)
        matchindex=np.argmin(faceDistance)
        # print('matchindex:',matchindex)
        if matches[matchindex]:
            name=classnames[matchindex].capitalize()
            print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)
    cv2.imshow('face recognition',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break