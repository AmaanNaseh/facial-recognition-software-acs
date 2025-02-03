#Importing libraries to use in code.
import cv2
import os
import imutils
import time
import csv

#Initializing dataset & data-entries of students
dataset = 'dataset'
name = str(input("Name : "))
roll_no = int(input("Roll No. : "))

#Getting the path/Creating the path to save student's image data
path = os.path.join(dataset,name)
if not os.path.isdir(path):
    os.makedirs(path)
    print(name)

#Loading our algorithm file for detecting face
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

#Creating an excel file to save registered candidates
info = [str(name), str(roll_no)]
with open('student.csv', 'a') as csvFile1:
    write = csv.writer(csvFile1)
    write.writerow(info)
csvFile1.close()

#Initializing Inbuilt camera of Laptop
##We can use our mobile too for this purpose, the code is commented out in the end of this code
print("Starting video stream....")
cam = cv2.VideoCapture(0) #Change 0 to 1 or 2 (hit & trial) if you are using external Camera like USB or WebCam
time.sleep(1)

#Setting frame size
(width,height) = (130,100)
count = 1

# main body of code where we have used imutils to resize image & OpenCV to convert image into
## grayscale(black and white) for easy recognition purpose & then saving it into dataset directory
while count < 31:
    print(count)
    _,img = cam.read()

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        faceOnly = grayImg[y:y+h,x:x+w]
        resizeImg = cv2.resize(faceOnly, (width,height))
        cv2.imwrite("%s/%s.jpg"%(path,count), resizeImg)
        count += 1
    cv2.imshow("Face_Detection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

#Concluding the code by closing respective cam & windows
print("Image Captured Successfully")
cam.release()
cv2.destroyAllWindows

#Code for using Mobile Camera for face capturing is below :
"""
import cv2
import os
import time
import csv
import numpy as np
import imutils
import urllib.request

dataset = 'dataset'
name = str(input("Name : "))
roll_no = int(input("Roll No. : "))

path = os.path.join(dataset,name)
if not os.path.isdir(path):
    os.makedirs(path)
    print(name)

alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

info = [str(name), str(roll_no)]
with open('student2.csv', 'a') as csvFile1:
    write = csv.writer(csvFile1)
    write.writerow(info)
csvFile1.close()

#cam = cv2.VideoCapture(0)
print("Starting video stream....")
# Download IP WebCam App in Phone
## Select Start Server & Copy paste the IPV4 address below in url like shown :
### url = 'YOUR LINK/shot,jpg'
#### Example is :- (re edit this line of code below for your IP Address)
url = 'http://192.168.0.102:8080/shot.jpg'

(width,height) = (130,100)
count = 1

while count < 31:
    print(count)
    #_,img = cam.read()
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    #_,img = cam.read()
    img = cv2.imdecode(imgNp, -1)
    img = imutils.resize(img, width=450)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        faceOnly = grayImg[y:y+h,x:x+w]
        resizeImg = cv2.resize(faceOnly, (width,height))
        cv2.imwrite("%s/%s.jpg"%(path,count), resizeImg)
        count += 1
    cv2.imshow("Face_Detection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

print("Image Captured Successfully")
#cam.release()
cv2.destroyAllWindows 
"""