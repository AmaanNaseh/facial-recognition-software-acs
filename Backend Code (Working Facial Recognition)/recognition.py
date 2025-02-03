# Importing Libraries.
import cv2, os, numpy, csv
#Loading Haar Cascade Algorithm for face detection.
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

#Initializing dataset folder to check images from.
datasets = 'dataset'
print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

#Tracing the path to obtain trained images for face detection
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subpath):
            path = subpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

#Using array for better data processing
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)

#Setting frame size
(width, height) = (130, 100)

#Using face recognizer models in CV2, multiple syntax & models are commented out below for ease
model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face_LBPHFaceRecognizer.create()
#model = cv2.face.createLBPHFaceRecognizer()
#model = cv2.face.FisherFaceRecognizer_create()
haar_cascade = cv2.CascadeClassifier(alg)
#Training model
model.train(images, labels)

#Initializing in-built camera, COde for using  Mobile Camera is in end.
cam = cv2.VideoCapture(0) #Change 0 to 1 or 2 (hit & trial) if you are using external Camera like USB or WebCam
cnt = 0

# main body of code where we have used imutils to resize image & OpenCV
## to recognize face from images stored in respective dataset directory.
while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        face = grayImg[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

        if prediction[1]<800:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]],prediction[1]), (x-10,y-10), cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
            print(names[prediction[0]])
            cnt = 0

        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x-10,y-10), cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
            if (cnt > 100):
                print("Unknown Person")
                cv2.imwrite('unknwon.jpg', img)
                cnt = 0

    cv2.imshow("Face_Recognition", img)
    key = cv2.waitKey(10)
    if key == 27:
        break
    
#Concluding the code by closing respective cam & windows
cam.release()
cv2.destroyAllWindows

#Code for using Mobile Camera for face recognition is below :
"""
import cv2, os, numpy
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

datasets = 'dataset'
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subpath):
            path = subpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)

(width, height) = (130, 100)

model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face_LBPHFaceRecognizer.create()
#model = cv2.face.createLBPHFaceRecognizer()
#model = cv2.face.FisherFaceRecognizer_create()
haar_cascade = cv2.CascadeClassifier(alg)
model.train(images, labels)

#cam = cv2.VideoCapture(0)
# Download IP WebCam App in Phone
## Select Start Server & Copy paste the IPV4 address below in url like shown :
### url = 'YOUR LINK/shot,jpg'
#### Example is :- (re edit this line of code below for your IP Address)
url = 'http://192.168.0.102:8080/shot.jpg'
cnt = 0

while True:
    #_,img = cam.read()
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        face = grayImg[img, (x,y), (x+w,y+h), (255,255,0), 2]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

        if prediction[1]<800:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]],prediction[1]), (x-10,y-10), cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
            print(names[prediction[0]])
            cnt = 0
            with open('attendence.csv', 'a') as csvFile2:
                write = csv.writer(csvFile2)
                write.writerow(names)
            csvFile2.close()
        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x-10,y-10), cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
            if (cnt > 100):
                print("Unknown Person")
                cv2.imwrite('unknwon.jpg', img)
                cnt = 0

    cv2.imshow("Face_Recognition", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

#cam.release()
#cv2.destroyAllWindows  
"""