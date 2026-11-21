
import cv2
import os
import sys

main_folder = 'datasets'
# Usage: python collect_faces.py <person_name>
sub_folder = sys.argv[1] if len(sys.argv) > 1 else 'papa'

# Load Haar Cascade using OpenCV's built-in path
alg = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar_cas = cv2.CascadeClassifier(alg)

# Check whether Haar Cascade is loaded
if haar_cas.empty():
    print("Error: Haar Cascade not loaded!")
    exit()

print("Haar Cascade loaded successfully!")

width, height = (130, 100)

path = os.path.join(main_folder, sub_folder)

if not os.path.isdir(path):
    os.makedirs(path)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera not opened!")
    exit()

count = 1

while count < 100:
    print(count)

    ret, img = cam.read()

    if not ret:
        print("Cannot read camera frame")
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haar_cas.detectMultiScale(grayImg, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        face = grayImg[y:y+h, x:x+w]

        resizeImg = cv2.resize(face, (width, height))

        cv2.imwrite('%s/%s.png' % (path, count), resizeImg)

        count += 1

    cv2.imshow("Video", img)

    key = cv2.waitKey(10)

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
