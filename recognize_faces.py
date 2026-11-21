
import cv2
import os
import numpy

main_folder = 'datasets'

print("Training....")

(images, labels, names, id) = ([], [], {}, 0)

# Read dataset
for root, dirs, files in os.walk(main_folder):

    for subdir in dirs:

        names[id] = subdir
        subjectpath = os.path.join(root, subdir)

        for filename in os.listdir(subjectpath):

            path = os.path.join(subjectpath, filename)

            image = cv2.imread(path, 0)

            if image is not None:
                image = cv2.resize(image, (130, 100))
                images.append(image)
                labels.append(id)

        id += 1

# Check dataset
if len(images) == 0:
    print("No training images found!")
    exit()

images = numpy.array(images, dtype=numpy.uint8)
labels = numpy.array(labels, dtype=numpy.int32)

print("Total training images:", len(images))
print("Persons:", names)

# Train LBPH model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

print("Training completed!")

# Load Haar Cascade
alg = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar_cas = cv2.CascadeClassifier(alg)

if haar_cas.empty():
    print("Haar Cascade not loaded!")
    exit()

width, height = (130, 100)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera not opened!")
    exit()

cnt = 0

while True:

    ret, img = cam.read()

    if not ret:
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haar_cas.detectMultiScale(grayImg, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 225, 0), 2)

        face = grayImg[y:y+h, x:x+w]
        resizeImg = cv2.resize(face, (width, height))

        prediction = model.predict(resizeImg)

        if prediction[1] < 100:

            cv2.putText(
                img,
                '%s-%.0f' % (names[prediction[0]], prediction[1]),
                (x-10, y-10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (54, 67, 98),
                2
            )

            print(names[prediction[0]])
            cnt = 0

        else:

            cnt += 1

            cv2.putText(
                img,
                "Unknown",
                (x-10, y-10),
                cv2.FONT_HERSHEY_PLAIN,
                0.5,
                (255, 0, 0),
                2
            )

            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("input.jpg", img)
                cnt = 0

    cv2.imshow("Output Screen", img)

    key = cv2.waitKey(10)

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
