# Face Recognition with OpenCV

A simple real-time face recognition system using **OpenCV**, a **Haar Cascade** face detector and the **LBPH** (Local Binary Patterns Histograms) recognizer.

It works in two steps:

1. **Collect** – capture ~100 face images of a person from the webcam.
2. **Recognize** – train an LBPH model on the collected images and identify faces live from the webcam.

## Demo

Watch the project in action: [demo.mp4](demo.mp4)

## Project structure

```
face-recognition-opencv/
├── collect_faces.py      # Step 1: capture face images from webcam
├── recognize_faces.py    # Step 2: train + live recognition
├── demo.mp4              # Demo video
└── README.md
```

The `datasets/` folder is created automatically the first time you run `collect_faces.py`.

## Installation

```bash
git clone https://github.com/Nandhini-1234-03/face-recognition-opencv.git
cd face-recognition-opencv

pip install opencv-contrib-python numpy
```

> `opencv-contrib-python` is required (not plain `opencv-python`) because the
> LBPH recognizer lives in the `cv2.face` contrib module.

## Usage

### 1. Collect training images

```bash
python collect_faces.py 
```

Replace `Isaac Newton` with the person's name. Images are saved to `datasets/<name>/`.
Look at the camera until it reaches 100 images. Press **Esc** to quit early.
Repeat for each person you want to recognize.

### 2. Run recognition

```bash
python recognize_faces.py
```

The script trains on everything in `datasets/` and then opens the webcam.
Known faces are labelled with their name and a confidence score (lower = better match).
Faces with a score of 100 or more are shown as **Unknown**; after a sustained run of unknown
frames, the frame is saved as `input.jpg`. Press **Esc** to quit.

## How it works

- **Detection:** a Haar Cascade (`haarcascade_frontalface_default.xml`, included with OpenCV) finds faces in each frame.
- **Preprocessing:** each face is cropped, converted to grayscale and resized to 130×100.
- **Recognition:** an LBPH model compares the face with the training images.

## Privacy note

The face images in `datasets/` are not included in this repository.
Only collect and use face data with the person's consent otherwise collect it from google.
