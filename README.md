# Face Recognition with OpenCV

A simple real-time face recognition system using **OpenCV**, a **Haar Cascade** face detector and the **LBPH** (Local Binary Patterns Histograms) recognizer.

It works in two steps:

1. **Collect** – capture ~100 face images of a person from the webcam.
2. **Recognize** – train an LBPH model on the collected images and identify faces live from the webcam.

## Project structure

```
face-recognition-opencv/
├── collect_faces.py      # Step 1: capture face images from webcam
├── recognize_faces.py    # Step 2: train + live recognition
├── datasets/             # Face images go here (one sub-folder per person)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/face-recognition-opencv.git
cd face-recognition-opencv

python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

> `opencv-contrib-python` is required (not plain `opencv-python`) because the
> LBPH recognizer lives in the `cv2.face` contrib module.

## Usage

### 1. Collect training images

```bash
python collect_faces.py papa
```

Replace `papa` with the person's name. Images are saved to `datasets/<name>/`.
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

- **Detection:** `haarcascade_frontalface_default.xml` finds faces in each frame.
- **Preprocessing:** each face is cropped, converted to grayscale and resized to 130×100.
- **Recognition:** an LBPH model compares the face with the training images.

## Privacy note

The `datasets/` folder contains photos of real people and is excluded from Git via `.gitignore`.
Only collect and use face data with the person's consent.

## License

MIT – see [LICENSE](LICENSE).
