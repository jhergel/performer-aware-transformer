import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

HAND_MODEL_URL = (
"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def download_model_if_needed(local_path="models/hand_landmarker.task"):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.isfile(local_path):
        print(f"Downloading hand_landmarker.model to {local_path} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, local_path)
        print("Download complete.")
    return local_path

class MPHandTracker:
    def __init__(self):
        BaseOptions = mp_tasks.BaseOptions

        model_path = download_model_if_needed()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,      # or VIDEO
            num_hands=2,
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        h, w, _ = frame_bgr.shape

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.landmarker.detect(mp_image)

        hands = []
        if results and results.hand_landmarks:
            for i, lm_list in enumerate(results.hand_landmarks):
                hand_data = {
                    "handedness": results.handedness[i][0].category_name,
                    "landmarks": []
                }

                for lm in lm_list:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    hand_data["landmarks"].append((x, y))

                hands.append(hand_data)

        return hands
