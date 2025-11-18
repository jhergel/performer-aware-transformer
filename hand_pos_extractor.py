import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands

FINGERS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20
}


class HandPositionExtractor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, frame):
        """
        Input: BGR frame (OpenCV)
        Output: dict with LH and RH data
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return {"LH": None, "RH": None}

        hands = []
        for hand_lm, handedness in zip(result.multi_hand_landmarks,
                                       result.multi_handedness):

            label = handedness.classification[0].label  # "Left" or "Right"
            pts = {}

            for name, idx in FINGERS.items():
                lm = hand_lm.landmark[idx]
                pts[name] = (int(lm.x * w), int(lm.y * h))

            wrist_lm = hand_lm.landmark[0]
            pts["wrist"] = (int(wrist_lm.x * w), int(wrist_lm.y * h))

            hands.append((label, pts, hand_lm))

        # Normalize to LH/RH keys
        out = {"LH": None, "RH": None}
        for label, pts, lm in hands:
            if label == "Left":
                out["LH"] = {"fingers": pts, "raw": lm}
            else:
                out["RH"] = {"fingers": pts, "raw": lm}

        return out


def draw_hand_points(frame, hands):
    for side in ["LH", "RH"]:
        if hands[side] is None:
            continue

        for name, (x,y) in hands[side]["fingers"].items():
            cv2.circle(frame, (x,y), 6, (0,255,0), -1)
            cv2.putText(frame, f"{side}-{name}", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
