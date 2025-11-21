import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker

from hand_model import HandModel, FINGER_JOINTS
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Fingertip indices in MediaPipe
FINGERTIPS = [4, 8, 12, 16, 20]

COLORS = {
    "thumb":  (0, 255, 255),
    "index":  (255, 0, 0),
    "middle": (0, 255, 0),
    "ring":   (0, 128, 255),
    "pinky":  (128, 0, 255)
}

def draw_fingertips(image, hand_landmarks, handedness):
    h, w, _ = image.shape

    color_index = {
        4: COLORS["thumb"],
        8: COLORS["index"],
        12: COLORS["middle"],
        16: COLORS["ring"],
        20: COLORS["pinky"]
    }

    for idx in FINGERTIPS:
        lm = hand_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image, (x, y), 8, color_index[idx], -1)

    # Label hand
    label = handedness[0].classification[0].label
    wrist = hand_landmarks.landmark[0]
    wx, wy = int(wrist.x * w), int(wrist.y * h)
    cv2.putText(image, label, (wx - 20, wy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,255,255), 2, cv2.LINE_AA)


def draw_full_hand(frame, hand_model: HandModel):
    if hand_model is None:
        return

    # draw all joints
    for idx, (x,y) in hand_model.keypoints.items():
        cv2.circle(frame, (x,y), 4, (0,255,0), -1)
        cv2.putText(frame, str(idx), (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

    # draw finger names
    for finger in FINGER_JOINTS:
        joints = hand_model.joints(finger)
        for x,y in joints:
            cv2.circle(frame, (x,y), 6, (0,128,255), 2)
        tip = hand_model.fingertip(finger)
        cv2.putText(frame, finger, (tip[0]+4, tip[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    # draw wrist
    wx, wy = hand_model.wrist()
    cv2.circle(frame, (wx,wy), 8, (255,0,0), -1)
    cv2.putText(frame, hand_model.handedness, (wx-20, wy-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


# Mediapipe hand connections (21 joints)
MP_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

def draw_hands(frame, hands):
    for hand in hands:
        pts = hand["landmarks"]

        # Draw joints
        for (x,y) in pts:
            cv2.circle(frame, (x,y), 4, (0,255,0), -1)

        # Draw skeleton connections
        for a,b in MP_CONNECTIONS:
            (x1,y1) = pts[a]
            (x2,y2) = pts[b]
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # Label hand
        (wx, wy) = pts[0]
        cv2.putText(frame, hand["handedness"], (wx-20, wy-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,255,255), 2)
