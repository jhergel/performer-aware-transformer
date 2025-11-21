# process_video.py
import cv2
from video_loader import VideoLoader
from hand_pos_extractor import *  # whatever you called them
from video_loader import VideoLoader
from hand_model import HandModel
from hand_overlay import draw_full_hand
from mp_hand_tracker import MPHandTracker
from hand_overlay import draw_hands
tracker = MPHandTracker()

loader = VideoLoader("satie.f399.mp4")
for vf in loader:
    frame = vf.image_bgr
    hands = tracker.detect(frame)

    draw_hands(frame, hands)

    cv2.imshow("HandLandmarker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break