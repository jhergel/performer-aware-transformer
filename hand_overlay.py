import cv2
import mediapipe as mp
import numpy as np

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

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):

                    # Draw skeleton
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)

                    # Draw fingertips
                    draw_fingertips(frame, hand_landmarks, handedness)

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
