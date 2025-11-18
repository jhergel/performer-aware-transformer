# process_video.py
import cv2
from video_loader import VideoLoader
from hands import HandPositionExtractor, draw_hand_points  # whatever you called them

def main(video_path: str):
    loader = VideoLoader(video_path)   # optionally resize_to=(1280, 720)
    extractor = HandPositionExtractor()

    for vf in loader:
        frame = vf.image_bgr
        hands = extractor.extract(frame)

        # draw overlay for debug
        # e.g. function we previously wrote:
        # draw_hand_points(frame, hands)
        if hands["LH"] is not None or hands["RH"] is not None:
            # simple debug: draw fingertips
            for side in ("LH", "RH"):
                if hands[side] is None:
                    continue
                for name, (x, y) in hands[side]["fingers"].items():
                    cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                    cv2.putText(frame, f"{side}-{name}", (x+5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # show timestamp
        cv2.putText(frame, f"t={vf.time_sec:.3f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Video + Hands", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break

    loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
