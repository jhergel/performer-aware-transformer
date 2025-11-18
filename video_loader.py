import cv2
from dataclasses import dataclass
from typing import Iterator, Optional

@dataclass
class VideoFrame:
    index: int
    time_sec: float
    image_bgr: "cv2.Mat" 

class VideoLoader:
    def __init__(self, path: str, resize_to: Optional[tuple[int, int]] = None):
        """
        path: path to video file
        resize_to: (width, height) if you want to resize frames (optional)
        """
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.resize_to = resize_to

    def __iter__(self) -> Iterator[VideoFrame]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.resize_to is not None:
                frame = cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_AREA)
            t = idx / self.fps
            yield VideoFrame(index=idx, time_sec=t, image_bgr=frame)
            idx += 1

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
