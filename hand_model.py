import numpy as np
from dataclasses import dataclass

# Mediapipe indices per finger
FINGER_JOINTS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

@dataclass
class HandModel:
    """Full hand representation for a single frame."""
    keypoints: dict        # {0: (x,y), 1:(x,y), ...}
    handedness: str        # "Left" or "Right"

    # --- Accessors ---
    def fingertip(self, name):
        idx = FINGER_JOINTS[name][-1]
        return self.keypoints[idx]

    def joints(self, name):
        return [self.keypoints[idx] for idx in FINGER_JOINTS[name]]

    def wrist(self):
        return self.keypoints[0]

    # --- Geometry helpers ---
    def finger_direction(self, name):
        """Unit direction vector from MCP to tip."""
        pts = self.joints(name)
        v = np.array(pts[-1]) - np.array(pts[0])
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    def palm_normal(self):
        """
        Compute palm orientation using (index_mcp - wrist) x (pinky_mcp - wrist).
        """
        w = np.array(self.keypoints[0])

        index_mcp = np.array(self.keypoints[5])
        pinky_mcp = np.array(self.keypoints[17])

        v1 = index_mcp - w
        v2 = pinky_mcp - w
        n = np.cross(np.append(v1, 0), np.append(v2, 0))
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            return np.zeros(3)
        return n / n_norm

    def bounding_box(self):
        pts = np.array(list(self.keypoints.values()))
        x1 = int(np.min(pts[:,0]))
        y1 = int(np.min(pts[:,1]))
        x2 = int(np.max(pts[:,0]))
        y2 = int(np.max(pts[:,1]))
        return (x1, y1, x2, y2)
