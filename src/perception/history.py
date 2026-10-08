"""Past-only semantic BEV history with ego-motion compensation and validity.

Each physical step captures once. Reading an observation never advances time.
Uses simulator poses, not a learned localization system. Warping cannot recover
objects which were outside the old field of view; zero remains unknown/empty.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
import cv2
import numpy as np
from src.envs.observation import Snapshot
from src.perception.bev_generator import rasterize


@dataclass
class Frame:
    image: np.ndarray
    pose: tuple[float, float, float]
    time_s: float


def align_frame(image: np.ndarray, old_pose, new_pose) -> np.ndarray:
    """Transform pixels via old local -> world -> current local, nearest neighbor."""
    size = image.shape[-1]
    if tuple(old_pose) == tuple(new_pose):
        return image.copy()
    source = np.array([[0, 0], [size - 1, 0], [0, size - 1]], dtype=np.float32)
    local = np.column_stack((52 - source[:, 1] * 64 / (size - 1),
                             24 - source[:, 0] * 48 / (size - 1)))
    c, s = math.cos(old_pose[2]), math.sin(old_pose[2])
    world = local @ np.array([[c, s], [-s, c]]) + np.asarray(old_pose[:2])
    c, s = math.cos(new_pose[2]), math.sin(new_pose[2])
    current = (world - np.asarray(new_pose[:2])) @ np.array([[c, -s], [s, c]])
    target = np.column_stack(((24 - current[:, 1]) / 48 * (size - 1),
                              (52 - current[:, 0]) / 64 * (size - 1))).astype(np.float32)
    affine = cv2.getAffineTransform(source, target)
    return np.stack([cv2.warpAffine(channel, affine, (size, size), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) for channel in image])


class BEVHistory:
    def __init__(self, count: int, size: int, dt: float):
        if count < 1 or count > 8 or size < 32 or dt <= 0:
            raise ValueError('Invalid BEV history dimensions or time interval')
        self.count, self.size, self.dt = count, size, dt
        self.frames: deque[Frame] = deque(maxlen=count)

    def capture(self, snapshot: Snapshot, time_s: float) -> None:
        if self.frames and time_s <= self.frames[-1].time_s:
            raise ValueError('Capture must occur once per increasing physical timestamp')
        ego = snapshot.ego
        self.frames.append(Frame(rasterize(snapshot, self.size), (ego.x, ego.y, ego.yaw), float(time_s)))

    def observation(self, snapshot: Snapshot) -> tuple[np.ndarray, np.ndarray]:
        if not self.frames:
            raise RuntimeError('Capture a frame before observing')
        image = np.zeros((self.count * 5, self.size, self.size), np.uint8)
        validity = np.zeros((self.count, 2), np.float32)
        current_time = self.frames[-1].time_s
        ego = snapshot.ego
        pose = (ego.x, ego.y, ego.yaw)
        for i, frame in enumerate(reversed(self.frames)):
            image[5 * i:5 * (i + 1)] = align_frame(frame.image, frame.pose, pose)
            validity[i] = [min(1.0, (current_time - frame.time_s) / max(self.dt * (self.count - 1), self.dt)), 1.0]
        return image, validity.reshape(-1)
