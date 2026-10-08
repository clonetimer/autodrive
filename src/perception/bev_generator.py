"""Five-channel ego-aligned semantic BEV, deterministic on CPU.

Channels: drivable corridor, boundaries, route, other vehicles, ego.
Pixel axes: forward = up, left = left. Metric extent: [-12, 52] m forward,
[-24, 24] m laterally. Ground-truth geometry is intentionally explicit.
"""
from __future__ import annotations
import cv2
import numpy as np
from src.envs.observation import Snapshot, to_local


def local_to_pixels(points: np.ndarray, size: int) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    cols = (24 - y) / 48 * (size - 1)
    rows = (52 - x) / 64 * (size - 1)
    return np.rint(np.column_stack((cols, rows))).astype(np.int32)


def rasterize(snapshot: Snapshot, size: int = 48) -> np.ndarray:
    if size < 32:
        raise ValueError('BEV must be at least 32x32')
    image = np.zeros((5, size, size), dtype=np.uint8)
    ego = snapshot.ego

    def pixels(points):
        return local_to_pixels(to_local(points, ego), size)

    if len(snapshot.boundaries) >= 2:
        polygon = np.concatenate((snapshot.boundaries[0], snapshot.boundaries[-1][::-1]))
        cv2.fillPoly(image[0], [pixels(polygon)], 255)
    for boundary in snapshot.boundaries:
        cv2.polylines(image[1], [pixels(boundary)], False, 255, 1)
    cv2.polylines(image[2], [pixels(snapshot.route)], False, 255, 1)
    for vehicle in snapshot.vehicles:
        if vehicle.id != ego.id and vehicle.active:
            cv2.fillPoly(image[3], [pixels(vehicle.corners())], 255)
    cv2.fillPoly(image[4], [pixels(ego.corners())], 255)
    return image


def semantic_rgb(bev: np.ndarray) -> np.ndarray:
    """For a demo only; training uses independent binary channels."""
    # Semantic palette is part of the simulator, not a statistical chart.
    rgb = np.zeros((*bev.shape[1:], 3), dtype=np.uint8)
    palette = [(62, 65, 72), (210, 210, 210), (220, 170, 40), (205, 77, 65), (65, 160, 230)]
    for mask, color in zip(bev, palette):
        rgb[mask > 0] = color
    return rgb
