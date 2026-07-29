"""H-frame 扰动模板流程共用的旋转与环形信号工具。"""

import numpy as np


def normalize_quaternions_wxyz(quaternions):
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim == 1:
        quaternions = quaternions.reshape(1, 4)
        squeeze = True
    else:
        squeeze = False
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("四元数必须为 shape=(4,) 或 (N,4)，顺序为 wxyz。")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if (
        np.any(norms < 1e-12)
        or not np.all(np.isfinite(quaternions))
    ):
        raise ValueError("四元数包含零范数、NaN 或 Inf。")
    normalized = quaternions / norms
    return normalized[0] if squeeze else normalized


def markley_quaternion_mean_wxyz(quaternions):
    """在 SO(3) 上做对 q/-q 符号不敏感的四元数均值。"""
    quaternions = normalize_quaternions_wxyz(quaternions)
    if quaternions.ndim != 2 or len(quaternions) == 0:
        raise ValueError("Markley 均值至少需要一个四元数。")
    scatter = np.einsum("ni,nj->ij", quaternions, quaternions)
    _, eigenvectors = np.linalg.eigh(scatter)
    mean = eigenvectors[:, -1]
    if np.dot(mean, quaternions[0]) < 0.0:
        mean = -mean
    return mean / np.linalg.norm(mean)


def align_quaternion_sequence_wxyz(quaternions):
    aligned = normalize_quaternions_wxyz(quaternions).copy()
    if aligned.ndim != 2:
        raise ValueError("四元数序列必须为 shape=(N,4)。")
    if aligned[0, 0] < 0.0:
        aligned[0] *= -1.0
    for i in range(1, len(aligned)):
        if np.dot(aligned[i - 1], aligned[i]) < 0.0:
            aligned[i] *= -1.0
    return aligned


def quaternion_wxyz_to_rotmat(quaternions):
    quaternions = normalize_quaternions_wxyz(quaternions)
    squeeze = quaternions.ndim == 1
    if squeeze:
        quaternions = quaternions.reshape(1, 4)
    w, x, y, z = quaternions.T
    rotations = np.empty((len(quaternions), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations[0] if squeeze else rotations


def rotmat_to_quaternion_wxyz(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)
    squeeze = rotations.ndim == 2
    if squeeze:
        rotations = rotations.reshape(1, 3, 3)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("旋转矩阵必须为 shape=(3,3) 或 (N,3,3)。")
    quaternions = np.empty((len(rotations), 4), dtype=np.float64)
    for i, rotation in enumerate(rotations):
        if (
            not np.all(np.isfinite(rotation))
            or not np.allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-7
            )
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)
        ):
            raise ValueError("输入包含无效旋转矩阵。")
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = 2.0 * np.sqrt(
                    1.0 + rotation[0, 0]
                    - rotation[1, 1] - rotation[2, 2]
                )
                quaternion = np.array(
                    [
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = 2.0 * np.sqrt(
                    1.0 + rotation[1, 1]
                    - rotation[0, 0] - rotation[2, 2]
                )
                quaternion = np.array(
                    [
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                    ]
                )
            else:
                scale = 2.0 * np.sqrt(
                    1.0 + rotation[2, 2]
                    - rotation[0, 0] - rotation[1, 1]
                )
                quaternion = np.array(
                    [
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        quaternions[i] = quaternion / np.linalg.norm(quaternion)
    quaternions = align_quaternion_sequence_wxyz(quaternions)
    return quaternions[0] if squeeze else quaternions


def circular_moving_average(values, window_size):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("环形滑动平均要求 shape=(N,D)。")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size 必须为正奇数。")
    if window_size == 1:
        return values.copy()
    pad = window_size // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]], axis=0)
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.column_stack(
        [
            np.convolve(padded[:, i], kernel, mode="valid")
            for i in range(values.shape[1])
        ]
    )


def circular_quaternion_moving_average(quaternions, window_size):
    quaternions = align_quaternion_sequence_wxyz(quaternions)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size 必须为正奇数。")
    if window_size == 1:
        return quaternions.copy()
    pad = window_size // 2
    smoothed = np.empty_like(quaternions)
    for i in range(len(quaternions)):
        indices = np.arange(i - pad, i + pad + 1) % len(quaternions)
        smoothed[i] = markley_quaternion_mean_wxyz(
            quaternions[indices]
        )
    return align_quaternion_sequence_wxyz(smoothed)


def circular_central_difference(values, dt):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or dt <= 0.0:
        raise ValueError("环形中心差分要求 shape=(N,D) 且 dt>0。")
    return (
        np.roll(values, -1, axis=0)
        - np.roll(values, 1, axis=0)
    ) / (2.0 * dt)


def rotation_z(angle):
    angle = np.asarray(angle, dtype=np.float64)
    flat = angle.reshape(-1)
    rotations = np.zeros((len(flat), 3, 3), dtype=np.float64)
    cosine = np.cos(flat)
    sine = np.sin(flat)
    rotations[:, 0, 0] = cosine
    rotations[:, 0, 1] = -sine
    rotations[:, 1, 0] = sine
    rotations[:, 1, 1] = cosine
    rotations[:, 2, 2] = 1.0
    return rotations[0] if angle.ndim == 0 else rotations.reshape(
        angle.shape + (3, 3)
    )


def rotation_to_rpy(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)
    squeeze = rotations.ndim == 2
    if squeeze:
        rotations = rotations.reshape(1, 3, 3)
    rpy = np.column_stack(
        [
            np.arctan2(rotations[:, 2, 1], rotations[:, 2, 2]),
            np.arcsin(np.clip(-rotations[:, 2, 0], -1.0, 1.0)),
            np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]),
        ]
    )
    return rpy[0] if squeeze else rpy


def rotation_geodesic_angle(rotation_a, rotation_b):
    rotation_a = np.asarray(rotation_a, dtype=np.float64)
    rotation_b = np.asarray(rotation_b, dtype=np.float64)
    relative = np.swapaxes(rotation_a, -1, -2) @ rotation_b
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.arccos(cosine)
