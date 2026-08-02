"""MPC 预测运动学的可替换模型后端。"""

from .base import PredictionKinematics, PredictionKinematicsBackend
from .factory import create_prediction_backend
from .mujoco_backend import MujocoPredictionBackend
from .pinocchio_backend import (
    PinocchioPredictionBackend,
    resolve_robot_mjcf_path,
)

__all__ = (
    "PredictionKinematics",
    "PredictionKinematicsBackend",
    "MujocoPredictionBackend",
    "PinocchioPredictionBackend",
    "create_prediction_backend",
    "resolve_robot_mjcf_path",
)
