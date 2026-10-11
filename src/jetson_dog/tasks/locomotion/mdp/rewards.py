"""
Custom rewards for Jetson Dog.

Compatible with Isaac Lab 3.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_posture_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Penalize deviation from the default standing joint positions.

    Default pose:
        shoulder = 0.0
        thigh    = 0.4
        calves   = -0.8

    Reward term:
        sum((q - q_default)^2)

    The reward manager multiplies this result by
    the configured negative weight.
    """

    robot = env.scene["robot"]

    joint_pos = robot.data.joint_pos

    default_joint_pos = robot.data.default_joint_pos

    error = joint_pos - default_joint_pos

    return torch.sum(
        torch.square(error),
        dim=-1,
    )