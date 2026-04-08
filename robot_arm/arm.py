"""
RobotArm: 3-DoF arm in PyBullet.

Arm geometry (custom URDF):

    ee_link (sphere)        <- end-effector tip
       |
    link_3 (0.3m forearm)
       |  joint_2 (elbow pitch, Y-axis)
    link_2 (0.3m upper arm)
       |  joint_1 (shoulder pitch, Y-axis)
    link_1 (0.1m connector)
       |  joint_0 (base yaw, Z-axis)
    base_link (fixed to ground)

Total reach: ~0.7m from base (0.1 + 0.3 + 0.3)
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pybullet as p


URDF_PATH = str(Path(__file__).parent / "robot_arm.urdf")


def _urdf_hash() -> str:
    with open(URDF_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


class RobotArm:
    ACTIVE_JOINTS = [0, 1, 2]  # joint_0, joint_1, joint_2
    NUM_JOINTS = 3
    EE_LINK_INDEX = 3  # ee_link (link1=0, link2=1, link3=2, ee=3)

    def __init__(self, client: int):
        self.client = client
        self.arm_id = p.loadURDF(
            URDF_PATH,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client,
        )
        self.urdf_hash = _urdf_hash()

        # Read joint limits from URDF
        self.joint_lower = np.zeros(self.NUM_JOINTS)
        self.joint_upper = np.zeros(self.NUM_JOINTS)
        for i, joint_idx in enumerate(self.ACTIVE_JOINTS):
            info = p.getJointInfo(self.arm_id, joint_idx, physicsClientId=self.client)
            self.joint_lower[i] = info[8]   # lowerLimit
            self.joint_upper[i] = info[9]   # upperLimit

    def get_joint_angles(self) -> np.ndarray:
        angles = np.zeros(self.NUM_JOINTS)
        for i, joint_idx in enumerate(self.ACTIVE_JOINTS):
            state = p.getJointState(self.arm_id, joint_idx, physicsClientId=self.client)
            angles[i] = state[0]
        return angles

    def get_joint_velocities(self) -> np.ndarray:
        velocities = np.zeros(self.NUM_JOINTS)
        for i, joint_idx in enumerate(self.ACTIVE_JOINTS):
            state = p.getJointState(self.arm_id, joint_idx, physicsClientId=self.client)
            velocities[i] = state[1]
        return velocities

    def get_ee_position(self) -> np.ndarray:
        state = p.getLinkState(self.arm_id, self.EE_LINK_INDEX, physicsClientId=self.client)
        return np.array(state[0])

    def apply_action(self, action: np.ndarray, max_delta: float = 0.2) -> np.ndarray:
        """Apply delta joint angle action. Returns actual joint angles after clipping.

        Note: max_delta default must match the value used during training.
        Checkpoints trained with a different max_delta will produce wrong behavior
        because action=1.0 maps to max_delta radians of joint movement.
        """
        current = self.get_joint_angles()
        target = current + action * max_delta
        # Clip to joint limits
        target = np.clip(target, self.joint_lower, self.joint_upper)

        for i, joint_idx in enumerate(self.ACTIVE_JOINTS):
            p.setJointMotorControl2(
                self.arm_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=float(target[i]),
                force=100,
                physicsClientId=self.client,
            )
        return target

    def reset(self, joint_angles: np.ndarray | None = None):
        """Reset arm to given joint angles (or zeros)."""
        if joint_angles is None:
            joint_angles = np.zeros(self.NUM_JOINTS)
        for i, joint_idx in enumerate(self.ACTIVE_JOINTS):
            p.resetJointState(
                self.arm_id,
                joint_idx,
                float(joint_angles[i]),
                physicsClientId=self.client,
            )

    def compute_workspace_bounds(self, samples_per_joint: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Sweep joint limits and return (min_xyz, max_xyz) bounding box of reachable workspace."""
        points = []
        for j0 in np.linspace(self.joint_lower[0], self.joint_upper[0], samples_per_joint):
            for j1 in np.linspace(self.joint_lower[1], self.joint_upper[1], samples_per_joint):
                for j2 in np.linspace(self.joint_lower[2], self.joint_upper[2], samples_per_joint):
                    self.reset(np.array([j0, j1, j2]))
                    # Step physics once to update link positions
                    p.stepSimulation(physicsClientId=self.client)
                    points.append(self.get_ee_position())

        points = np.array(points)
        self.reset()  # Return to home position
        return points.min(axis=0), points.max(axis=0)
