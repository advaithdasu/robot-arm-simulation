import numpy as np
import pybullet as p
import pytest

from robot_arm.arm import RobotArm


@pytest.fixture
def arm():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    robot = RobotArm(client)
    yield robot
    p.disconnect(client)


def test_arm_loads(arm):
    """RobotArm loads URDF with correct number of active joints."""
    assert arm.arm_id >= 0
    assert arm.NUM_JOINTS == 3
    assert len(arm.ACTIVE_JOINTS) == 3
    assert len(arm.joint_lower) == 3
    assert len(arm.joint_upper) == 3


def test_locked_joints_stay_fixed(arm):
    """No locked joints in custom URDF -- all 3 joints are active and respond."""
    arm.reset(np.array([0.5, 0.3, -0.2]))
    angles = arm.get_joint_angles()
    np.testing.assert_allclose(angles, [0.5, 0.3, -0.2], atol=0.01)


def test_ee_position_changes_with_action(arm):
    """Applying an action moves the end-effector."""
    arm.reset(np.zeros(3))
    p.stepSimulation(physicsClientId=arm.client)
    pos_before = arm.get_ee_position().copy()

    arm.apply_action(np.array([1.0, 1.0, 1.0]))
    for _ in range(10):
        p.stepSimulation(physicsClientId=arm.client)
    pos_after = arm.get_ee_position()

    assert not np.allclose(pos_before, pos_after, atol=0.001), "End-effector should move after action"


def test_workspace_bounds_valid(arm):
    """Workspace bounding box has positive volume in all axes."""
    ws_min, ws_max = arm.compute_workspace_bounds(samples_per_joint=10)
    extents = ws_max - ws_min

    assert extents[0] > 0.1, f"X extent too small: {extents[0]:.3f}"
    assert extents[1] > 0.1, f"Y extent too small: {extents[1]:.3f}"
    assert extents[2] > 0.1, f"Z extent too small: {extents[2]:.3f}"

    volume = np.prod(extents)
    assert volume > 0.001, f"Workspace volume too small: {volume:.4f} m^3"
