import numpy as np
import pytest

from robot_arm.goal_env import RobotArmGoalEnv


@pytest.fixture
def env():
    e = RobotArmGoalEnv(render_mode=None)
    yield e
    e.close()


def test_reset_returns_dict_obs(env):
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert set(obs.keys()) == {"observation", "achieved_goal", "desired_goal"}
    assert obs["observation"].shape == (6,)
    assert obs["achieved_goal"].shape == (3,)
    assert obs["desired_goal"].shape == (3,)


def test_step_returns_dict_obs(env):
    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, dict)
    assert obs["observation"].shape == (6,)
    assert isinstance(reward, float)
    assert "is_success" in info
    assert "success" in info


def test_compute_reward_close_gets_bonus(env):
    close = np.array([0.0, 0.0, 0.0])
    target = np.array([0.01, 0.0, 0.0])  # distance = 0.01 < 0.05
    reward = env.compute_reward(close, target, {})
    # -0.01 + 5.0 = 4.99
    assert reward > 4.0, f"Close target should get success bonus, got {reward}"

    far = np.array([0.0, 0.0, 0.0])
    far_target = np.array([1.0, 0.0, 0.0])  # distance = 1.0
    reward_far = env.compute_reward(far, far_target, {})
    # -1.0 + 0 = -1.0
    assert reward_far < 0, f"Far target should be negative, got {reward_far}"
    assert reward > reward_far, "Closer should always score higher"


def test_compute_reward_vectorized(env):
    achieved = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    desired = np.array([[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rewards = env.compute_reward(achieved, desired, {})
    assert rewards.shape == (2,)
    assert rewards[0] > rewards[1]  # closer = higher reward


def test_compute_reward_progressive(env):
    """Reward increases as distance decreases."""
    achieved = np.array([0.0, 0.0, 0.0])
    r_far = env.compute_reward(achieved, np.array([0.5, 0.0, 0.0]), {})
    r_mid = env.compute_reward(achieved, np.array([0.2, 0.0, 0.0]), {})
    r_close = env.compute_reward(achieved, np.array([0.04, 0.0, 0.0]), {})
    assert r_close > r_mid > r_far, f"Expected progressive: {r_close} > {r_mid} > {r_far}"


def test_episode_truncates_at_max_steps(env):
    env.reset(seed=42)
    env.set_target(np.array([10.0, 10.0, 10.0]))  # unreachable
    for i in range(env.max_steps + 5):
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        if terminated or truncated:
            break
    assert truncated is True
    assert i + 1 == env.max_steps


def test_set_target_updates_desired_goal(env):
    env.reset(seed=42)
    new_target = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    env.set_target(new_target)
    obs = env._get_obs()
    np.testing.assert_array_almost_equal(obs["desired_goal"], new_target)


def test_compute_reward_threshold_boundary(env):
    """Sparse component flips exactly at success_threshold (0.05m)."""
    achieved = np.array([0.0, 0.0, 0.0])
    # Just above threshold: sparse = -1.0
    just_above = np.array([0.0500001, 0.0, 0.0])
    r_above = env.compute_reward(achieved, just_above, {})
    # Just below threshold: sparse = 0.0
    just_below = np.array([0.0499999, 0.0, 0.0])
    r_below = env.compute_reward(achieved, just_below, {})
    # The flip should be ~1.0 (sparse goes from -1 to 0)
    assert r_below - r_above > 0.9, f"Expected threshold flip ~1.0, got {r_below - r_above:.4f}"


def test_compute_reward_known_distances(env):
    """Reward matches design doc values at known distances."""
    achieved = np.array([0.0, 0.0, 0.0])
    # d=0.30m: sparse=-1.0, proximity=5*exp(-6)=0.012 → ~-0.99
    r_030 = float(env.compute_reward(achieved, np.array([0.30, 0.0, 0.0]), {}))
    assert -1.05 < r_030 < -0.90, f"d=0.30m: expected ~-0.99, got {r_030:.4f}"
    # d=0.10m: sparse=-1.0, proximity=5*exp(-2)=0.677 → ~-0.32
    r_010 = float(env.compute_reward(achieved, np.array([0.10, 0.0, 0.0]), {}))
    assert -0.45 < r_010 < -0.20, f"d=0.10m: expected ~-0.32, got {r_010:.4f}"
    # d=0.05m exactly: sparse=-1.0 (0.05 >= 0.05 is True), proximity=1.839 → ~+0.84
    r_005 = float(env.compute_reward(achieved, np.array([0.05, 0.0, 0.0]), {}))
    assert 0.5 < r_005 < 1.1, f"d=0.05m: expected ~+0.84, got {r_005:.4f}"
    # d=0.04m: sparse=0.0 (below threshold), proximity=5*exp(-0.8)=2.247 → ~+2.25
    r_004 = float(env.compute_reward(achieved, np.array([0.04, 0.0, 0.0]), {}))
    assert 1.9 < r_004 < 2.6, f"d=0.04m: expected ~+2.25, got {r_004:.4f}"
    # d=0.02m: sparse=0.0, proximity=5*exp(-0.4)=3.352 → ~+3.35
    r_002 = float(env.compute_reward(achieved, np.array([0.02, 0.0, 0.0]), {}))
    assert 3.0 < r_002 < 3.7, f"d=0.02m: expected ~+3.35, got {r_002:.4f}"


def test_max_delta_magnitude():
    """apply_action with max_delta=0.2 produces ~0.2 rad joint change."""
    import pybullet as p
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    from robot_arm.arm import RobotArm
    arm = RobotArm(client)
    arm.reset(np.zeros(3))
    for _ in range(5):
        p.stepSimulation(physicsClientId=client)
    before = arm.get_joint_angles().copy()
    arm.apply_action(np.array([1.0, 0.0, 0.0]))  # max action on joint 0
    for _ in range(20):
        p.stepSimulation(physicsClientId=client)
    after = arm.get_joint_angles()
    delta = abs(after[0] - before[0])
    p.disconnect(client)
    assert 0.15 < delta < 0.25, f"Expected ~0.2 rad change, got {delta:.4f}"


def test_sb3_her_compatibility():
    """SAC with MultiInputPolicy + HerReplayBuffer can create and predict."""
    from stable_baselines3 import SAC
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    env = RobotArmGoalEnv(render_mode=None)
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=0,
    )
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert env.action_space.contains(action)
    env.close()
