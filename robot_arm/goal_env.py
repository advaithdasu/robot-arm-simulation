"""
RobotArmGoalEnv: GoalEnv-compatible environment for 3-DoF arm reaching with HER.

Observation space (Dict):
  observation:    [6] joint angles (rad) + joint velocities (rad/s)
  achieved_goal:  [3] end-effector XYZ (m)
  desired_goal:   [3] target XYZ (m)

Action space (3-dim):
  [-1, 1] per joint, scaled by max_delta (0.2 rad)

Reward:
  Hybrid: sparse for HER + exponential proximity bonus.
  Sparse: 0.0 if distance < threshold, -1.0 otherwise (enables HER relabeling).
  Proximity: 5.0 * exp(-20 * d) (exponential bonus, stronger as arm approaches).

Note: Subclasses gym.Env (not GoalEnv). SB3's HerReplayBuffer only requires
a Dict observation space and a compute_reward() method, not a GoalEnv base class.
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

from robot_arm.arm import RobotArm

# Module-level workspace cache keyed by URDF hash
_workspace_cache: dict[str, dict] = {}

# Arm geometry constants (from URDF)
_L1 = 0.3   # upper arm length
_L2 = 0.3   # forearm length
_BASE_HEIGHT = 0.125  # height of shoulder joint above ground
_SAFETY_MARGIN = 1.0  # no additional margin, limits are set conservatively


def _compute_max_radius_at_z(z: float) -> float:
    """Max reachable horizontal radius at a given Z height, from arm kinematics.

    The arm is a 2-link planar arm (L1+L2) on a rotating base. For a target
    at height z, the pitch-plane constraint is:
        r² + (z - base_height)² <= (L1 + L2)²
    This gives the outer boundary of a sphere centered at the shoulder joint.
    """
    z_rel = z - _BASE_HEIGHT
    max_reach_sq = (_L1 + _L2) ** 2 - z_rel ** 2
    if max_reach_sq <= 0:
        return 0.0
    return min(float(np.sqrt(max_reach_sq) * _SAFETY_MARGIN), 0.525)




def _get_workspace_bounds() -> dict:
    """Compute cylindrical workspace bounds once, cache by URDF hash."""
    from robot_arm.arm import _urdf_hash
    h = _urdf_hash()
    if h in _workspace_cache:
        return _workspace_cache[h]

    # Z range: analytically derived from arm kinematics.
    # Z=0.10m is reachable (280 configs, radius 0.23-0.51m) but has a wider
    # hollow core (min radius ~0.22m). The arm can't reach near-center at low Z.
    z_min = 0.10
    z_max = _BASE_HEIGHT + (_L1 + _L2) * 0.85 - 0.05

    # Also compute a bounding box for observation space bounds (used by server)
    max_r = _compute_max_radius_at_z(_BASE_HEIGHT)  # max radius at shoulder height
    ws_min = np.array([-max_r, -max_r, z_min], dtype=np.float32)
    ws_max = np.array([max_r, max_r, z_max], dtype=np.float32)

    bounds = {
        "z_min": z_min,
        "z_max": z_max,
        "ws_min": ws_min,  # bounding box (for obs space / server compat)
        "ws_max": ws_max,
    }
    _workspace_cache[h] = bounds
    return bounds


class RobotArmGoalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=75, success_threshold=0.05):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.success_threshold = success_threshold

        # Cached workspace bounds (cylindrical)
        bounds = _get_workspace_bounds()
        self._ws_min = bounds["ws_min"]
        self._ws_max = bounds["ws_max"]
        self._z_min = bounds["z_min"]
        self._z_max = bounds["z_max"]

        # Connect PyBullet
        if render_mode == "human":
            self.client = p.connect(p.GUI)
            p.setRealTimeSimulation(1, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.3], physicsClientId=self.client,
            )
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.arm = RobotArm(self.client)

        # Dict observation space for HER
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "achieved_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        # State
        self._target = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._target_visual = None

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> np.ndarray:
        """Hybrid reward: sparse for HER + exponential proximity for precision. Vectorized.

        - Sparse: 0.0 if distance < threshold, -1.0 otherwise (enables HER relabeling)
        - Proximity: 5.0 * exp(-20 * d) (exponential bonus, stronger as arm approaches)

        Combined reward at various distances:
          d=0.30m → -0.99  (sparse penalty dominates)
          d=0.10m → -0.32  (proximity starts pulling)
          d=0.05m → +1.84  (success + strong proximity)
          d=0.02m → +3.35  (success + very strong proximity)
        """
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        sparse = -(d >= self.success_threshold).astype(np.float32)
        proximity = 5.0 * np.exp(-20.0 * d)
        return (sparse + proximity).astype(np.float32)

    def _sample_reachable_target(self) -> np.ndarray:
        """Sample a target within the cylindrical workspace, 35% biased toward edges."""
        for _ in range(100):
            z = self.np_random.uniform(self._z_min, self._z_max)
            max_r = _compute_max_radius_at_z(z)
            if max_r < 0.01:
                continue
            angle = self.np_random.uniform(0, 2 * np.pi)
            if self.np_random.uniform() < 0.35:
                # Edge bias: sample between 80-100% of max radius
                r = max_r * self.np_random.uniform(0.80, 1.0)
            else:
                # Uniform area sampling
                r = max_r * np.sqrt(self.np_random.uniform(0, 1))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            return np.array([x, y, z], dtype=np.float32)
        return np.array([0.0, 0.0, (_BASE_HEIGHT + _L1)], dtype=np.float32)

    def _step_physics(self, n: int = 15):
        if self.render_mode == "human":
            import time
            time.sleep(n / 240.0)
        else:
            for _ in range(n):
                p.stepSimulation(physicsClientId=self.client)

    def _get_obs(self) -> dict[str, np.ndarray]:
        angles = self.arm.get_joint_angles().astype(np.float32)
        velocities = self.arm.get_joint_velocities().astype(np.float32)
        ee_pos = self.arm.get_ee_position().astype(np.float32)
        return {
            "observation": np.concatenate([angles, velocities]),
            "achieved_goal": ee_pos,
            "desired_goal": self._target.copy(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial joint angles (within 85% of limits for wider edge coverage)
        low = self.arm.joint_lower * 0.85
        high = self.arm.joint_upper * 0.85
        init_angles = self.np_random.uniform(low, high)
        self.arm.reset(init_angles)
        self._step_physics(10)

        # Random target within cylindrical workspace (rejection sampling)
        self._target = self._sample_reachable_target()

        # Visualize target
        if self.render_mode == "human":
            if self._target_visual is not None:
                p.removeBody(self._target_visual, physicsClientId=self.client)
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.03,
                rgbaColor=[1, 0, 0, 0.7],
                physicsClientId=self.client,
            )
            self._target_visual = p.createMultiBody(
                baseVisualShapeIndex=visual,
                basePosition=self._target.tolist(),
                physicsClientId=self.client,
            )

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.arm.apply_action(action)
        self._step_physics(15)
        self._step_count += 1

        obs = self._get_obs()

        # NaN guard
        if np.any(np.isnan(obs["observation"])) or np.any(np.isnan(obs["achieved_goal"])):
            obs, info = self.reset()
            return obs, -1.0, False, True, info

        distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))

        terminated = distance < self.success_threshold
        truncated = self._step_count >= self.max_steps

        info = {
            "distance": distance,
            "success": terminated,
            "is_success": terminated,
            "goal": obs["desired_goal"].copy(),
        }
        return obs, reward, terminated, truncated, info

    def set_target(self, target: np.ndarray):
        """Manually set target position (for server/demo)."""
        self._target = np.array(target, dtype=np.float32)
        if self.render_mode == "human":
            if self._target_visual is not None:
                p.removeBody(self._target_visual, physicsClientId=self.client)
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.03,
                rgbaColor=[1, 0, 0, 0.7],
                physicsClientId=self.client,
            )
            self._target_visual = p.createMultiBody(
                baseVisualShapeIndex=visual,
                basePosition=self._target.tolist(),
                physicsClientId=self.client,
            )

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
