"""
FastAPI server for robot arm RL simulation.

Loads SAC checkpoints, runs PyBullet inference on demand,
returns joint trajectories for Three.js frontend rendering.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import pybullet as p
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from stable_baselines3 import SAC

# Add parent dir to path so robot_arm package is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_arm.goal_env import RobotArmGoalEnv


# --- Models ---

class SimulateRequest(BaseModel):
    target: list[float] = Field(..., min_length=3, max_length=3)
    checkpoint_step: int = Field(..., ge=0)


class TrajectoryStep(BaseModel):
    angles: list[float]
    ee: list[float]


class SimulateResponse(BaseModel):
    target: list[float]
    checkpoint_step: int
    trajectory: list[TrajectoryStep]
    success: bool
    total_reward: float
    steps: int


# --- Globals ---

models: dict[int, SAC] = {}
available_checkpoints: list[int] = []
workspace_min: list[float] = []
workspace_max: list[float] = []
sim_lock = asyncio.Lock()
last_joint_angles: np.ndarray | None = None


def load_checkpoints():
    """Load all checkpoint models into memory at startup."""
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    loaded = {}

    # HER models require an env during load
    env = RobotArmGoalEnv(render_mode=None)

    for fname in sorted(os.listdir(checkpoint_dir)):
        if not fname.endswith(".zip"):
            continue
        name = fname.replace(".zip", "")
        if name.startswith("checkpoint_"):
            step = int(name.split("_")[1])
        elif name == "final_model":
            continue  # skip, use the numbered checkpoints
        else:
            continue

        path = os.path.join(checkpoint_dir, name)
        loaded[step] = SAC.load(path, env=env)

    env.close()
    return loaded


def compute_workspace():
    """Compute workspace bounds using a headless env."""
    env = RobotArmGoalEnv(render_mode=None)
    ws_min = env._ws_min.tolist()
    ws_max = env._ws_max.tolist()
    env.close()
    return ws_min, ws_max


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and compute workspace on startup."""
    global models, available_checkpoints, workspace_min, workspace_max

    print("Loading checkpoints...")
    start = time.time()
    models = load_checkpoints()
    available_checkpoints = sorted(models.keys())
    print(f"  Loaded {len(models)} checkpoints in {time.time() - start:.1f}s: {available_checkpoints}")

    print("Computing workspace bounds...")
    workspace_min, workspace_max = compute_workspace()
    print(f"  Workspace: {workspace_min} → {workspace_max}")

    yield

    models.clear()


# --- App ---

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "*"
).split(",")

app = FastAPI(
    title="Robot Arm RL Simulation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "checkpoints_loaded": len(models),
        "available_steps": available_checkpoints,
    }


@app.get("/checkpoints")
async def get_checkpoints():
    return {"checkpoints": available_checkpoints}


@app.get("/workspace")
async def get_workspace():
    from robot_arm.goal_env import _L1, _L2, _BASE_HEIGHT, _SAFETY_MARGIN
    return {
        "min": workspace_min,
        "max": workspace_max,
        "arm": {
            "L1": _L1,
            "L2": _L2,
            "base_height": _BASE_HEIGHT,
            "safety_margin": _SAFETY_MARGIN,
        },
    }


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest):
    # Validate checkpoint
    if req.checkpoint_step not in models:
        closest = min(available_checkpoints, key=lambda x: abs(x - req.checkpoint_step))
        raise HTTPException(
            400,
            f"Checkpoint {req.checkpoint_step} not found. "
            f"Available: {available_checkpoints}. Closest: {closest}",
        )

    target = np.array(req.target, dtype=np.float32)

    # Serialize simulation (PyBullet is not thread-safe)
    async with sim_lock:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_simulation, req.checkpoint_step, target
        )

    return result


def _run_simulation(checkpoint_step: int, target: np.ndarray, _models: dict | None = None) -> SimulateResponse:
    """Run one episode of simulation. Called in a thread."""
    model_dict = _models if _models is not None else models
    model = model_dict[checkpoint_step]
    env = RobotArmGoalEnv(render_mode=None, success_threshold=0.05)

    global last_joint_angles
    obs, _ = env.reset(seed=0)
    start_angles = last_joint_angles if last_joint_angles is not None else np.zeros(3)
    env.arm.reset(start_angles)
    env.set_target(target)
    obs = env._get_obs()

    trajectory = []
    total_reward = 0.0
    success = False
    step_count = 0

    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        trajectory.append(TrajectoryStep(
            angles=env.arm.get_joint_angles().tolist(),
            ee=obs["achieved_goal"].tolist(),
        ))

        if terminated:
            success = True
            break
        if truncated:
            break

    last_joint_angles = env.arm.get_joint_angles().copy()
    env.close()

    return SimulateResponse(
        target=target.tolist(),
        checkpoint_step=checkpoint_step,
        trajectory=trajectory,
        success=success,
        total_reward=round(total_reward, 2),
        steps=step_count,
    )
