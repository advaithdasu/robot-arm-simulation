"""
Train a SAC + HER agent to reach targets with the 3-DoF robot arm.

Usage:
    python scripts/train.py                     # 500k steps
    python scripts/train.py --steps 1000000     # 1M steps
    python scripts/train.py --resume --steps 1000000  # resume from latest checkpoint
"""

import argparse
import json
import os
import time

# Fix OpenMP duplicate library issue on macOS with conda
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_arm.goal_env import RobotArmGoalEnv


def make_env(seed: int):
    """Factory function for SubprocVecEnv — each worker gets its own env."""
    def _init():
        env = RobotArmGoalEnv(render_mode=None)
        env.reset(seed=seed)
        return env
    return _init


class CheckpointCallback(BaseCallback):
    """Save model checkpoints, record eval episodes, and stop if plateaued."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        recording_path: str,
        eval_target: np.ndarray,
        patience: int = 3,
        min_improvement: float = 0.02,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.recording_path = recording_path
        self.eval_target = eval_target
        self.patience = patience
        self.min_improvement = min_improvement
        self._best_success_rate = 0.0
        self._stale_checks = 0
        self._check_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"  Checkpoint saved: {path}")
            self._record_eval_episode(self.num_timesteps)

            # Only stop early if we've hit the target success rate
            self._check_count += 1
            if self._check_count >= 3:
                current_sr = 0.0
                if hasattr(self.model, 'logger') and self.model.logger is not None:
                    try:
                        current_sr = self.logger.name_to_value.get('rollout/success_rate', 0.0)
                    except Exception:
                        pass

                if current_sr > self._best_success_rate:
                    self._best_success_rate = current_sr
                if self.verbose:
                    print(f"  Success rate: {current_sr:.2%} (best: {self._best_success_rate:.2%})")

                if self._best_success_rate >= 0.90:
                    print(f"\n  TARGET REACHED: {self._best_success_rate:.2%} success rate. Stopping.")
                    return False
        return True

    def _record_eval_episode(self, step: int):
        env = RobotArmGoalEnv(render_mode=None, success_threshold=0.05)
        obs, _ = env.reset(seed=0)
        env.set_target(self.eval_target)
        obs = env._get_obs()

        recording = {
            "step": step,
            "target": self.eval_target.tolist(),
            "urdf_hash": env.arm.urdf_hash,
            "joint_angles": [],
            "ee_positions": [],
            "rewards": [],
        }

        total_reward = 0.0
        success = False
        for _ in range(env.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            recording["joint_angles"].append(env.arm.get_joint_angles().tolist())
            recording["ee_positions"].append(obs["achieved_goal"].tolist())
            recording["rewards"].append(float(reward))
            total_reward += reward
            if terminated:
                success = True
                break
            if truncated:
                break

        recording["total_reward"] = total_reward
        recording["success"] = success

        path = os.path.join(self.recording_path, f"recording_{step}.json")
        with open(path, "w") as f:
            json.dump(recording, f)

        env.close()


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[str | None, int]:
    """Find the latest checkpoint and return (path, step)."""
    import re
    latest_step = -1
    latest_path = None
    if not os.path.exists(checkpoint_dir):
        return None, 0
    for fname in os.listdir(checkpoint_dir):
        match = re.match(r"checkpoint_(\d+)\.zip", fname)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    if latest_path is None:
        return None, 0
    return latest_path, latest_step


def main():
    parser = argparse.ArgumentParser(description="Train robot arm RL agent with HER")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Steps between checkpoints")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Replay buffer size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--n-sampled-goal", type=int, default=4, help="HER virtual goals per transition")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    eval_target = np.array([0.2, 0.15, 0.4], dtype=np.float32)

    device = args.device
    if device == "auto":
        device = "cpu"

    # Create parallel envs
    n_envs = args.n_envs
    env = SubprocVecEnv([make_env(args.seed + i) for i in range(n_envs)])

    # Resume logic
    resumed_steps = 0
    if args.resume:
        ckpt_path, resumed_steps = find_latest_checkpoint("checkpoints")
        if ckpt_path is None:
            print("No checkpoint found, starting fresh.")
        else:
            print(f"Resuming from {ckpt_path} (step {resumed_steps:,})")

    remaining_steps = args.steps - resumed_steps
    if remaining_steps <= 0:
        print(f"Already completed {resumed_steps:,} / {args.steps:,} steps. Nothing to do.")
        env.close()
        return

    # Read env params from one instance for display
    _tmp_env = RobotArmGoalEnv(render_mode=None)
    print(f"Training SAC+HER agent for {args.steps:,} total steps (seed={args.seed})")
    print(f"  Resumed from step: {resumed_steps:,}")
    print(f"  Remaining steps: {remaining_steps:,}")
    print(f"  Device: {device}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  HER n_sampled_goal: {args.n_sampled_goal}")
    print(f"  Max steps/episode: {_tmp_env.max_steps}")
    print(f"  Success threshold: {_tmp_env.success_threshold}m")
    _tmp_env.close()

    # Create or load model
    if args.resume and resumed_steps > 0:
        ckpt_path, _ = find_latest_checkpoint("checkpoints")
        model = SAC.load(
            ckpt_path,
            env=env,
            device=device,
            tensorboard_log="./logs/",
        )
        model.num_timesteps = resumed_steps
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy="future",
            ),
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=256,
            tau=0.005,
            gamma=0.95,
            learning_starts=n_envs * 75 * 2,  # need full episodes before HER can sample
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log="./logs/",
        )

    callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path="checkpoints",
        recording_path="recordings",
        eval_target=eval_target,
        verbose=1,
    )

    if resumed_steps == 0:
        print("Recording untrained baseline...")
        callback.model = model
        callback._record_eval_episode(0)

    start = time.time()
    model.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        log_interval=10,
        reset_num_timesteps=False,
    )
    elapsed = time.time() - start

    model.save("checkpoints/final_model")
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Final model saved to checkpoints/final_model")

    # Final eval with relaxed threshold
    eval_env = RobotArmGoalEnv(render_mode=None, success_threshold=0.05)
    successes = 0
    for trial in range(20):
        obs, _ = eval_env.reset(seed=trial + 100)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        if info.get("is_success"):
            successes += 1

    print(f"Final eval: {successes}/20 targets reached ({successes/20*100:.0f}% success rate)")
    eval_env.close()
    env.close()

    # Generate training curve graph from TensorBoard logs
    _generate_training_curve(args.steps)


def _generate_training_curve(total_steps: int):
    """Read TensorBoard logs and generate a training curve PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("matplotlib or tensorboard not available, skipping graph")
        return

    # Find the latest tfevents file
    log_dir = "./logs/"
    latest_run = None
    latest_mtime = 0
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if "tfevents" in f:
                path = os.path.join(root, f)
                mtime = os.path.getmtime(path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_run = root

    if latest_run is None:
        print("No TensorBoard logs found, skipping graph")
        return

    ea = EventAccumulator(latest_run)
    ea.Reload()

    if "rollout/success_rate" not in ea.Tags().get("scalars", []):
        print("No success_rate in logs, skipping graph")
        return

    events = ea.Scalars("rollout/success_rate")
    steps = np.array([e.step for e in events])
    sr = np.array([e.value * 100 for e in events])

    # Smooth
    window = min(50, len(sr) // 4) if len(sr) > 20 else 1
    if window > 1:
        sr_smooth = np.convolve(sr, np.ones(window) / window, mode="valid")
        steps_smooth = steps[window - 1:]
    else:
        sr_smooth = sr
        steps_smooth = steps

    peak_idx = np.argmax(sr_smooth)
    peak_step = steps_smooth[peak_idx]
    peak_val = sr_smooth[peak_idx]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.scatter(steps / 1000, sr, s=1.5, alpha=0.15, color="#60a5fa", zorder=1)
    ax.plot(steps_smooth / 1000, sr_smooth, color="#3b82f6", linewidth=2.5, zorder=2, label="Rolling average")

    ax.annotate(
        f"Peak: {peak_val:.0f}%\n({peak_step / 1000:.0f}k steps)",
        xy=(peak_step / 1000, peak_val),
        xytext=(peak_step / 1000 - 150, peak_val + 8),
        fontsize=11, fontweight="bold", color="#22c55e",
        arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.5),
        zorder=3,
    )

    ax.axhline(y=90, color="#22c55e", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(20, 91.5, "90% target", fontsize=9, color="#22c55e", alpha=0.6)

    ax.set_xlabel("Training Steps (thousands)", fontsize=13, labelpad=10)
    ax.set_ylabel("Success Rate (%)", fontsize=13, labelpad=10)
    ax.set_title("SAC + HER Training: Learning Curve", fontsize=16, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_xlim(0, max(steps / 1000) * 1.05)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.3)

    plt.tight_layout()
    out_path = "training_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"Training curve saved to {out_path}")


if __name__ == "__main__":
    main()
