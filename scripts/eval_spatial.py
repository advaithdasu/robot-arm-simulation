"""
Evaluate a trained model across a uniform spatial grid to map success/failure regions.

Usage:
    python scripts/eval_spatial.py --checkpoint checkpoints/checkpoint_1000000
    python scripts/eval_spatial.py --checkpoint checkpoints/checkpoint_1000000 --output spatial_map.png
"""

import argparse
import csv
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_arm.goal_env import RobotArmGoalEnv, _compute_max_radius_at_z, _get_workspace_bounds


def generate_targets(grid_size: int = 8) -> list[np.ndarray]:
    """Generate uniform Cartesian grid, filter to cylindrical workspace."""
    bounds = _get_workspace_bounds()
    ws_min, ws_max = bounds["ws_min"], bounds["ws_max"]
    z_min, z_max = bounds["z_min"], bounds["z_max"]

    # 4 Z layers (thinner Z range)
    z_layers = max(4, grid_size // 2)
    xs = np.linspace(ws_min[0], ws_max[0], grid_size)
    ys = np.linspace(ws_min[1], ws_max[1], grid_size)
    zs = np.linspace(z_min, z_max, z_layers)

    targets = []
    for z in zs:
        max_r = _compute_max_radius_at_z(z)
        for x in xs:
            for y in ys:
                r = np.sqrt(x**2 + y**2)
                if r <= max_r and max_r > 0.01:
                    targets.append(np.array([x, y, z], dtype=np.float32))
    return targets


def evaluate_checkpoint(checkpoint_path: str, targets: list[np.ndarray]) -> list[dict]:
    """Run one episode per target, return results."""
    env = RobotArmGoalEnv(render_mode=None, success_threshold=0.05)
    model = SAC.load(checkpoint_path, env=env)

    results = []
    for i, target in enumerate(targets):
        obs, _ = env.reset(seed=i)
        env.set_target(target)
        obs = env._get_obs()

        success = False
        final_distance = float("inf")
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                success = True
                final_distance = info["distance"]
                break
            if truncated:
                final_distance = info["distance"]
                break

        results.append({
            "x": float(target[0]),
            "y": float(target[1]),
            "z": float(target[2]),
            "success": success,
            "final_distance": final_distance,
        })

        if (i + 1) % 20 == 0:
            sr = sum(1 for r in results if r["success"]) / len(results)
            print(f"  {i + 1}/{len(targets)} targets evaluated ({sr:.0%} so far)")

    env.close()
    return results


def write_csv(results: list[dict], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "z", "success", "final_distance"])
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]):
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    print(f"\n{'='*50}")
    print(f"SPATIAL EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total valid targets: {total}")
    print(f"Overall success rate: {successes}/{total} ({successes/total:.0%})")

    distances = [r["final_distance"] for r in results]
    print(f"Mean final distance: {np.mean(distances):.4f}m")
    print(f"Median final distance: {np.median(distances):.4f}m")

    # Per Z-layer
    z_vals = sorted(set(r["z"] for r in results))
    print(f"\nPer Z-layer:")
    for z in z_vals:
        layer = [r for r in results if r["z"] == z]
        layer_success = sum(1 for r in layer if r["success"])
        print(f"  z={z:.3f}m: {layer_success}/{len(layer)} ({layer_success/len(layer):.0%})")

    # Per radial band
    print(f"\nPer radial band:")
    for lo, hi, label in [(0, 0.2, "inner"), (0.2, 0.4, "mid"), (0.4, 0.6, "outer")]:
        band = [r for r in results if lo <= np.sqrt(r["x"]**2 + r["y"]**2) < hi]
        if band:
            band_success = sum(1 for r in band if r["success"])
            print(f"  {label} (r={lo:.1f}-{hi:.1f}m): {band_success}/{len(band)} ({band_success/len(band):.0%})")


def plot_results(results: list[dict], output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    for r in results:
        color = "#22c55e" if r["success"] else "#ef4444"
        size = max(8, min(60, r["final_distance"] * 300))
        alpha = 0.8 if r["success"] else 0.5
        ax.scatter(r["x"], r["y"], r["z"], c=color, s=size, alpha=alpha, edgecolors="none")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Spatial Success Map (green=success, red=fail, size=distance)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Spatial evaluation of trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (without .zip)")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid resolution per XY axis")
    parser.add_argument("--output", default="spatial_eval.png", help="Output plot path")
    parser.add_argument("--csv", default="spatial_eval.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Generating target grid (size={args.grid_size})...")
    targets = generate_targets(args.grid_size)
    print(f"  {len(targets)} valid targets in cylindrical workspace")

    print(f"Evaluating {args.checkpoint}...")
    results = evaluate_checkpoint(args.checkpoint, targets)

    write_csv(results, args.csv)
    print(f"CSV saved to {args.csv}")

    print_summary(results)
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
