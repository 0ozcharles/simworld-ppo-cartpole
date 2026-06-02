"""Generate an evaluation curve for a completed PPO run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_returns(evaluation: dict, summary: dict) -> list[float]:
    episode_returns = evaluation.get("episode_returns", [])
    if episode_returns:
        return [float(value) for value in episode_returns]

    eval_summary = summary.get("evaluation", evaluation)
    episodes = int(eval_summary.get("episodes", 0))
    mean_return = float(eval_summary.get("mean_return", 0.0))
    min_return = float(eval_summary.get("min_return", mean_return))
    max_return = float(eval_summary.get("max_return", mean_return))

    if episodes and min_return == max_return == mean_return:
        return [mean_return] * episodes
    return []


def save_curve(path: Path, episode_returns: list[float]) -> None:
    mean_return = sum(episode_returns) / len(episode_returns)
    plt.figure(figsize=(9, 5))
    plt.plot(episode_returns, marker="o", linewidth=1.5, label="Evaluation return")
    plt.axhline(mean_return, color="tab:green", linestyle="-", linewidth=2, label=f"Mean {mean_return:.1f}")
    plt.axhline(450, color="tab:red", linestyle="--", linewidth=1, label="Target 450")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Return")
    plt.title("Deterministic Policy Evaluation on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation_curve.png for a run directory")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    evaluation = load_json(run_dir / "evaluation.json")
    summary = load_json(run_dir / "summary.json")
    episode_returns = infer_returns(evaluation, summary)
    if not episode_returns:
        raise SystemExit("Could not infer per-episode evaluation returns from this run.")

    output_path = run_dir / "evaluation_curve.png"
    save_curve(output_path, episode_returns)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
