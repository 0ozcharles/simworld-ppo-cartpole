"""Train PPO from scratch on CartPole-v1."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from torch import nn, optim
from torch.distributions.categorical import Categorical


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    seed: int = 42
    total_timesteps: int = 500_000
    learning_rate: float = 1.0e-3
    num_envs: int = 8
    num_steps: int = 256
    update_epochs: int = 8
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_reward: float = 450.0
    eval_episodes: int = 100
    output_dir: str = "runs"
    device: str = "auto"
    log_interval: int = 5

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden_dim = 128
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs.float())

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.forward_features(obs)).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.forward_features(obs)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        features = self.forward_features(obs)
        logits = self.actor(features)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        return Categorical(logits=logits).sample()


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(env_id: str, seed: int, index: int):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + index)
        env.observation_space.seed(seed + index)
        return env

    return thunk


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_episode_returns(infos: dict[str, Any]) -> list[float]:
    returns: list[float] = []

    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "episode" in info:
                returns.append(float(np.asarray(info["episode"]["r"]).item()))

    if "episode" in infos:
        episode_info = infos["episode"]
        mask = infos.get("_episode", np.ones_like(episode_info["r"], dtype=bool))
        for reward, is_episode in zip(episode_info["r"], mask):
            if is_episode:
                returns.append(float(np.asarray(reward).item()))

    return returns


def save_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_episode_returns_csv(path: Path, episode_returns: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "return"])
        writer.writeheader()
        for episode, episode_return in enumerate(episode_returns, start=1):
            writer.writerow({"episode": episode, "return": episode_return})


def save_reward_curve(path: Path, episode_returns: list[float]) -> None:
    if not episode_returns:
        return
    moving_average = [
        float(np.mean(episode_returns[max(0, index - 99) : index + 1]))
        for index in range(len(episode_returns))
    ]
    plt.figure(figsize=(9, 5))
    plt.plot(episode_returns, alpha=0.35, label="Episode return")
    plt.plot(moving_average, linewidth=2, label="100-episode moving average")
    plt.axhline(450, color="tab:red", linestyle="--", linewidth=1, label="Target 450")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_evaluation_curve(path: Path, episode_returns: list[float]) -> None:
    if not episode_returns:
        return
    mean_return = float(np.mean(episode_returns))
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


def evaluate(
    agent: ActorCritic,
    env_id: str,
    device: torch.device,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    env = gym.make(env_id)
    returns: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            episode_return += float(reward)

        returns.append(episode_return)

    env.close()
    return {
        "episodes": float(episodes),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "episode_returns": returns,
    }


def train(config: PPOConfig) -> dict[str, Any]:
    if config.batch_size % config.num_minibatches != 0:
        raise ValueError("batch_size must be divisible by num_minibatches")

    seed_everything(config.seed)
    device = resolve_device(config.device)

    run_name = f"{config.env_id.lower()}_ppo_seed{config.seed}_{int(time.time())}"
    run_dir = Path(config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    envs = SyncVectorEnv(
        [make_env(config.env_id, config.seed, index) for index in range(config.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("This implementation expects a discrete action space")

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(envs.single_action_space.n)
    agent = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs = torch.zeros((config.num_steps, config.num_envs, obs_dim), device=device)
    actions = torch.zeros((config.num_steps, config.num_envs), device=device, dtype=torch.long)
    log_probs = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones = torch.zeros((config.num_steps, config.num_envs), device=device)
    values = torch.zeros((config.num_steps, config.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=config.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(config.num_envs, obs_dim)
    next_done = torch.zeros(config.num_envs, device=device)

    episode_returns: list[float] = []
    metric_rows: list[dict[str, float]] = []
    global_step = 0
    start_time = time.time()
    num_updates = config.total_timesteps // config.batch_size

    for update in range(1, num_updates + 1):
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value

            actions[step] = action
            log_probs[step] = log_prob

            next_obs_np, reward_np, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done_np = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(config.num_envs, obs_dim)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)
            episode_returns.extend(extract_episode_returns(infos))

        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards, device=device)
            last_gae_lam = 0.0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + config.gamma * next_values * next_non_terminal - values[t]
                last_gae_lam = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        indices = np.arange(config.batch_size)
        clip_fracs: list[float] = []

        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, config.batch_size, config.minibatch_size):
                minibatch_indices = indices[start : start + config.minibatch_size]

                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_obs[minibatch_indices], b_actions[minibatch_indices]
                )
                log_ratio = new_log_prob - b_log_probs[minibatch_indices]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fracs.append(
                        float(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                    )

                minibatch_advantages = b_advantages[minibatch_indices]
                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (
                    minibatch_advantages.std() + 1e-8
                )

                policy_loss_unclipped = -minibatch_advantages * ratio
                policy_loss_clipped = -minibatch_advantages * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

                new_value = new_value.view(-1)
                value_loss = 0.5 * ((new_value - b_returns[minibatch_indices]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - config.ent_coef * entropy_loss + config.vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        explained_variance = np.nan
        if np.var(y_true) > 0:
            explained_variance = float(1 - np.var(y_true - y_pred) / np.var(y_true))

        last_100 = float(np.mean(episode_returns[-100:])) if episode_returns else 0.0
        steps_per_second = int(global_step / max(time.time() - start_time, 1e-9))
        row = {
            "update": float(update),
            "global_step": float(global_step),
            "episodes": float(len(episode_returns)),
            "last_100_return": last_100,
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": explained_variance,
            "steps_per_second": float(steps_per_second),
        }
        metric_rows.append(row)

        if update == 1 or update % config.log_interval == 0:
            print(
                f"update={update:03d} step={global_step:06d} "
                f"episodes={len(episode_returns):04d} last100={last_100:.1f} sps={steps_per_second}"
            )

        if len(episode_returns) >= 100 and last_100 >= config.target_reward:
            print(f"Target reached: last 100 episode mean return = {last_100:.1f}")
            break

    envs.close()

    torch.save(agent.state_dict(), run_dir / "model.pt")
    save_metrics_csv(run_dir / "metrics.csv", metric_rows)
    save_reward_curve(run_dir / "reward_curve.png", episode_returns)

    evaluation = evaluate(
        agent=agent,
        env_id=config.env_id,
        device=device,
        episodes=config.eval_episodes,
        seed=config.seed + 10_000,
    )
    save_episode_returns_csv(run_dir / "evaluation_returns.csv", evaluation["episode_returns"])
    save_evaluation_curve(run_dir / "evaluation_curve.png", evaluation["episode_returns"])
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "episodes": len(episode_returns),
        "last_100_train_return": float(np.mean(episode_returns[-100:])) if episode_returns else 0.0,
        "evaluation": {
            key: value for key, value in evaluation.items() if key != "episode_returns"
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="PPO from scratch on CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=PPOConfig.total_timesteps)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    parser.add_argument("--num-steps", type=int, default=PPOConfig.num_steps)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--ent-coef", type=float, default=PPOConfig.ent_coef)
    parser.add_argument("--eval-episodes", type=int, default=PPOConfig.eval_episodes)
    parser.add_argument("--output-dir", type=str, default=PPOConfig.output_dir)
    parser.add_argument("--device", type=str, default=PPOConfig.device)
    args = parser.parse_args()
    return PPOConfig(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        ent_coef=args.ent_coef,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    train(parse_args())
