# SimWorld PPO CartPole Submission

This repository is a compact reinforcement learning project for the SimWorld
research exploration task. It implements Proximal Policy Optimization (PPO) from
scratch on `CartPole-v1` using PyTorch and Gymnasium.

## Task

Option A: Reinforcement Learning

- Environment: `CartPole-v1`
- Algorithm: PPO implemented from scratch
- Target: average reward >= 450 over 100 episodes
- Required artifacts: training curves, logs, Colab notebook, and a short README

## Project Structure

```text
.
├── notebooks/
│   └── ppo_cartpole_colab.ipynb
├── scripts/
│   └── check_project.py
├── src/
│   └── ppo_cartpole.py
├── requirements.txt
└── README.md
```

## Method

The implementation includes the standard PPO training loop:

- A shared MLP feature encoder with separate policy and value heads
- Categorical action sampling for the discrete CartPole action space
- Vectorized environment rollouts with `SyncVectorEnv`
- Generalized Advantage Estimation (GAE)
- PPO clipped policy objective
- Value-function regression loss
- Entropy bonus for exploration
- Gradient clipping for stable optimization

No `stable-baselines` or similar high-level RL trainer is used.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python src/ppo_cartpole.py
```

For a quick smoke run:

```bash
python src/ppo_cartpole.py --total-timesteps 4096 --num-envs 2 --num-steps 64 --eval-episodes 5
```

Outputs are written to `runs/<run_name>/`:

- `config.json`
- `metrics.csv`
- `reward_curve.png`
- `evaluation_curve.png`
- `model.pt`
- `evaluation.json`
- `evaluation_returns.csv`
- `summary.json`

## Run In Google Colab

Open directly:

[https://colab.research.google.com/github/0ozcharles/simworld-ppo-cartpole/blob/main/notebooks/ppo_cartpole_colab.ipynb](https://colab.research.google.com/github/0ozcharles/simworld-ppo-cartpole/blob/main/notebooks/ppo_cartpole_colab.ipynb)

Open `notebooks/ppo_cartpole_colab.ipynb` in Colab and run all cells. The
notebook installs dependencies, loads the project implementation, trains the
agent, plots the reward curve, and runs a 100-episode evaluation.

Recommended Colab settings:

- Runtime: Python 3
- Hardware accelerator: GPU is fine but not required for CartPole

## Results

Final Colab run included in this repository:

```text
Run directory: runs/cartpole-v1_ppo_seed42_1780362664
Training episodes collected: 4018
Training last-100 stochastic rollout average return: 139.29
Deterministic evaluation mean return over 100 episodes: 500.00
Evaluation standard deviation: 0.00
Evaluation min / max return: 500.00 / 500.00
```

The training curve is collected from stochastic rollout episodes during PPO
optimization, so it can remain noisy even when the learned deterministic policy
is strong. The final pass/fail result is the deterministic 100-episode
evaluation, which reaches the maximum CartPole-v1 return of 500. Generated
artifacts are saved under `runs/cartpole-v1_ppo_seed42_1780362664/`.

Key result artifacts:

- `runs/cartpole-v1_ppo_seed42_1780362664/reward_curve.png`
- `runs/cartpole-v1_ppo_seed42_1780362664/evaluation_curve.png`
- `runs/cartpole-v1_ppo_seed42_1780362664/metrics.csv`
- `runs/cartpole-v1_ppo_seed42_1780362664/summary.json`
- `runs/cartpole-v1_ppo_seed42_1780362664/evaluation.json`

## Design Notes

I chose CartPole because it provides a clean test of PPO mechanics while staying
small enough to run reliably on free Colab. Vectorized rollouts improve sample
collection speed, and the clipped PPO objective keeps updates stable without
requiring careful trust-region machinery. The code is intentionally explicit so
the algorithmic pieces are easy to inspect. The default run uses a larger
500,000-step budget because short PPO runs can be noisy on CartPole and may not
consistently reach the 450-return target.

## Reproducibility

The default seed is `42`. For more confidence, run several seeds:

```bash
python src/ppo_cartpole.py --seed 1
python src/ppo_cartpole.py --seed 2
python src/ppo_cartpole.py --seed 3
```

## Project Check

Run the lightweight structure and syntax check:

```bash
python scripts/check_project.py
```

If a completed run has final evaluation metrics but is missing
`evaluation_curve.png`, regenerate it with:

```bash
python scripts/make_evaluation_curve.py runs/<run_name>
```
