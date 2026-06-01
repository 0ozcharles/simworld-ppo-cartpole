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
python src/ppo_cartpole.py --total-timesteps 2048 --num-envs 2 --num-steps 64 --eval-episodes 5
```

Outputs are written to `runs/<run_name>/`:

- `config.json`
- `metrics.csv`
- `reward_curve.png`
- `model.pt`
- `evaluation.json`
- `summary.json`

## Run In Google Colab

Open `notebooks/ppo_cartpole_colab.ipynb` in Colab and run all cells. The
notebook installs dependencies, loads the project implementation, trains the
agent, plots the reward curve, and runs a 100-episode evaluation.

Recommended Colab settings:

- Runtime: Python 3
- Hardware accelerator: GPU is fine but not required for CartPole

## Results

After training, record the final result here before submitting:

```text
Training last-100 average return: TODO
Evaluation mean return over 100 episodes: TODO
Evaluation standard deviation: TODO
```

Add the generated reward curve image from `runs/<run_name>/reward_curve.png` to
the final repository or leave it in the run output folder.

## Design Notes

I chose CartPole because it provides a clean test of PPO mechanics while staying
small enough to run reliably on free Colab. Vectorized rollouts improve sample
collection speed, and the clipped PPO objective keeps updates stable without
requiring careful trust-region machinery. The code is intentionally explicit so
the algorithmic pieces are easy to inspect.

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
