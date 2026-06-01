# Submission Checklist

Use this before submitting the SimWorld form.

## Repository

- [ ] `README.md` explains the task, algorithm, and how to run the code.
- [ ] `notebooks/ppo_cartpole_colab.ipynb` runs end-to-end in Colab.
- [ ] `src/ppo_cartpole.py` contains the PPO implementation.
- [ ] `requirements.txt` installs all dependencies.
- [ ] `runs/<final_run>/metrics.csv` is included.
- [ ] `runs/<final_run>/reward_curve.png` is included.
- [ ] `runs/<final_run>/summary.json` or `evaluation.json` is included.

## Training

- [ ] Run the notebook or script to train PPO on `CartPole-v1`.
- [ ] Confirm the last-100 training average is near or above 450.
- [ ] Confirm the 100-episode evaluation mean is reasonable.
- [ ] Copy the final metrics into the `README.md` Results section.

## GitHub

- [ ] Commit the repository.
- [ ] Push it to GitHub.
- [ ] Open the notebook from GitHub/Colab once to confirm it loads.
- [ ] Use the GitHub repository URL as the form `Submission URL`.

## Form

- [ ] Full name
- [ ] Email
- [ ] Submission URL
- [ ] Optional reviewer note: mention PPO from scratch, CartPole-v1, and where results are saved.
