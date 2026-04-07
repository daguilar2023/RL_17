# Bonus Report — Advanced Reinforcement Learning (Part 3)

## 1. Reward Function Analysis (Part 2)

### The problem with simple rewards

The original (sparse) reward signal gives only minimal feedback:

```
reward = +1  (alive)
reward = -1  (crashed)
```

This is misaligned with the real goal. An agent can collect +1 rewards indefinitely by spinning in circles — it never learns to actually drive forward or avoid crashes efficiently.

Reinforcement learning is *"you get what you reward"*. A bad signal produces bad behaviour.

### Improved reward design (used in both files)

Both `Pyrace_RL_DQN.py` and `Pyrace_PPO.py` use the **`Pyrace-v3`** environment, which has `reward_mode="shaped"` built in. This provides a dense, aligned reward signal:

| Signal | What it does |
|---|---|
| Progress toward next checkpoint | Rewards moving forward |
| Checkpoint reached | Bonus for each waypoint |
| Crash | Large penalty |
| Per-step time penalty | Encourages finishing quickly |

This directly addresses the original flaws:
- **Spinning in circles** no longer earns reward — forward progress is required
- **Crashing** carries a strong penalty
- **Efficiency** is rewarded — less time spent = better score

---

## 2. Bonus — Proximal Policy Optimization (PPO)

### Why move beyond DQN?

DQN approximates Q-values and picks the best discrete action. It works, but it is limited to discrete action spaces and can be unstable.

**PPO** is a policy-gradient method — it directly learns a *policy* π(a | s) that maps states to action probabilities. Key advantages:

| | DQN | PPO |
|---|---|---|
| Action space | Discrete only | Discrete and Continuous |
| Training stability | Sensitive to hyperparameters | Stable (clipped objective) |
| Implementation | From scratch | Library-ready (Stable-Baselines3) |

PPO uses a **clipped objective** to prevent the policy from changing too drastically in one update, making it reliably stable to train.

### Implementation

`Pyrace_PPO.py` uses `stable_baselines3.PPO` with the same `Pyrace-v3` environment as the DQN agent, ensuring the reward signal is identical across both approaches.

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("models_PPO_v01/ppo_pyrace")
```

---

## 3. How to Run

### Install dependencies

```bash
pip install stable-baselines3==2.3.2 shimmy==1.3.0
```

> `stable-baselines3 >= 2.0.0` is required for native `gymnasium` support.

### Train

```bash
python Pyrace_PPO.py --mode train --steps 200000
```

### Evaluate

```bash
python Pyrace_PPO.py --mode eval --eval-episodes 20
```

---

## 4. Summary

| Part | What was done |
|---|---|
| Part 1 — DQN | Discrete-action agent trained from scratch with replay buffer and target network |
| Part 2 — Reward | Switched to `Pyrace-v3` shaped reward: forward progress + checkpoints + crash penalty + time penalty |
| Bonus — PPO | Same environment, more advanced algorithm via Stable-Baselines3; more stable training, extensible to continuous control |

