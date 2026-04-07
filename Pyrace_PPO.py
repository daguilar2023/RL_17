"""
Bonus — PPO Agent for PyRace using Stable-Baselines3

Required dependencies:
    pip install stable-baselines3==2.3.2 shimmy==1.3.0

Usage:
    python Pyrace_PPO.py --mode train
    python Pyrace_PPO.py --mode eval
"""

import argparse
import os

import gymnasium as gym
import gym_race  # registers Pyrace-v1 and Pyrace-v3
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ─── Config ───────────────────────────────────────────────────────────────────

# Pyrace-v3: continuous obs (5-D), discrete(4) actions, shaped reward built-in.
# The shaped reward already covers: forward progress, checkpoint bonuses,
# crash penalties, and a per-step time penalty — consistent with Pyrace_RL_DQN.py.
ENV_ID        = "Pyrace-v3"
MODEL_DIR     = "models_PPO_v01"
MODEL_NAME    = "ppo_pyrace"
TOTAL_STEPS   = 200_000
EVAL_EPISODES = 20

# ─── Train ────────────────────────────────────────────────────────────────────

def train(total_steps: int = TOTAL_STEPS):
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = Monitor(gym.make(ENV_ID))

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_steps)

    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"Model saved → {save_path}.zip")
    env.close()


# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(episodes: int = EVAL_EPISODES):
    load_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(load_path + ".zip"):
        raise FileNotFoundError(f"No saved model at {load_path}.zip — run train first.")

    env   = Monitor(gym.make(ENV_ID))
    model = PPO.load(load_path, env=env)

    rewards, crashes, goals = [], 0, 0
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        crashed = info.get("crash", False)
        crashes += int(crashed)
        goals   += int(not crashed)
        print(f"  ep {ep+1:3d}  reward={ep_reward:7.1f}  crash={crashed}")

    env.close()
    print(
        f"\n--- Evaluation ({episodes} episodes) ---\n"
        f"  Mean reward : {np.mean(rewards):.2f}\n"
        f"  Max  reward : {np.max(rewards):.2f}\n"
        f"  Crashes     : {crashes}\n"
        f"  Goals       : {goals}\n"
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyRace PPO — Bonus Part 3")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    if args.mode == "train":
        train(total_steps=args.steps)
    else:
        evaluate(episodes=args.eval_episodes)
