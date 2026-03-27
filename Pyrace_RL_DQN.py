import argparse
import copy
import os
import random
from collections import deque
from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import gym_race

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION_NAME = "DQN_v01"
REPORT_EPISODES = 500
DISPLAY_EPISODES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([e.state for e in batch], dtype=np.float32)
        actions = np.array([e.action for e in batch], dtype=np.int64)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([e.done for e in batch], dtype=np.float32)
        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self):
        return len(self.buffer)


def hard_update_target(target_network, source_network):
    target_network.load_state_dict(source_network.state_dict())


def soft_update_target(target_network, source_network, tau):
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        return dqn_network(state_tensor).argmax(1).item()


def get_epsilon(step, cfg):
    if not cfg["epsilon_decay_steps"]:
        return cfg["epsilon_end"]
    progress = min(1.0, step / float(cfg["epsilon_decay_steps"]))
    return cfg["epsilon_start"] + progress * (cfg["epsilon_end"] - cfg["epsilon_start"])


def default_configs():
    baseline_cfg = {
        "name": "baseline",
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "batch_size": 32,
        "replay_capacity": 10000,
        "warmup_steps": 32,
        "epsilon_start": 0.8,
        "epsilon_end": 0.8,
        "epsilon_decay_steps": 1,
        "use_target_network": False,
        "target_update_interval": 250,
        "target_soft_tau": 0.0,
        "double_dqn": False,
        "loss": "mse",
        "gradient_clip": None,
    }
    improved_cfg = {
        "name": "step2_improved",
        "learning_rate": 0.0005,
        "discount_factor": 0.99,
        "batch_size": 64,
        "replay_capacity": 20000,
        "warmup_steps": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 20000,
        "use_target_network": True,
        "target_update_interval": 250,
        "target_soft_tau": 0.01,
        "double_dqn": True,
        "loss": "huber",
        "gradient_clip": 10.0,
    }
    return baseline_cfg, improved_cfg


def simulate(
    cfg,
    learning=True,
    episode_start=0,
    num_episodes=1000,
    max_t=2000,
    report_episodes=200,
    display_episodes=100,
    checkpoint_every=500,
    save_prefix=VERSION_NAME,
    enable_render=False,
):
    global dqn_network, optimizer, replay_buffer, target_network

    env.set_view(enable_render)
    losses_window = deque(maxlen=500)
    rewards = []
    crashes = 0
    goals = 0
    max_reward = -10_000
    global_step = 0
    loss_fn = nn.SmoothL1Loss() if cfg["loss"] == "huber" else nn.MSELoss()

    for episode in range(episode_start, episode_start + num_episodes):
        obv, _ = env.reset()
        state = obv.astype(np.float32)
        episode_reward = 0.0
        episode_losses = []
        last_info = {"check": 0, "dist": 0, "crash": False}

        if not learning:
            env.pyrace.mode = 2

        for t in range(max_t):
            epsilon = get_epsilon(global_step, cfg) if learning else 0.0
            action = select_action(state, epsilon)
            obv, reward, done, _, info = env.step(action)
            last_info = info
            next_state = obv.astype(np.float32)
            episode_reward += reward

            if learning:
                replay_buffer.add(state, action, reward, next_state, done)
                if len(replay_buffer) >= cfg["batch_size"] and global_step >= cfg["warmup_steps"]:
                    states, actions, rewards_b, next_states, dones = replay_buffer.sample(cfg["batch_size"])
                    current_q = dqn_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        if cfg["use_target_network"]:
                            if cfg["double_dqn"]:
                                next_actions = dqn_network(next_states).argmax(1, keepdim=True)
                                next_q = target_network(next_states).gather(1, next_actions).squeeze(1)
                            else:
                                next_q = target_network(next_states).max(1)[0]
                        else:
                            next_q = dqn_network(next_states).max(1)[0]
                        target_q = rewards_b + cfg["discount_factor"] * next_q * (1.0 - dones)

                    loss = loss_fn(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    if cfg["gradient_clip"] is not None:
                        nn.utils.clip_grad_norm_(dqn_network.parameters(), cfg["gradient_clip"])
                    optimizer.step()
                    episode_losses.append(float(loss.item()))
                    losses_window.append(float(loss.item()))

                    if cfg["use_target_network"]:
                        if cfg["target_soft_tau"] > 0.0:
                            soft_update_target(target_network, dqn_network, cfg["target_soft_tau"])
                        elif global_step % cfg["target_update_interval"] == 0:
                            hard_update_target(target_network, dqn_network)

            state = next_state
            global_step += 1

            if enable_render and ((episode % display_episodes == 0) or (env.pyrace.mode == 2)):
                env.set_msgs(
                    [
                        f"Mode: {cfg['name']}",
                        f"Episode: {episode}",
                        f"Time steps: {t}",
                        f"check: {info['check']}",
                        f"dist: {info['dist']}",
                        f"crash: {info['crash']}",
                        f"Reward: {episode_reward:.0f}",
                        f"Max Reward: {max_reward:.0f}",
                        f"eps: {epsilon:.3f}",
                    ]
                )
                env.render()

            if done or t >= max_t - 1:
                crashes += int(bool(last_info.get("crash", False)))
                goals += int(not bool(last_info.get("crash", False)))
                if episode_reward > max_reward:
                    max_reward = episode_reward
                break

        rewards.append(float(episode_reward))
        avg20_reward = float(np.mean(rewards[-20:]))
        avg_loss = float(np.mean(episode_losses)) if episode_losses else float("nan")
        rolling_loss = float(np.mean(losses_window)) if losses_window else float("nan")

        if learning and (episode + 1) % checkpoint_every == 0:
            os.makedirs(f"models_{save_prefix}", exist_ok=True)
            file = f"models_{save_prefix}/model_{episode + 1}.pt"
            torch.save(dqn_network.state_dict(), file)
            print(file, "saved")

        if learning and (episode + 1) % report_episodes == 0:
            print(
                f"[{cfg['name']}] ep={episode + 1} avg20_reward={avg20_reward:.2f} "
                f"ep_loss={avg_loss:.4f} roll_loss={rolling_loss:.4f} "
                f"crashes={crashes} goals={goals} eps={get_epsilon(global_step, cfg):.4f}"
            )

    return {
        "episodes": len(rewards),
        "avg_reward_all": float(np.mean(rewards)) if rewards else 0.0,
        "avg_reward_last20": float(np.mean(rewards[-20:])) if rewards else 0.0,
        "max_reward": float(max(rewards)) if rewards else 0.0,
        "min_reward": float(min(rewards)) if rewards else 0.0,
        "avg_loss_last500": float(np.mean(losses_window)) if losses_window else None,
        "crashes": int(crashes),
        "goals": int(goals),
        "rewards": rewards,
    }


def init_agent(cfg):
    global dqn_network, target_network, optimizer, replay_buffer
    dqn_network = DQNNetwork(STATE_SIZE, NUM_ACTIONS, hidden_size=128).to(device)
    target_network = copy.deepcopy(dqn_network).to(device)
    optimizer = optim.Adam(dqn_network.parameters(), lr=cfg["learning_rate"])
    replay_buffer = ReplayBuffer(capacity=cfg["replay_capacity"])
    hard_update_target(target_network, dqn_network)


def run_single(cfg, episodes, max_t, report_every, checkpoint_every, render):
    init_agent(cfg)
    return simulate(
        cfg,
        learning=True,
        episode_start=0,
        num_episodes=episodes,
        max_t=max_t,
        report_episodes=report_every,
        display_episodes=DISPLAY_EPISODES,
        checkpoint_every=checkpoint_every,
        save_prefix=VERSION_NAME,
        enable_render=render,
    )


def run_benchmark(episodes, max_t, report_every, checkpoint_every):
    baseline_cfg, improved_cfg = default_configs()
    print("Running baseline benchmark...")
    baseline_stats = run_single(baseline_cfg, episodes, max_t, report_every, checkpoint_every, render=False)
    print("Running improved Step 2 benchmark...")
    improved_stats = run_single(improved_cfg, episodes, max_t, report_every, checkpoint_every, render=False)

    print("BASELINE:", {k: v for k, v in baseline_stats.items() if k != "rewards"})
    print("IMPROVED:", {k: v for k, v in improved_stats.items() if k != "rewards"})
    delta_last20 = improved_stats["avg_reward_last20"] - baseline_stats["avg_reward_last20"]
    delta_goals = improved_stats["goals"] - baseline_stats["goals"]
    delta_crashes = improved_stats["crashes"] - baseline_stats["crashes"]
    print(
        "DELTA:",
        {
            "avg_reward_last20_delta": float(delta_last20),
            "goals_delta": int(delta_goals),
            "crashes_delta": int(delta_crashes),
        },
    )

    plt.figure(figsize=(10, 4))
    plt.plot(baseline_stats["rewards"], label="baseline")
    plt.plot(improved_stats["rewards"], label="step2_improved")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Step 2 benchmark reward curves")
    plt.legend()
    os.makedirs(f"models_{VERSION_NAME}", exist_ok=True)
    plot_file = f"models_{VERSION_NAME}/benchmark_rewards.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=130)
    plt.close()
    print("Saved benchmark plot:", plot_file)
    return baseline_stats, improved_stats


def load_and_play(episode):
    file = f"models_{VERSION_NAME}/model_{episode}.pt"
    if not os.path.exists(file):
        raise FileNotFoundError(f"Checkpoint not found: {file}")
    baseline_cfg, _ = default_configs()
    init_agent(baseline_cfg)
    dqn_network.load_state_dict(torch.load(file, map_location=device))
    print(file, "loaded")
    simulate(
        baseline_cfg,
        learning=False,
        episode_start=episode,
        num_episodes=50,
        max_t=2000,
        enable_render=True,
        report_episodes=50,
        checkpoint_every=1000000,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="PyRace DQN with Step 2 algorithm upgrades")
    parser.add_argument("--env-id", default="Pyrace-v3", help="Gymnasium environment id to run")
    parser.add_argument("--mode", choices=["train", "play", "benchmark"], default="benchmark")
    parser.add_argument("--variant", choices=["baseline", "improved"], default="improved")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-t", type=int, default=700)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--play-checkpoint", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", help="Disable rendering during training")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_id).unwrapped
    os.makedirs(f"models_{VERSION_NAME}", exist_ok=True)

    STATE_SIZE = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.n
    print(f"State size: {STATE_SIZE}, Action size: {NUM_ACTIONS}, Device: {device}")

    baseline_cfg, improved_cfg = default_configs()

    if args.mode == "play":
        load_and_play(args.play_checkpoint)
    elif args.mode == "train":
        cfg = baseline_cfg if args.variant == "baseline" else improved_cfg
        stats = run_single(
            cfg,
            episodes=args.episodes,
            max_t=args.max_t,
            report_every=args.report_every,
            checkpoint_every=args.checkpoint_every,
            render=not args.headless,
        )
        print("TRAIN RESULT:", {k: v for k, v in stats.items() if k != "rewards"})
    else:
        run_benchmark(
            episodes=args.episodes,
            max_t=args.max_t,
            report_every=args.report_every,
            checkpoint_every=args.checkpoint_every,
        )
