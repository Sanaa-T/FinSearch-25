import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


EnvName = "CartPole-v1"
Episodes = 500
Gamma = 0.99
LR = 1e-3
BatchSize = 64
MemorySize = 10000
TargetUpdate = 10
EPSStart = 1.0
EPSEnd = 0.01
EPSDecay = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def _init_(self, obs_dim, act_dim):
        super(DQN, self)._init_()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def _init_(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, BatchSize):
        samples = random.sample(self.buffer, BatchSize)
        s, a, r, s_, d = zip(*samples)
        return (
            torch.tensor(s, dtype=torch.float32).to(device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(s_, dtype=torch.float32).to(device),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def _len_(self):
        return len(self.buffer)


def select_action(state, steps_done):
    epsilon = EPSEnd + (EPSStart - EPSEnd) * np.exp(-1. * steps_done / EPSDecay)
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax(1).item()


env = gym.make(EnvName, render_mode='human')
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(obs_dim, n_actions).to(device)
target_net = DQN(obs_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MemorySize)


plt.ion()
fig, ax = plt.subplots()
rewards_plot = []
reward_line, = ax.plot([], [], label="Episode Reward")
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('DQN Learning Progress')
ax.grid(True)
ax.legend()

steps_done = 0
for episode in range(Episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render()
        action = select_action(state, steps_done)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps_done += 1

        if len(memory) >= BatchSize:
            s, a, r, s_, d = memory.sample(BatchSize)
            q_vals = policy_net(s).gather(1, a)
            next_q = target_net(s_).max(1)[0].unsqueeze(1).detach()
            expected_q = r + Gamma * next_q * (1 - d)

            loss = nn.functional.mse_loss(q_vals, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % TargetUpdate == 0:
        target_net.load_state_dict(policy_net.state_dict())

    rewards_plot.append(total_reward)
    reward_line.set_data(range(len(rewards_plot)), rewards_plot)
    ax.set_xlim(0, len(rewards_plot))
    ax.set_ylim(0, max(rewards_plot) + 10)
    plt.pause(0.01)

    epsilon_now = EPSEnd + (EPSStart - EPSEnd) * np.exp(-1. * steps_done / EPSDecay)
    print(f"Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon_now:.3f}")

env.close()
plt.ioff()
plt.show()
