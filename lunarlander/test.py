import math
import random
import signal
from collections import namedtuple, deque
from itertools import count
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from train import DQN, select_action, plot_durations

stop_signal = False

def keyboard_interrupt_handler():
   global stop_signal
   stop_signal = True
   print("Received keyboard interrup signal, waiting for episode to finish")

signal.signal(signal.SIGINT, keyboard_interrupt_handler)
   
env = gym.make("LunarLander-v2", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu" )
print(f"Using device: {device}")

n_actions = env.action_space.n
observation, info = env.reset(seed=42)
n_observations = len(observation)

model = DQN(n_actions, n_observations)
model.load_state_dict(torch.load("weights/moodel_weights.pth"))

episode_rewards = []

for i_episode in range(600):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward= 0

    if stop_signal:
      break

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        total_reward += reward
        done = terminated or truncated

        if done:
         episode_rewards.append(total_reward)
         break

print("Complete!")
plot_durations(show_result=True)
plt.show()
plt.savefig("logs/testing.png")
env.close()