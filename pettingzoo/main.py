import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

#env = gym.make('FrozenLake-v1', map_name="8x8", render_mode="human")
env = gym.make('FrozenLake-v1', map_name="8x8")
observation_initial, info = env.reset()

print(env.action_space)
print(env.observation_space)
print(env.observation_space.n, type(env.observation_space.n))
print("Initial Observation: ", observation_initial)

obs_size = env.observation_space.n
act_size = env.action_space.n

def one_hot(o):
    return F.one_hot(torch.tensor(o), num_classes=obs_size).to(torch.float32)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.inputL = nn.Linear(obs_size, 10)
        self.activation = nn.ReLU()
        self.outputL = nn.Linear(10, act_size)

    def forward(self, inp):
        inp = self.inputL(inp)
        inp = self.activation(inp)
        inp = self.outputL(inp)
        return inp

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.inputL = nn.Linear(obs_size, 10)
        self.activation = nn.ReLU()
        self.outputL = nn.Linear(10, 1)

    def forward(self, inp):
        inp = self.inputL(inp)
        inp = self.activation(inp)
        inp = self.outputL(inp)
        return inp

actor = Actor()
critic = Critic()

actor_optimizer = torch.optim.Adam(actor.parameters())
critic_optimizer = torch.optim.Adam(critic.parameters()) 

print(one_hot(observation_initial).dtype)

print(actor(one_hot(observation_initial)))
print(critic(one_hot(observation_initial)))

rollout_episodes = 32
gamma = 0.99
lambda_gae = 0.95
trajectories = []
for i in range(rollout_episodes):
    current = []
    observation, _ = env.reset()
    observation = one_hot(observation)
    while True:
        logits = actor(observation)
        actions_dist = torch.distributions.categorical.Categorical(logits=logits)
        action = actions_dist.sample().item()

        new_observation, reward, terminated, truncated, info = env.step(action)
        current.append({"obs": observation, 
                        "act": action, 
                        "log": actions_dist.log_prob(torch.tensor(action)).item(), 
                        "val": critic(observation).item(), 
                        "rew": reward})
        observation = one_hot(new_observation)

        if terminated or truncated:
            break

    for j in reversed(range(len(current))):
        current[j]["disc_val"] = current[j]["rew"] + current[j+1]["disc_val"] * gamma if j != len(current)-1 else current[j]["rew"] # discounted value
        delta = current[j]["rew"] + gamma * current[j+1]["val"] - current[j]["val"] if j != len(current)-1 else 0
        current[j]["adv"] = delta + gamma * lambda_gae * current[j+1]["adv"] if j != len(current)-1 else 0

    trajectories.extend(current)

advantages = torch.tensor([step["adv"] for step in trajectories])
adv_mean = advantages.mean()
adv_std = advantages.std()
for step in trajectories:
    step["adv"] -= adv_mean
    step["adv"] /= adv_std

random.shuffle(trajectories)

print(trajectories)