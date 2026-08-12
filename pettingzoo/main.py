import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

#env = gym.make('FrozenLake-v1', map_name="8x8", render_mode="human")
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
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

rollout_episodes = 64
gamma = 0.99
lambda_gae = 0.95

epsilon = 0.05
value_loss_coef = 0.5
entropy_coef = 0.2
epochs = 5

all_returns = []
all_losses = []

for rollout_num in range(200):
    print(f"\n=== Rollout {rollout_num + 1} ===")

    trajectories = []
    episode_returns = []

    for i in range(rollout_episodes):
        current = []
        observation, _ = env.reset()
        observation = one_hot(observation)
        episode_reward = 0

        while True:
            logits = actor(observation)
            actions_dist = torch.distributions.categorical.Categorical(logits=logits)
            action = actions_dist.sample().item()

            new_observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            current.append({"obs": observation,
                            "act": action,
                            "log": actions_dist.log_prob(torch.tensor(action)).item(),
                            "val": critic(observation).item(),
                            "rew": reward})
            observation = one_hot(new_observation)

            if terminated or truncated:
                break

        episode_returns.append(episode_reward)

        for j in reversed(range(len(current))):
            current[j]["disc_val"] = current[j]["rew"] + current[j+1]["disc_val"] * gamma if j != len(current)-1 else current[j]["rew"]
            delta = current[j]["rew"] + gamma * current[j+1]["val"] - current[j]["val"] if j != len(current)-1 else 0
            current[j]["adv"] = delta + gamma * lambda_gae * current[j+1]["adv"] if j != len(current)-1 else 0

        trajectories.extend(current)

    random.shuffle(trajectories)

    advantages = torch.tensor([step["adv"] for step in trajectories])
    adv_mean = advantages.mean()
    adv_std = advantages.std()

    for step in trajectories:
        step["adv"] -= adv_mean
        step["adv"] /= adv_std

    minibatch_size = 8
    split_trajectories = [trajectories[i*minibatch_size:(i+1)*minibatch_size] for i in range(len(trajectories) // minibatch_size)]

    print(f"Episode returns: min={min(episode_returns)}, max={max(episode_returns)}, mean={sum(episode_returns)/len(episode_returns):.2f}")
    all_returns.extend(episode_returns)

    for epoch in range(epochs):
        random.shuffle(split_trajectories)

        for batch in split_trajectories:
            batch_dict = {}
            for key in batch[0].keys():
                if key == "obs":
                    batch_dict[key] = torch.stack([t[key] for t in batch])
                else:
                    batch_dict[key] = torch.tensor([t[key] for t in batch])

            new_logits = actor(batch_dict["obs"])
            new_actions_dist = torch.distributions.categorical.Categorical(logits=new_logits)
            new_values = critic(batch_dict["obs"])

            new_log_probs = new_actions_dist.log_prob(batch_dict["act"])

            ratio = torch.exp(new_log_probs - batch_dict["log"])
            surr1 = ratio * batch_dict["adv"]
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_dict["adv"]
            clipped_surr = torch.min(surr1, surr2)
            surr_loss = -clipped_surr.mean()
            value_loss = ((new_values.squeeze() - batch_dict["disc_val"]) ** 2).mean()
            entropy = new_actions_dist.entropy().mean()
            loss = surr_loss + value_loss_coef * value_loss - entropy_coef * entropy
            all_losses.append(loss.item())

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

print(f"Final returns: min={min(all_returns)}, max={max(all_returns)}, mean={sum(all_returns)/len(all_returns):.2f}")