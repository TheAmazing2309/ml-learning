import gymnasium as gym, torch

env = gym.make('FrozenLake-v1', map_name="8x8", render_mode="human")
observation_initial, info = env.reset()

print(env.action_space)
print(env.observation_space)
print(env.observation_space.n, type(env.observation_space.n))
print("Initial Observation: ", observation_initial)

obs_size = env.observation_space.n
act_size = env.action_space.n

def one_hot(o):
    return torch.nn.functional.one_hot(torch.tensor(o), num_classes=obs_size).to(torch.float32)

actor = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, act_size)
)

critic = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

print(one_hot(observation_initial).dtype)

print(actor(one_hot(observation_initial)))
print(critic(one_hot(observation_initial)))

batch = 32
trajectories = []
for i in range(batch):
    current = []
    observation, _ = env.reset()
    while True:
        logits = actor(one_hot(observation))
        actions_dist = torch.distributions.categorical.Categorical(logits=logits)
        action = actions_dist.sample().item()

        new_observation, reward, terminated, truncated, info = env.step(action)
        current.append((observation, action, actions_dist[action], reward))
        observation = new_observation

        if terminated or truncated:
            break

    trajectories.append(current)

print(trajectories)