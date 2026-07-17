import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()