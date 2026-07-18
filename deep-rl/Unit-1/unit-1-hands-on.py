import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = gym.make("LunarLander-v3")
observation, info = env.reset()

print("Action space:", env.action_space, "Action space type:", type(env.action_space))
print("Observation space:", env.observation_space)

model = PPO(                #Proximal Policy Optimization
    policy="MlpPolicy",     #The policy model to use (Multi Layer Perceptron - regular nn)
    env=env,                #The environment to train in
    n_steps=1024,           #The number of steps to run the env for
    batch_size=64,          #Smaller batch size to split the n_steps into
    n_epochs=4,             #How many epochs to train the policy nn for
    gamma=0.999,            #Value of short vs. long term reward
    gae_lambda=0.98,        #Controls advantage calculation
    ent_coef=0.01,          #Makes action probabilities more spread out - allows exploartion
    verbose=1,              #Type of print messages outputted
)
# Train the agent       
#model.learn(total_timesteps=int(1e6))

# Save the model

model_name = "ppo-LunarLander-v3"
model.save(model_name)

eval_env = gym.make("LunarLander-v3") #, render_mode="human"
model = PPO.load(model_name, env=eval_env)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

repo_id = "TheAmazing2309/ppo-LunarLander-v3"
env_id = "LunarLander-v3"
eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
model_architecture = "PPO"
commit_message = "Uploading trained PPO LunarLander agent"

package_to_hub(model=model, # Our trained model
               model_name=model_name, # The name of our trained model 
               model_architecture=model_architecture, # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message=commit_message)


"""
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 392         | - average number of steps before the episode ended
|    ep_rew_mean          | -32.3       | - average total reward per episode (what to maximize during training)
| time/                   |             |
|    fps                  | 1013        | - env steps processed per second (speed)
|    iterations           | 98          | - number of full rollout cycles completed (each with n_steps size)
|    time_elapsed         | 98          | - seconds since training started
|    total_timesteps      | 100352      | - total number of env steps (iterations * n_steps = total_timesteps)
| train/                  |             |
|    approx_kl            | 0.003949701 | - update between old and new policy
|    clip_fraction        | 0.00952     | - fraction of samples where loss needed to be clipped
|    clip_range           | 0.2         | - default range for clippig
|    entropy_loss         | -1.08       | - randomness in action distribution
|    explained_variance   | 0.477       | - how well value predictions matched actual returns
|    learning_rate        | 0.0003      | - learning rate of the policy nn based on Adam optimzier
|    loss                 | 89.7        | - combined total loss
|    n_updates            | 388         | - number of gradient updates done so far
|    policy_gradient_loss | -0.00295    | - loss for the policy update
|    value_loss           | 206         | - loss for the value function
-----------------------------------------
"""

'''
for _ in range(20):
    action = env.action_space.sample()
    print("Actions selected:", action, "Action type:", type(action))
    '''