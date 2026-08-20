from pettingzoo import make
from jax import random, Array
import jax.numpy as jnp

env = make("aec", "classic/chess-v6")
env.reset()

print("Agents:", env.agents)
print(*(f"Obs space for {i}: {env.observation_space(i)}\n" for i in env.agents))
print(*(f"Act space for {i}: {env.action_space(i)}\n" for i in env.agents))

OBSERVATION_SIZE = 8 * 8 * 111
ACTION_SIZE = 4672

key = random.PRNGKey(67)

def initialize_parameters(key : Array, actor_layer_sizes : list[int], critic_layer_sizes : list[int]) -> tuple[dict[str, Array], dict[str, Array]]:
    actor_key, critic_key = random.split(key, 2)
    actor_keys = random.split(actor_key, len(actor_layer_sizes) - 1)
    critic_keys = random.split(critic_key, len(critic_layer_sizes) - 1)
    actor_parameters = {}
    critic_parameters = {}
    for i in range(len(actor_layer_sizes) - 1):
        actor_parameters[f"w{i}"] = random.normal(actor_keys[i], shape=(actor_layer_sizes[i], actor_layer_sizes[i+1]), dtype=jnp.float32) * 0.01
        actor_parameters[f"b{i}"] = jnp.zeros((actor_layer_sizes[i+1],), dtype=jnp.float32)
    for i in range(len(critic_layer_sizes) - 1):
        critic_parameters[f"w{i}"] = random.normal(critic_keys[i], shape=(critic_layer_sizes[i], critic_layer_sizes[i+1]), dtype=jnp.float32) * 0.01
        critic_parameters[f"b{i}"] = jnp.zeros((critic_layer_sizes[i+1],), dtype=jnp.float32)
    return actor_parameters, critic_parameters