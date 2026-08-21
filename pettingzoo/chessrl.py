from pettingzoo import make
import jax
import time
from jax import random, Array, lax, nn
import jax.numpy as jnp

env = make("aec", "classic/chess-v6")
env.reset()

print("Agents:", env.agents)
print(*(f"Obs space for {i}: {env.observation_space(i)}\n" for i in env.agents))
print(*(f"Act space for {i}: {env.action_space(i)}\n" for i in env.agents))

OBSERVATION_SIZE = 8 * 8 * 111
ACTION_SIZE = 4672
ACTOR_LAYER_SIZES = [OBSERVATION_SIZE, 256, 256, ACTION_SIZE]
CRITIC_LAYER_SIZES = [OBSERVATION_SIZE, 256, 256, 1]

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

@jax.jit
def network_forward(parameters : dict, observation : Array) -> Array: #for now uses hard-coded params, idk how to jit it while keeping variable model size
    # num_layers = len(parameters) // 2

    # def compute_layer(i : int, x : Array) -> Array:
    #     x = jnp.dot(x, parameters[f"w{i}"]) + parameters[f"b{i}"]
    #     x = jnp.where(i < num_layers - 1, nn.relu(x), x)
    #     return x, None

    # x = observation.reshape(-1)
    # x = lax.fori_loop(0, num_layers, compute_layer, x)
    # return x
    x = observation.reshape(-1)
    x = nn.relu(jnp.dot(x, parameters["w0"]) + parameters["b0"])
    x = nn.relu(jnp.dot(x, parameters["w1"]) + parameters["b1"])
    x = jnp.dot(x, parameters["w2"]) + parameters["b2"]
    return x

if __name__ == "__main__":
    
    actor_parameters, critic_parameters = initialize_parameters(key, ACTOR_LAYER_SIZES, CRITIC_LAYER_SIZES)
    # print(actor_parameters)
    # print(critic_parameters)
    sample_observation = env.observe(env.agents[0])["observation"]
    print(network_forward(critic_parameters, sample_observation))
    start = time.time()
    result = network_forward(critic_parameters, sample_observation)
    print(time.time() - start)