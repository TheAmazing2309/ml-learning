from pettingzoo import make
import time
import jax
from jax import random, Array, lax, nn, numpy as jnp
from dataclasses import dataclass

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

@dataclass
class Timestep:
    obs : Array
    action : int
    log_prob : float
    reward : float
    value : float
    agent : str
    done : bool

@dataclass
class Parameters:
    actor : dict[str, Array]
    critic : dict[str, Array]

def initialize_parameters(key : Array, actor_layer_sizes : list[int], critic_layer_sizes : list[int]) -> Parameters:
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
    return Parameters(actor=actor_parameters, critic=critic_parameters)

@jax.jit
def network_forward(parameters : dict, observation : Array) -> Array: #for now uses hard-coded params, idk how to jit it while keeping variable model size
    # num_layers = len(parameters) // 2

    # def compute_layer(i : int, x : Array) -> Array:
    #     x = jnp.dot(x, parameters[f"w{i}"]) + parameters[f"b{i}"]
    #     x = jnp.where(i < num_layers - 1, nn.relu(x), x)
    #     return x, None

    # x = observation.reshape(-1)
    # x = lax.fori_loop(0, num_layers, compute_layer, x)
    # return 
    
    x = observation.reshape(-1)
    x = nn.relu(jnp.dot(x, parameters["w0"]) + parameters["b0"])
    x = nn.relu(jnp.dot(x, parameters["w1"]) + parameters["b1"])
    x = jnp.dot(x, parameters["w2"]) + parameters["b2"]
    return x

def collect_trajectories(key : Array, timesteps : int, parameters_0 : Parameters, parameters_1 : Parameters) -> list[Timestep]:
    trajectories = []
    keys = random.split(key, timesteps)
    parameters = {
        env.agents[0] : parameters_0,
        env.agents[1] : parameters_1,
    }
    env.reset()
    while timesteps > 0:
        for agent in env.agent_iter():
            timesteps -= 1
            observation, reward, termination, truncation, info = env.last()
            actor_logits = network_forward(parameters[agent].actor, observation["observation"])
            masked_actor_logits = jnp.where(observation["action_mask"] == 1, actor_logits, -jnp.inf)
            masked_log_probs = nn.log_softmax(masked_actor_logits)
            value = network_forward(parameters[agent].critic, observation["observation"])
            action_index = int(random.categorical(keys[timesteps], masked_actor_logits))
            trajectories.append(Timestep(obs=observation, 
                                        reward=reward, 
                                        log_prob=masked_log_probs[action_index], 
                                        action=action_index, 
                                        value=value, 
                                        agent=agent, 
                                        done=(termination or truncation)))
            if truncation or termination or timesteps <= 0:
                break
            env.step(action_index)
        env.reset()
    return trajectories

if __name__ == "__main__":
    k1, k2 = random.split(key, 2)
    p0 = initialize_parameters(k1, ACTOR_LAYER_SIZES, CRITIC_LAYER_SIZES)
    p1 = initialize_parameters(k2, ACTOR_LAYER_SIZES, CRITIC_LAYER_SIZES)
    # print(actor_parameters)
    # print(critic_parameters)
    sample_observation = env.observe(env.agents[0])["observation"]
    print(network_forward(p0.critic, sample_observation))
    start = time.time()
    result = network_forward(p0.critic, sample_observation)
    print(time.time() - start)
    rollout = collect_trajectories(key, 2048, p0, p1)
  #  print(rollout)
    print("ROLLOUT SIZE:", len(rollout))