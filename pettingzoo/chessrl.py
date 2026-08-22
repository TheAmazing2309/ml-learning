from pettingzoo import make
import time
import optax
import random as rand
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
GAMMA = 0.99
LAMBDA = 0.95
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
    advantage : float = None
    discounted_return : float = None

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
            trajectories.append(Timestep(obs=observation["observation"], 
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

def compute_advantages(trajectories : list[Timestep]) -> list[Timestep]:
    prev_advantage = {"player_0" : 0, "player_1" : 0}
    prev_value = {"player_0" : 0, "player_1" : 0}
    for timestep in reversed(trajectories):
        if timestep.done:
            prev_advantage = {"player_0" : 0, "player_1" : 0}
            prev_value = {"player_0" : 0, "player_1" : 0}
        delta = timestep.reward + GAMMA * prev_value[timestep.agent] - timestep.value
        timestep.advantage = delta + (GAMMA * LAMBDA) * prev_advantage[timestep.agent]
        timestep.discounted_return = timestep.advantage + timestep.value
        prev_advantage[timestep.agent] = timestep.advantage
        prev_value[timestep.agent] = timestep.value
    advantages = jnp.array([t.advantage for t in trajectories])
    mean = jnp.mean(advantages)
    std = jnp.maximum(jnp.std(advantages), 1e-8)
    for timestep in trajectories:
        timestep.advantage -= mean
        timestep.advantage /= std
    return trajectories

def compute_loss_grads(
        parameters : Parameters,
        batch : list[Timestep],
        epsilon : float = 0.2,
        vf_coef : float = 0.5,
        ent_coef : float = 0.01
) -> tuple[float, Parameters]:
    observations = jnp.stack([t.obs for t in batch])
    actions = jnp.array([t.action for t in batch])
    old_log_probs = jnp.array([t.log_prob for t in batch])
    advantages = jnp.array([t.advantage for t in batch])
    discounted_returns = jnp.array([t.discounted_return for t in batch])
    new_logits = jax.vmap(lambda obs: network_forward(parameters.actor, obs))(observations)
    new_values = jax.vmap(lambda obs: network_forward(parameters.critic, obs))(observations)
    new_log_probs = nn.log_softmax(new_logits)[:,actions]
    return None

def train(
        epochs : int, 
        batches : int, 
        rollout : list[Timestep], 
        parameters_0 : Parameters, 
        parameters_1 : Parameters,
        optimizer_0 : optax.GradientTransformation,
        optimizer_1 : optax.GradientTransformation
) -> tuple[Parameters, Parameters]:
    assert len(rollout) % batches == 0
    batch_size = len(rollout) // batches
    opt_state_0 = optimizer_0.init(parameters_0)
    opt_state_1 = optimizer_1.init(parameters_1)
    for epoch in range(epochs):
        rollout = rand.shuffle(rollout)
        batched_rollout = [rollout[i * batch_size : (i+1) * batch_size] for i in range(batches)]
        for batch in batched_rollout:
            batch_0 = [t for t in batch if t.agent == "player_0"]
            batch_1 = [t for t in batch if t.agent == "player_1"]


if __name__ == "__main__":
    k1, k2 = random.split(key, 2)
    p0 = initialize_parameters(k1, ACTOR_LAYER_SIZES, CRITIC_LAYER_SIZES)
    p1 = initialize_parameters(k2, ACTOR_LAYER_SIZES, CRITIC_LAYER_SIZES)
    rollout = collect_trajectories(key, 2048, p0, p1)
    compute_loss_grads(p0, rollout)
#     # print(actor_parameters)
#     # print(critic_parameters)
#     sample_observation = env.observe(env.agents[0])["observation"]
#     print(network_forward(p0.critic, sample_observation))
#     start = time.time()
#     result = network_forward(p0.critic, sample_observation)
#     print(time.time() - start)
#     
#   #  print(rollout)
#     print("ROLLOUT SIZE:", len(rollout))
#     i = 0
#     for x in rollout:
#         i += 1 if x.done else 0
#     print("Num of episodes:", i)
#     rollout = compute_advantages(rollout)
#     print(*(j.advantage for j in rollout))