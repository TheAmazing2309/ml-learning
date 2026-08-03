# PPO Implementation Plan (Single Agent, PyTorch)

Target env: `FrozenLake-v1` (8x8), `Discrete` observation space, `Discrete` action
space. Plan assumes what you've already built: one-hot encoding of observations,
separate actor/critic `Sequential` networks, `Categorical` distribution over
actor logits. No code here — this is the checklist of pieces to build and how
they fit together.

## 1. Nail down the two networks

- Actor: maps encoded observation -> action logits (size = `act_size`). No
  activation on the final layer — logits are consumed by `Categorical`, not by you.
- Critic: maps encoded observation -> a single scalar (the estimated value of
  that state). No activation on the final layer either — value is unbounded.
- Decide now: separate networks (what you have) vs a shared trunk with two
  heads. Separate is simpler to reason about and totally standard for a first
  PPO — stick with it unless you have a specific reason to share.
- Each network needs its own optimizer, or one optimizer over the combined
  parameter list (`list(actor.parameters()) + list(critic.parameters())`).
  Either is fine; pick one and be consistent.

## 2. Rollout collection (the "experience" phase)

For a fixed number of steps (or fixed number of full episodes, your choice —
FrozenLake episodes are short, so full episodes are reasonable), interact with
the environment and store, per timestep:

- the observation (encoded, ready to feed a network again later)
- the action taken
- the log-probability of that action under the policy *at the time it was
  taken* (this is your "old" log-prob — critical, don't recompute it later
  with an updated policy)
- the value estimate from the critic at that state, at the time
- the reward received
- whether the episode terminated at that step

You already have the skeleton for this loop. The key discipline: everything
you store must reflect the policy *as it was when the action was chosen*.
PPO's whole mechanism depends on comparing "old" vs "new" policy on the same
stored actions.

## 3. Compute returns and advantages after the rollout ends

This happens once per rollout, after all trajectories are collected, before
any gradient step.

- **Returns**: the discounted sum of future rewards from each timestep
  onward. Standard discount factor `gamma` (0.99 is a typical default, but for
  an episodic sparse-reward env like FrozenLake you may want to experiment).
- **Advantages**: how much better an action was than the critic's baseline
  expectation for that state. The standard choice for PPO is **Generalized
  Advantage Estimation (GAE)**, controlled by `gamma` and a second parameter
  `lambda` (0.95 typical). GAE is a weighted combination of multi-step
  TD-errors — look up the recursive formula
  (`delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`, then advantages are an
  exponentially-weighted sum of these deltas going backward through the
  trajectory). You compute this walking **backward** through each trajectory.
- Handle episode boundaries carefully: when an episode ends, there is no
  "next state" to bootstrap from, so the bootstrap value there should be
  treated as 0 (or you restart the backward recursion at each episode
  boundary if a rollout contains multiple episodes concatenated).
- After computing advantages, **normalize them** (subtract mean, divide by
  std across the whole batch) — this is a small detail but it matters a lot
  for PPO's stability in practice.

## 4. The PPO update phase

This is where you actually train, and it happens for several epochs over the
same rollout data (unlike vanilla policy gradient, which uses each rollout
once). For each epoch:

- Shuffle the rollout data and split it into minibatches.
- For each minibatch, run the **current** (being-updated) actor and critic on
  the stored observations to get:
  - new log-probs for the stored actions
  - new value estimates
  - entropy of the current action distribution
- Compute the probability ratio: `ratio = exp(new_log_prob - old_log_prob)`.
  This measures how much the policy has shifted for that specific
  state-action pair since the rollout was collected.
- Compute the **clipped surrogate objective**: the minimum of
  `ratio * advantage` and `clip(ratio, 1 - epsilon, 1 + epsilon) * advantage`,
  where `epsilon` is a small hyperparameter (0.2 is the standard default).
  This clipping is the core trick of PPO — it discourages the policy from
  moving too far in one update, without needing a hard trust-region
  constraint like TRPO.
- Compute the **value loss**: typically mean-squared-error between the new
  value predictions and the stored returns. Optionally, PPO papers also clip
  the value function update similarly to the policy ratio — this is a common
  variant, worth knowing about but not mandatory for a first pass.
- Compute the **entropy bonus**: the mean entropy of the current action
  distribution across the minibatch. This gets **added** to the loss (with a
  small positive coefficient) to encourage continued exploration and prevent
  the policy from collapsing too early.
- Combine into one total loss:
  `loss = -clipped_surrogate_objective + value_loss_coef * value_loss - entropy_coef * entropy`
  (signs matter: you're maximizing the surrogate objective and entropy, so
  they get negated since you're minimizing `loss`; value loss is already a
  thing you want to minimize, so it stays positive).
- Zero gradients, backpropagate, optionally clip gradient norm (`clip_grad_norm_`
  is standard in PPO implementations, typically to a max norm around 0.5),
  then step the optimizer(s).

## 5. Hyperparameters to expose (don't hardcode magic numbers inline)

- `gamma` (discount factor)
- `lambda` (GAE parameter)
- `epsilon` (clip range)
- `value_loss_coef`
- `entropy_coef`
- learning rate(s)
- number of epochs per rollout
- minibatch size
- rollout length / number of episodes per rollout
- max gradient norm

## 6. Logging / sanity checks while developing

- Track average episode return over time — should trend upward, even if
  noisily, for FrozenLake.
- Track the average ratio value during updates — if it's wildly far from 1.0
  often, your clipping is doing a lot of work and something about learning
  rate or epoch count may be too aggressive.
- Track entropy over training — should start high (close to uniform over 4
  actions) and gradually decrease as the policy commits. If it collapses to
  near-zero almost immediately, entropy coefficient is probably too low or
  learning rate too high.
- Track value loss — should generally trend down, though it can be noisy
  early on.

## 7. Things specific to FrozenLake worth anticipating

- Reward is sparse (0 everywhere except reaching the goal), so early
  training will look like near-total noise for a while — don't mistake that
  for a bug. Consider `is_slippery` setting when creating the env; the
  default stochastic transitions make credit assignment harder, useful to
  know if debugging feels unusually hard.
- Episodes can end in either success (goal reached) or failure (fell in a
  hole) or truncation (max steps) — make sure your bootstrap-value-at-done
  logic doesn't accidentally treat a truncation the same as a true terminal
  state if you want to be fully correct (bootstrapping from the critic's
  value estimate is the technically correct behavior on truncation, since the
  episode didn't actually end from the environment's perspective — it's a
  detail many implementations skip, up to you whether to handle it).

## Suggested build order

1. Rollout collection loop storing all required per-step data.
2. GAE + returns computation as a standalone pass over a finished rollout.
3. PPO loss computation, tested first on a single minibatch by hand/print
   statements before wiring up the full multi-epoch loop.
4. Full training loop: collect rollout -> compute advantages -> multiple
   epochs of minibatch updates -> repeat.
5. Logging on top once the loop runs end-to-end.
