import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.nn import relu, softmax
import mujoco
from mujoco import mjx
import mediapy as media
from MujocoSim import MujocoSim
import numpy as np
# @jit
def actor_network(params, state):
    hidden = relu(jnp.dot(state, params['W1']) + params['b1'])
    logits = jnp.dot(hidden, params['W2']) + params['b2']
    print("logits", logits)
    return softmax(logits)

# @jit
def critic_network(params, state):
    hidden = relu(jnp.dot(state, params['W1']) + params['b1'])
    value = jnp.dot(hidden, params['W2']) + params['b2']
    return value

def initialize_params(input_dim, hidden_dim, output_dim):
    params = {
        'W1': jnp.array(np.random.randn(input_dim, hidden_dim) * 0.01),
        'b1': jnp.zeros(hidden_dim),
        'W2': jnp.array(np.random.randn(hidden_dim, output_dim) * 0.01),
        'b2': jnp.zeros(output_dim)
    }
    return params

input_dim = 27  # Example input dimension
hidden_dim = 128
output_dim_actor = 7  # Number of actions
output_dim_critic = 7  # Single value output

actor_params = initialize_params(input_dim, hidden_dim, output_dim_actor)
critic_params = initialize_params(input_dim, hidden_dim, output_dim_critic)


class ActorCritic:
    def __init__(self, env : MujocoSim, actor_params : dict, critic_params : dict, lr : float=0.01):
        self.env = env
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.lr = lr

    # @jit
    def select_action(self, state):
        prob = actor_network(self.actor_params, state)
        # print(prob)
        # print(prob)
        # action = jax.random.choice(jax.random.PRNGKey(0), len(prob), p=prob)
        # print(action)
        return prob

    # @jit
    def update(self, state: jnp.array, action: jnp.array, reward, next_state: jnp.array):
        # Compute TD target
        value = critic_network(self.critic_params, state)
        next_value = critic_network(self.critic_params, next_state)
        td_target = reward + 0.99 * next_value  # gamma = 0.99
        td_error = td_target - value

        # Update critic
        critic_grads = grad(lambda params: jnp.mean((reward + 0.99 * critic_network(params, next_state) - critic_network(params, state)) ** 2))(self.critic_params)
        self.critic_params = {k: self.critic_params[k] - self.lr * critic_grads[k] for k in self.critic_params}

        # Update actor
        def actor_loss(params):
            pi = actor_network(params, state)
            # print(prob)
            log_pi = jnp.log(pi)
            print(log_pi.shape)
            return -jnp.mean(log_pi * td_error)**2

        actor_grads = grad(actor_loss)(self.actor_params)
        self.actor_params = {k: self.actor_params[k] - self.lr * actor_grads[k] for k in self.actor_params}

    def train(self, episodes=1000):
        for episode in range(episodes):
            
            state = self.env.reset()
            print("train for states", state)
            done = False
            while self.env.isnt_done(state):
                action = self.select_action(state)
                print("train", action)
                next_state, reward = self.env.step(self.env.mjx_model,self.env.mjx_data, action)
                self.update(state, action, reward, next_state)
                state = next_state
    @jit
    def batch_update(self, states, actions, rewards, next_states):
        # Vectorized operations using vmap
        v_update = vmap(self.update, in_axes=(0, 0, 0, 0))
        v_update(states, actions, rewards, next_states)

    @jit
    def batch_select_action(self, states):
        # Vectorized action selection
        v_select = vmap(self.select_action)
        return v_select(states)

    def train_batch(self, batch_size=32, episodes=1000):
        for episode in range(episodes):
            states=self.env.reset()
            # states=self.env.get_state(env.mjx_data)
            print(states)
            done = False
            while not done:
                actions = self.batch_select_action(states)
                next_states, rewards = self.env.batch_step(states, actions)
                self.batch_update(states, actions, rewards, next_states)
                states = next_states



# Instantiate your environment
env = MujocoSim()

# Create ActorCritic instance
ac = ActorCritic(env, actor_params, critic_params)

# # Train the model
# ac.train(episodes=1000)



# Add this method to the ActorCritic class
# ActorCritic.train_batch = train_batch
# ActorCritic.batch_update = batch_update
# ActorCritic.batch_select_action = batch_select_action

# Train the model with batches
# ac.train_batch(batch_size=32, episodes=1000)
ac.train()