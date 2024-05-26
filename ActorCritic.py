import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.nn import relu, softmax
import mujoco
from mujoco import mjx
import mediapy as media
from MujocoSim import MujocoSim
import numpy as np
@jit
def actor_network(params, state):
    hidden = relu(jnp.dot(state, params['W1']) + params['b1'])

    logits = jnp.dot(hidden, params['W2']) + params['b2']
    print(logits.shape, "logits.shaep")
    return softmax(logits)

@jit
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
output_dim_critic = 1  # Single value output

actor_params = initialize_params(input_dim, hidden_dim, output_dim_actor)
critic_params = initialize_params(input_dim, hidden_dim, output_dim_critic)


class ActorCritic:
    def __init__(self, env : MujocoSim, actor_params : dict, critic_params : dict, lr : float=0.01):
        self.env = env
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.lr = lr
        self.an=vmap(actor_network, in_axes=(None,0))

    # @jit
    def select_action(self, state,actor_params):
        rr=actor_network(actor_params, state)
        print(rr.shape, "rrshape")
        return rr

    # @jit
    def update(self, state, action, reward, next_state, actor_params, critic_params):
        # Compute TD target
        value = critic_network(critic_params, state)
        next_value = critic_network(critic_params, next_state)
        td_target = reward + 0.99 * next_value  # gamma = 0.99
        td_error = td_target - value

        # Update critic
        critic_grads = grad(lambda params: jnp.mean((reward + 0.99 * critic_network(params, next_state) - critic_network(params, state)) ** 2))(self.critic_params)
        critic_params = {k: critic_params[k] - self.lr * critic_grads[k] for k in critic_params}

        # Update actor
        def actor_loss(params):
            pi = actor_network(params, state)
            log_pi = jnp.log(pi)
            return -jnp.mean(log_pi * td_error)**2

        actor_grads = grad(actor_loss)(actor_params)
        actor_params = {k: actor_params[k] - self.lr * actor_grads[k] for k in actor_params}
        return actor_params, critic_params
    def bupdate(self, states, actions, rewards, next_states, actor_params, critic_params):
        # Compute TD target
        values = critic_network(critic_params, states)
        next_values = critic_network(critic_params, next_states)
        td_targets = rewards + 0.99 * next_values  # gamma = 0.99
        td_errors = td_targets - values

    # Update critic
        def critic_loss(params, s, ns, r):
            return jnp.mean((r + 0.99 * critic_network(params, ns) - critic_network(params, s)) ** 2)

        critic_grads = grad(critic_loss)(critic_params, states, next_states, rewards)
        critic_params -= self.lr * critic_grads

        # Update actor
        def actor_loss(params, s, a, r, ns):
            pi = actor_network(params, s)
            log_pi = jnp.log(pi)
            return -jnp.mean(log_pi * (r + 0.99 * critic_network(critic_params, ns) - critic_network(critic_params, s)) ** 2)

        actor_grads = grad(actor_loss)(actor_params, states, actions, rewards, next_states)
        actor_params -= self.lr * actor_grads

        return actor_params, critic_params

    def train(self, episodes=1000):
        select_action_vmap=jax.vmap(self.select_action, in_axes=(0,None))
        step_vmap=jax.vmap(self.env.step, in_axes=(None,0,0))
   
        update_vmap=jax.vmap(self.update, in_axes=(0,0,0,0,None,None))
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng,1024)
        fun=lambda rng: self.env.mjx_data.replace(ctrl=jax.random.uniform(rng, (8,)))
        fun_vmapped = jax.vmap(fun)
        batch=fun_vmapped(rng)
        reset_vmap=jax.vmap(self.env.get_state)
        step_jit=jax.jit(step_vmap)
        for episode in range(episodes):
            
            state = reset_vmap(batch)
            done = False
            action = select_action_vmap(state, self.actor_params)
            next_state, reward = step_jit(self.env.mjx_model, batch, action)

            # self.log_header(jnp.mean(state))
            # self.log_header(jnp.mean(action))
            # self.log_header(jnp.mean(reward))


            # self.log_newline()
            while self.env.isnt_done(state):
                print("go")
                action = select_action_vmap(state, self.actor_params)
                print("check1")
                # print(batch)
                # print(action)

                print(action.shape)
                next_state, reward = step_jit(self.env.mjx_model, batch, action)

                print("end step, begin update")
                actor_params, critic_params = update_vmap(state, action, reward, next_state, self.actor_params, self.critic_params)
                "end_update"
                state = next_state

                print("state", type(state), state.shape)
                for key in actor_params:
                    self.actor_params[key]=jnp.mean(actor_params[key],axis=0)
                    # print(actor_params[key].shape)
                    # print(self.actor_params[key].shape)

                for key in critic_params:
                    self.critic_params[key]=jnp.mean(critic_params[key],axis=0)

                print("reward", type(reward), reward.size)
                print(action.shape)
                # self.log_line()
                # self.log_newline()

                # self.log(jnp.mean(state), jnp.mean(action), jnp.mean(reward), [jnp.mean(actor_params[key]) for key in actor_params],[jnp.mean(critic_params[key]) for key in critic_params])
    def log_header(self,item):
        for i in range(len(item)):
            self.log_text=self.log_text+item.__str__+"_"+str(i)+", " 
    def log_newline(self):    
        self.log_text=self.log_text+"0\n"
        
    def log_line(self, item):
        for i in item:
            self.log_text=self.log_text+str(i)+", " 
        # Vectorized operations using vmap
        # v_update = vmap(self.update, in_axes=(0, 0, 0, 0))
        # v_update(states, actions, rewards, next_states)
        

    # @jit
    # def batch_select_action(self, states):
    #     # Vectorized action selection
    #     v_select = vmap(self.select_action)
    #     return self.v_select(states)

    def train_batch(self, batch_size=32, episodes=1000):
        for episode in range(episodes):
            states=self.env.reset()
            done = False
            while not done:
                print(states)
                actions = self.batch_select_action(states)
                next_states, rewards = self.env.batch_step(states, actions)
                self.batch_update(states, actions, rewards, next_states)
                states = next_states
                done=self.env.isnt_done()



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