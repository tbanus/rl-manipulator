import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.nn import relu, softmax
import mujoco
from mujoco import mjx
import mediapy as media
from MujocoSim import MujocoSim
import numpy as np
import csv
import os
def clip_grads(grads, max_norm):
    norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads.values()))
    clip_coef = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return {k: v * clip_coef for k, v in grads.items()}
# Network definitions
@jit
def actor_network(params, state):
    hidden = relu(jnp.dot(state, params['W1']) + params['b1'])
    logits = jnp.dot(hidden, params['W2']) + params['b2']
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
jax.config.update("jax_debug_nans", True)

class ActorCritic:
    def __init__(self, env : MujocoSim, actor_params : dict, critic_params : dict, lr : float=0.01):
        self.env = env
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.lr = lr

    # @jit
    def select_action(self, state,actor_params):
        return actor_network(actor_params, state)
        # return jnp.ones([7,1])
    # @jit
    def update(self, state, action, reward, next_state, actor_params, critic_params):
        gamma = 0.99  # Discount factor
        max_grad_norm = 1.0  # Adjust as needed

        # Compute TD target
        value = critic_network(critic_params, state)
        next_value = critic_network(critic_params, next_state)
        td_target = reward + gamma * next_value
        td_error = td_target - value


        # print("tderr", td_error)
        # print("value", value)
        # print("reward",reward)
        # Update critic
        def critic_loss(params):
            value = critic_network(params, state)
            return jnp.mean((td_target - value) ** 2)

        # tde=(reward + 0.9 * critic_network(critic_params, next_state) - critic_network(critic_params, state)[0])


        critic_grads = grad(critic_loss)(critic_params)
        critic_grads = clip_grads(critic_grads, max_grad_norm)

        critic_params = {k: critic_params[k] + self.lr * critic_grads[k] for k in critic_params}


        key = jax.random.PRNGKey(0)
        std_devs=jnp.ones([1,7])*50



        def actor_loss(params):
              
            pi=jnp.abs(jax.random.normal(key,(1,7))*std_devs+actor_network(actor_params, state)[0])

            pi=jnp.where(pi>0.01, pi, 0.01)
            log_pi = jnp.log(pi)

            # print("logpi",log_pi)
            return jnp.mean(log_pi * td_error)
      
        actor_grads = grad(actor_loss)(actor_params)
        actor_grads = clip_grads(actor_grads, max_grad_norm)

        # print("actorgrad",actor_grads)
        # print("acc", actor_grads)
        # for k in actor_params:
        #     print(k)
        #     for i in actor_params[k]:
        #         print(i)
        pi = actor_network(actor_params, state)   

        actor_params = {k: actor_params[k]  + self.lr * actor_grads[k] for k in actor_params}
        # actor_params['b2']=actor_params['b2']*pi
        # pi = actor_network(actor_params, state)   
        
        # print(f"pi: {pi}, log_pi{ jnp.log(jnp.where(pi != 0., pi, 0.0001))},")
        return actor_params, critic_params


    def batch_train(self, episodes=1000, batch_size=1024):
        
        select_action=jax.vmap(self.select_action, in_axes=(0,None))
        
   
        update=jax.jit(self.update)
        # update_vmap=self.update
  

        batch=jax.vmap(lambda rng: self.env.mjx_data.replace(ctrl=jax.random.uniform(rng, (8,))))(jax.random.split(jax.random.PRNGKey(0),batch_size))
        get_state=jax.vmap(self.env.get_state)
        step=jax.jit(jax.vmap(self.env.step, in_axes=(None,0,0)))
        for episode in range(episodes):
            
            state = get_state(batch)
            #done = false
            action = select_action(state, self.actor_params)
            next_state, reward, batch = step(self.env.mjx_model, batch, action)
            if episode==0:
              self.log_header(jnp.mean(state, axis=0), jnp.mean(action, axis=0), jnp.mean(reward, axis=0))
            
            #visualize
            renderer = mujoco.Renderer(self.env.mj_model)
            frames=[]
            framerate=100
            print("ep", episode)
            for i in range(100):

                action = select_action(state, self.actor_params)

                next_state, reward, batch = step(self.env.mjx_model, batch, action)
                
                # jax.debug.print(f"next_state.shape {next_state.shape} mean {jnp.mean(next_state, axis=0)}")
                for i in range(batch_size):    
                    actor_params, critic_params = update(state[i], action[i], reward[i], next_state[i], self.actor_params, self.critic_params)
                    for key in actor_params:
                        self.actor_params[key]=actor_params[key]

                    for key in critic_params:
                        self.critic_params[key]=critic_params[key]
                state = next_state
                
    


                self.log_line(jnp.mean(state, axis=0), jnp.mean(action, axis=0), jnp.mean(reward, axis=0))
                
                # batched_mj_data = mjx.get_data(self.env.mj_model, batch)

    def train(self, episodes=100):
   

        for episode in range(episodes):
            
            batch=self.env.mjx_data.replace(ctrl=jnp.ones([8]))
            state = self.env.get_state(self.env.mjx_data)

            done = False
            action = self.select_action(state, self.actor_params)
            next_state, reward, batch = self.env.step(self.env.mjx_model, batch, action)
            if episode==0:
              self.log_header(state,action,reward)
            renderer = mujoco.Renderer(self.env.mj_model)


            frames=[]
            framerate=100
            print("ep", episode)
            for i in range(100):

                action = self.select_action(state, self.actor_params)

                
                next_state, reward, batch = self.env.step(self.env.mjx_model, batch, action)

                # jax.debug.print(f"next_state.shape {next_state.shape} mean {jnp.mean(next_state, axis=0)}")
                for i in range(1):    
                    actor_params, critic_params = self.update(state, action, reward, next_state, self.actor_params, self.critic_params)
                    for key in actor_params:
                        self.actor_params[key]=actor_params[key]

                    for key in critic_params:
                        self.critic_params[key]=critic_params[key]
                state = next_state
                
    

                print(f"action: {action}")
                self.log_line(state,action,reward)
                
                # batched_mj_data = mjx.get_data(self.env.mj_model, batch)    
            # media.show_video(frames, fps=framerate)
    # def update_parameters(self, actor_params, critic_params):

    def log_header(self,state, action, reward):
        header_text=[]
        for i in range(len(state)):
            header_text.append("state"+"_"+str(i) )
        for i in range(len(action)):
            header_text.append("action"+"_"+str(i))
        header_text.append("reward")
        filename="ac_log.csv"

        
        
        user_input = 'y'
        if user_input == 'y': 
            file = 'logs/'+filename
            try:
                os.remove (file)
                data_f = open('logs/'+filename, 'a',newline='')
            except FileNotFoundError:
                data_f = open('logs/'+filename, 'x',newline='')
            # data_f = open('../opy_logs/'+filename, 'a',newline='')
            self.data_writer = csv.writer(data_f)
            Headers = header_text
            print(Headers)
            self.data_writer.writerow(Headers) 
        else:
            data_f = open('logs/'+filename, 'a',newline='')
            data_writer = csv.writer(data_f)
    def log_newline(self):    
        self.log_text=self.log_text+"0\n"        
    def log_line(self, state, action, reward):

        LogList=[]
        # print(f"next_state.shape {state.shape} mean {str(state)}")
        # LogList.append(state)
        for i in state:
            LogList.append(str(i)) 
        for i in action:
            LogList.append(str(i)) 
        LogList.append(reward) 
        self.data_writer.writerow(LogList) 
            
        # Vectorized operations using vmap
        # v_update = vmap(self.update, in_axes=(0, 0, 0, 0))
        # v_update(states, actions, rewards, next_states)
        

    # @jit
    # def batch_select_action(self, states):
    #     # Vectorized action selection
    #     v_select = vmap(self.select_action)
    #     return self.v_select(states)



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
# tr=jax.jit(ac.train)
# tr()

batch_train=jax.jit(ac.batch_train)
# batch_train()
ac.batch_train()
# ac.train()