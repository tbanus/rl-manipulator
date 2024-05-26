import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax import grad, jit, vmap
from jax.nn import relu, softmax
import mujoco
from mujoco import mjx
import mediapy as media
import mujoco.mjx

class MujocoSim:
    def __init__(self):
        # Initialize your environment
        # Make model, data, and renderer
        
        # states
        
        # 0-5: robot arm joints pos
        # 6: finger pos
        # 7-9: box pos
        # 10-13: box quat
        # 14-20: robot arm joints vel
        # 21: finger vel
        # 22-24: box vel
        # 25-27: box angular vel

        # actions
        # 0-5 :robot arm joints
        # 6: finger torque

        self.mj_model = mujoco.MjModel.from_xml_path('simple_arm/scene.xml')
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self.mj_model.opt.iterations = 6
        self.mj_model.opt.ls_iterations = 6
        self.mj_data = mujoco.MjData(self.mj_model)
        # renderer = mujoco.Renderer(mj_model)
        # weight_load_target_dist_reward = 1
        # weight_tip_to_load_position_reward = 1
        # weight_tip_to_load_velocity_reward = 1
        # weight_current_torque_cost= 1
        # weight_peak_torque_cost= 1
        # weight_timestep = 1
        self.weights=jnp.array([1,1,0,1,1,1]).transpose()
        self.load_dest=jnp.array([1,1,1]).transpose()

        self.max_allowable_distance=4
        self.max_allowable_target_error=0.1
        # self.peak_torque=0

        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # self.p=jnp.zeros([3,6]) #TODO 
        # self.J=jnp.zeros([3,6])
    def reset(self):
        # Reset the environment to the initial state
        self.mj_model = mujoco.MjModel.from_xml_path('simple_arm/scene.xml')
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self.mj_model.opt.iterations = 6
        self.mj_model.opt.ls_iterations = 6
        self.mj_data = mujoco.MjData(self.mj_model)
        # renderer = mujoco.Renderer(mj_model)

        self.weights=jnp.array([1,1,1,1,1,0.1]).transpose()
        self.load_dest=jnp.array([1,1,1]).transpose()



        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        
        return self.get_state(self.mjx_data)
    
    # @jax.vmap
    def step(self, model, data, action):
        # Execute the action and return the new state and reward
        # rng = jax.random.PRNGKey(0)
        # rng = jax.random.split(rng,1024)
        # print(rng.shape)

        # fun=lambda rng: mjx_data.replace(ctrl=jax.random.uniform(rng, (8,)))
        # fun_vmapped = jax.vmap(fun)
        # batch=fun_vmapped(rng)
        data.replace(ctrl=action)
        data=mjx.step(model, data)
        state=self.get_state(data)
        # self.peak_torque=jnp.max(jnp.array([self.peak_torque, norm(jnp.array([action[0:6]]))**2]))
        return state, self.get_reward(state,action)
           
        # fun_vmapped = jax.vmap(step, in_axes=(None,0,0))
        # batch=fun_vmapped(mjx_model, batch,rng)
        # print("next, I shall jit step")
        # jit_step = jax.jit(jax.vmap(step, in_axes=(None, 0,0)))
        # batch = jit_step(mjx_model, batch,rng)


    def batch_step(self, states, actions):
        batch = self.step(self.mjx_model, self.mjx_data, actions)
    
    def get_reward(self, state, action):
        #update peak torque
        
        
        rewards= sum([self.weights[0]*norm(state[7:10]-self.load_dest), 
                    self.weights[1]*norm(state[7:10]-self.mjx_data.geom_xpos[16]),
                    self.weights[2]*norm(self.mjx_data.geom_xpos[16]-state[22:25]) ,
                    self.weights[3]*norm(action[0:6])**2 ,
                    self.weights[4]*1])
        return rewards
        
    def get_state(self,data : mujoco.mjx.Data):

        state=jnp.concatenate([data.qpos[0:7],data.qpos[7:14],data.qvel[0:7], data.qpos[7:13]])
        # print(state.shape)
        return jnp.array(state)
    def isnt_done(self,state):
        rb = jnp.array(state[7:10]).transpose()
        rd=self.load_dest
        rm=jnp.array([0,0,0]).transpose()

        a=max([self.max_allowable_distance-norm(rb-rm),0]) 
            
        b=max([norm(rb-rd)-self.max_allowable_target_error,0])
        return a*b