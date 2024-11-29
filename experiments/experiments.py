import gymnasium as gym
import numpy as np
import sys
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, DQN


#TODO: 1. Make a virtual envrionment
#2. Have all the custom enviornment code files in the venv
#3. Have a requirements.txt file so that people can just easily install ali.

#first argument is the path to save
path = sys.argv[1]
seed = sys.argv[2]
t_multiplier = sys.argv[3]
alpha = sys.argv[4]
alg = sys.argv[5]
env_name = sys.argv[6]
env =  gym.make("CartPole-v1")


#TODO: What to do in the callback
class RewardCallback(BaseCallback):

    def __init__(self, verbose =0):
        super().__init__()
        self.eps_returns_list = []
        self.eps_return = 0
        self.steps_rewards = []
        self.done_status = []
    
    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        self.steps_rewards.append(self.locals['rewards'])
        self.done_status.append(self.locals['dones'])
        self.eps_return += self.locals['rewards']
        if self.locals['dones'] == True:
            self.eps_returns_list.append(self.eps_return)
            self.eps_return =0
       
        return True

    def _on_training_end(self) -> None:
        filename_returns = f"Alg{alg}_env{env_name}_seed{seed}_tmultiplier{t_multiplier}_alpha{alpha}_RETURNS.npy"
        filename_rewards = f"Alg{alg}_env{env_name}_seed{seed}_tmultiplier{t_multiplier}_alpha{alpha}_REWARDS.npy"

        full_returns = os.path.join(path, filename_returns)
        full_rewards = os.path.join(path, filename_rewards)

        np.save(full_returns, self.eps_returns_list)
        np.save(full_rewards, np.column_stack((self.steps_rewards, self.done_status)))

        return True

class Experiment:
    def __init__(self, 
                 model_id:str,
                 env_id:str, 
                 delta_t = 1,
                 gamma = 1,
                 epsilon = 0.3,
                 learning_rate = 0.001,
                 #steps_per_update = 1,
                 policy_kwargs = dict(net_arch=[64,64,64]),
                 device = 'cpu',
                 seed = 0, #TODO: Please think about this?
                 callback = RewardCallback(),
                 total_timesteps = 200_000,
                 max_episode_steps = None, 
                 save_path = path,
                 batch_size = 16,
                 buffer_size = 500, 
                 target_update = 100, #TODO: Find out the default values of these parameters.
                 ):
        
        self.returns = 0
        self.delta_t = delta_t #TODO: How can we compare graphs if we make the delta t as a multiple of the default time step?
        self.gamma = gamma
        self.callback = callback
        self.save_path = save_path
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.learning_rate = learning_rate
        #self.steps_per_update = steps_per_update
        self.policy_kwargs = policy_kwargs
        self.device = device
        self.seed = seed
        self.save_path = '',
        self.buffer_size = buffer_size
        self.target_update = target_update

        self.model_id = model_id
        self.max_episode_steps = max_episode_steps
        self.env_id = env_id
        self.total_timesteps = total_timesteps/self.delta_t


        #make the environment
        self.make_env()

        #define the model
        self.define_model()


    def make_env(self):

        #First register the environment:
        if self.env_id == "CartPole":
            self.env_id = "CustomCartPole-v0"
            gym.register(
            id="CustomCartPole-v0",
            entry_point="custom_cartpole:CustomCartPoleEnv",
            reward_threshold = 475.0,
            max_episode_steps = 500, 
            )
        elif self.env_id == "Acrobot":
            self.env_id ="CustomAcroBot-v1"
            gym.register(
            id="CustomAcroBot-v1",
            entry_point="custom_acrobot:CustomAcrobotEnv",
            reward_threshold = -100,
            max_episode_steps = 500, 
             )
        elif self.env_id == "LunarLander":
            self.env_id = "CustomAcroBot-v1" #TODO: Create CustomLunarLander
            gym.register(
            id="CustomAcroBot-v1",
            entry_point="custom_acrobot:CustomAcrobtEnv",
            reward_threshold = 200,
            max_episode_steps = 1000, 
             )

        self.env = gym.make(self.env_id, 
                        dt_multip = self.delta_t, 
                        max_episode_steps= self.max_episode_steps, 
                        )
        self.env.reset(seed = self.seed)

    def run(self):
        self.model.learn(total_timesteps =self.total_timesteps, callback = self.callback)

    def save_data(self):
        self.model.save(self.save_path) #TODO: Deal with the path after you realize how to run on compute canada.
  
    

    #Please first run make env
    def define_model(self):
        
        if self.model_id == 'PPO':
            self.model = PPO("MlpPolicy", 
                        env=self.env, 
                        verbose=1, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        gamma = self.gamma,
                        device= self.device,
                        batch_size=self.batch_size,
                        exploration_initial_eps = self.epsilon,
                        exploration_final_eps = self.epsilon,
                        )
        elif self.model_id == 'DQN':
            self.model = DQN("MlpPolicy", 
                        env = self.env, 
                        verbose=1, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        buffer_size=self.buffer_size,
                        gamma = self.gamma,
                        device=self.device, 
                        batch_size= self.batch_size,
                        target_update_interval = self.target_update,
                        exploration_initial_eps = self.epsilon,
                        exploration_final_eps = self.epsilon,
                        )




exp = Experiment(model_id = 'DQN',
                 env_id = 'CartPole', 
                 delta_t = 1,
                 gamma = 1,
                 epsilon = 0,
                 learning_rate = 0.001,
                 #steps_per_update = 1,
                 policy_kwargs = dict(net_arch=[64,64,64]),
                 device = 'cpu',
                 seed = int(seed), #TODO: Please think about this?
                 callback = RewardCallback(),
                 total_timesteps = 200,
                 max_episode_steps = None, 
                 save_path = path,
                 batch_size = 16,
                 buffer_size = 500, 
                 target_update = 100, #TODO: Find out the default values of these parameters.
                 )

exp.run()
