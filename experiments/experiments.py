import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

env =  gym.make("CartPole-v1")

#print(gym.__dict__)
print(vars(env))


class RewardCallback(BaseCallback):

    def __init__(self, verbose =0):
        super().__init__()
        self.eps_returns_list = []
        self.eps_return = 0
    
    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        
        self.eps_return += self.locals['rewards']
        if self.locals['dones'] == True:
            self.eps_returns_list.append(self.eps_return)
            self.eps_return =0
       
        return True

    def _on_training_end(self) -> None:
        print(np.mean(self.eps_returns_list))

class Experiment:
    def __init__(self, model, env_id, delta_t = 1,
                 gamma = 1,
                 learning_rate = 0.001,
                 steps_per_update = 1,
                 policy_kwargs = dict(net_arc=[64,64,64]),
                 device = 'cpu',
                 seed = 0,
                 callback = RewardCallback(),
                 total_timesteps = 1e5,
                 ):
        
        self.returns = 0
        self.delta_t = delta_t
        self.gamma = gamma

        self.learning_rate = 0.0001,
        self.steps_per_update = steps_per_update
        self.policy_kwargs = policy_kwargs
        self.device = device
        self.seed = seed

        self.model = model

        self.max_episode_steps = 0
        self.env_id = env_id

        self.total_timesteps = total_timesteps/self.delta_t


    def make_env(self):
        return gym.make(self.env_id, dt_multip = self.delta_t, max_episode_steps= self.max_episode_steps)

    def run(self):
        self.model.learn(total_timesteps =self.total_timesteps, callback = self.callback)

    def save_data(self, path: str):
        self.model.save(path)

    
    #Different types of plotting:
    #1. Plots of single runs
    #2. Average plot with p-percentile confidence interval
    
    def plot_results(self):
        pass

    def calculate_performance_metrics(self):
        pass

    def define_agent(self):
        pass

