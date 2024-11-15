import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import PPO



gym.register(
    id="CustomCartPole-v0",
    entry_point="custom_cartpole:CustomCartPoleEnv",
)

from stable_baselines3.common.callbacks import BaseCallback


##train



class RewardCallback(BaseCallback):

    def __init__(self, verbose =0):
        super().__init__()
        self.eps_returns_list = []
        self.eps_return = 0
    
    def _on_training_start(self) -> None:
        print("Ya ali")

    def _on_step(self) -> bool:
        
        self.eps_return += self.locals['rewards']
        if self.locals['dones'] == True:
            self.eps_returns_list.append(self.eps_return)
            self.eps_return =0
       
        return True
    
    def _on_training_end(self) -> None:
        print(np.mean(self.eps_returns_list))
    


if __name__ == "__main__":


    callback =  RewardCallback()

    #env = gym.make("CustomCartPole-v0")
    env = gym.make("CartPole-v1")
    policy_kwargs = dict(net_arch=[64, 64, 64])

    model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1)
    model.learn(total_timesteps=1e5, callback=callback)
    