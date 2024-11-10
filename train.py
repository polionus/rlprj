import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


n_timsteps = 500_000

policy_kwargs = dict(net_arch=[64, 64, 64])

env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,dt = 10)

model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1)
model.learn(total_timesteps=n_timsteps)
model.save("ppo_cartpole")