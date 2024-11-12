import sys
import gymnasium as gym
from config import settings
from stable_baselines3 import PPO


n_timesteps = settings["n_timesteps"]
policy_kwargs = settings["policy_kwargs"]
training_delta = settings["training_delta"]

def train_model():
    env = gym.make("CustomCartPole-v0", dt_multip = training_delta)
    model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1)
    model.learn(total_timesteps=n_timesteps)
    model.save("ppo_cartpole")
    env.close()