import gymnasium as gym
""" print(gym.__file__)
exit() """
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Define parameters
n_timsteps = 500_000 # Number of timesteps for training
policy_kwargs = dict(net_arch=[64, 64, 64]) # Neural network architecture
seeds = 100 # Testing different initial states
start_delta_exponent = -15 # Starting delta exponent multiplier for time step
end_delta_exponent = 0 # Ending delta exponent multiplier for time steps
num_deltas = 15 # Number of time steps
deltas = np.logspace(start_delta_exponent, end_delta_exponent, num_deltas) # Generate exponentially spaced values 
training_delta = 1e-10
checker_delta = 1e-10


# Function to train the PPO model on CartPole env
def train_model():
    env = gym.make("CartPole-v1", dt_multip = training_delta)
    model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1)
    model.learn(total_timesteps=n_timsteps)
    model.save("ppo_cartpole")
    env.close()


# Function to test the PPO model on the CartPole env with varying dt_multip

def test_model():
    desc_color = "\033[92m"  # Green color
    reset_color = "\033[0m"  # Reset to default color

    model = PPO.load("ppo_cartpole")
    returns_mean = []

    for dt in tqdm(deltas, desc="Testing different dt_multip values"):
        env = gym.make("CartPole-v1", dt_multip = dt)
        returns = [] # store returns for each seed

        for seed in tqdm(range(seeds), desc=f"{desc_color}Testing with dt_multip = {dt} {reset_color}"):
            _return = 0
            obs, _ = env.reset()

            # Run one episode (up to 500)
            for step in range(500):
                
                action, _states = model.predict(obs, deterministic=True)
                obs,reward, terminated, truncated, _ = env.step(action)
                _return += reward

                '''For checking with random policy'''
                #obs,reward, terminated, trunctated, _ = env.step(env.action_space.sample())
                # if terminated or truncated:
                #     reward = 0
            
            
            returns.append(_return)
        mean = np.mean(returns)
        returns_mean.append(mean)  
        env.close()
    plt.plot(deltas, returns_mean)
    plt.scatter(deltas, returns_mean, color='red', label="Data Points", zorder=5)
    plt.xlabel("Time Step Multiplier (dt_multip)")
    plt.ylabel("Mean Return")
    plt.title("PPO Performance on CartPole with Varying dt_multip")
    plt.show()



# Function to test the environment behaviour 
def code_checker():
    model = PPO.load("ppo_cartpole")
    env = gym.make("CartPole-v1", dt_multip = checker_delta)
    obs, _ = env.reset()
    for step in range(500):
        env.render()
        action, _states = model.predict(obs)
        obs,reward, terminated, truncated, _ = env.step(action)
        print(reward)

    env.close()



test_model()