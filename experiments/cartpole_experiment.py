import sys
import gymnasium as gym


#Import the Custom Envrionment and register it:


gym.register(
    id="CustomCartPole-v0",
    entry_point="../envs/custom_cartpole:CustomCartPoleEnv",
)

mode = str(sys.argv[1])



import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import pandas as pd
from config import experiment_settings #use the config file to change the experiment!



#TODO: Refactor into a function

# Define parameters
n_timesteps = experiment_settings["n_timesteps"] # Number of timesteps for training
policy_kwargs = experiment_settings["policy_kwargs"] # Neural network architecture
seeds = experiment_settings["seeds"] # Testing different initial states
start_delta_exponent = experiment_settings["start_delta_exponent"] # Starting delta exponent multiplier for time step
end_delta_exponent = experiment_settings["end_delta_exponent"] # Ending delta exponent multiplier for time steps
num_deltas = experiment_settings["num_deltas"] # Number of time steps
deltas = np.logspace(start_delta_exponent, end_delta_exponent, num_deltas) # Generate exponentially spaced values 
training_delta = experiment_settings["training_delta"]
checker_delta = experiment_settings["checker_delta"]
desc_color = experiment_settings["desc_color"]  # Green color
reset_color = experiment_settings["reset_color"] # Reset to default color
path_to_google_drive = experiment_settings["path_to_google_drive"]


# Function to train the PPO model on CartPole env
def train_model():
    env = gym.make("CustomCartPole-v0", dt_multip = training_delta)
    model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1)
    model.learn(total_timesteps=n_timesteps)
    model.save("ppo_cartpole")
    env.close()


# Function to test the PPO model on the CartPole env with varying dt_multip
    


#TODO: change this to only incorporate training and plotting training curves
#TODO: add seeds to the envionment and agent
#TODO: Write function that calculates the time_steps needed to get the same wall-clock time.
    
def test_model():
    model = PPO.load("ppo_cartpole")
    returns_mean = []

    for dt in tqdm(deltas, desc="Testing different dt_multip values"):
        env = gym.make("CustomCartPole-v0", dt_multip = dt)
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

    save = input("Do you want to save this experiment? (yes/no): ").strip().lower()
    if save == "yes":
        save_experiment(returns_mean, deltas)


# Function to save experiment results in a unique folder
def save_experiment(returns_mean, deltas):
    base_dir = path_to_google_drive

    # Determine the next experiment number
    existing_experiments = [f for f in os.listdir(base_dir) if f.startswith("exp")]
    next_experiment_num = len(existing_experiments) + 1
    experiment_dir = os.path.join(base_dir, f"exp{next_experiment_num:03d}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the plot as an image in the experiment folder
    plot_path = os.path.join(experiment_dir, "plot.png")
    plt.plot(deltas, returns_mean)
    plt.scatter(deltas, returns_mean, color='red', label="Data Points", zorder=5)
    plt.xlabel("Time Step Multiplier (dt_multip)")
    plt.ylabel("Mean Return")
    plt.title("PPO Performance on CartPole with Varying dt_multip")
    plt.savefig(plot_path)
    plt.close()

    # Save the experiment data in an Excel file
    excel_path = os.path.join(experiment_dir, "experiment_data.xlsx")

    # Save parameters and results in two separate sheets
    with pd.ExcelWriter(excel_path) as writer:
        # Parameters sheet
        parameters_df = pd.DataFrame({
            "Parameter": ["n_timesteps", "policy_net_arch", "seeds", "num_deltas", "start_delta_exponent", "end_delta_exponent", "training_delta", "checker_delta"],
            "Value": [n_timesteps, str(policy_kwargs["net_arch"]), seeds, num_deltas, start_delta_exponent, end_delta_exponent, training_delta, checker_delta]
        })
        parameters_df.to_excel(writer, sheet_name="Parameters", index=False)

        results_df = pd.DataFrame({
            "delta": deltas,
            "mean returns": returns_mean
        })
        results_df.to_excel(writer, sheet_name="Results", index=False)

    print(f"Experiment saved to {experiment_dir}")

# Function to test the environment behaviour 
def code_checker():
    model = PPO.load("ppo_cartpole")
    env = gym.make("CustomCartPole-v0", dt_multip = checker_delta)
    obs, _ = env.reset()
    for step in range(500):
        env.render()
        action, _states = model.predict(obs)
        obs,reward, terminated, truncated, _ = env.step(action)
        print(reward)

    env.close()

if mode == "train":
    train_model()
elif mode == "test":
    test_model()