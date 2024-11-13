import gymnasium as gym
from config import settings
import stable_baselines3
from stable_baselines3 import PPO

print(stable_baselines3.__file__)
exit()


gym.register(
    id="CustomCartPole-v0",
    entry_point="custom_cartpole:CustomCartPoleEnv",
)

# Define parameters
n_timesteps = settings["n_timesteps"] # Number of timesteps for training
policy_kwargs = settings["policy_kwargs"] # Neural network architecture
seeds = settings["seeds"] # Testing different initial states
start_delta_exponent = settings["start_delta_exponent"] # Starting delta exponent multiplier for time step
end_delta_exponent = settings["end_delta_exponent"] # Ending delta exponent multiplier for time steps
num_deltas = settings["num_deltas"] # Number of time steps
deltas = np.logspace(start_delta_exponent, end_delta_exponent, num_deltas) # Generate exponentially spaced values 
training_delta = settings["training_delta"]
checker_delta = settings["checker_delta"]
desc_color = settings["desc_color"]  # Green color
reset_color = settings["reset_color"] # Reset to default color
path_to_google_drive = settings["path_to_google_drive"]

# Function to train the PPO model on CartPole env
def train_model():
    env = gym.make("CustomCartPole-v0", dt_multip = training_delta)
    model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1)
    model.learn(total_timesteps=n_timesteps)
    model.save("ppo_cartpole")
    env.close()