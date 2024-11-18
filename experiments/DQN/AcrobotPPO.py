import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
import os

#TODO: CustomAcrobot
gym.register(
    id="CustomAcrobot-v1",
    entry_point="custom_acrobot:CustomAcrobotEnv",
    reward_threshold=-100.0,  # Success if cumulative reward is better than -100
    max_episode_steps=500,    # Default max steps (adjust if dt changes significantly)
)


# Function to create environments
def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env = gym.make(env_id,dt_multip = dt_multip, max_episode_steps=max_episode_steps) #"CustomAcrobot-v1"

        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init



class MeanReturnCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MeanReturnCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_timestamps = []
        self.total_timesteps = 0

    def _on_training_start(self) -> None:
        # Initialize the episode rewards after the environment is set
        self.episode_rewards = [[] for _ in range(self.training_env.num_envs)]
        # Initialize the plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.line, = self.ax.plot([], [], label='Return')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Return')
        self.ax.set_title('Return over Episodes')
        self.ax.grid(True)
        self.ax.legend()
        plt.show()

    def _on_step(self) -> bool:
        # Get rewards and done flags for each environment
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for i in range(len(rewards)):
            self.episode_rewards[i].append(rewards[i])
            if dones[i]:
                # Compute episode return
                episode_return = np.sum(self.episode_rewards[i])
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(len(self.episode_rewards[i]))
                self.episode_timestamps.append(self.num_timesteps)
                self.episode_rewards[i] = []  # Reset rewards for the next episode

                # Update the plot
                self.line.set_xdata(range(len(self.episode_returns)))
                self.line.set_ydata(self.episode_returns)
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        return True

    def _on_training_end(self) -> None:
        # Calculate mean return over all episodes
        mean_return = np.mean(self.episode_returns)
        print(f"\nMean return over all episodes: {mean_return}")

        # Compute area under the curve (AUC)
        auc = np.trapz(self.episode_returns)
        print(f"Area under the curve (AUC): {auc}")

        # Final performance (mean return over the last N episodes)
        N = 100  # Adjust N as needed
        if len(self.episode_returns) >= N:
            final_performance = np.mean(self.episode_returns[-N:])
            print(f"Final performance (mean return over last {N} episodes): {final_performance}")
        else:
            print("Not enough episodes for final performance calculation.")

        # Keep the plot open after training ends
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    # Configuration
    # env_id = "Acrobot-v1"
    env_id = "CustomAcrobot-v1"
    num_cpu = 8  # Number of parallel environments
    total_timesteps = int(1e6)  # Total timesteps for training

    dt_multip = 0.1
    max_episode_steps = 500/dt_multip

    # Create vectorized environments
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # Callback for tracking returns
    callback = MeanReturnCallback()

    # Define the policy network architecture
    policy_kwargs = dict(net_arch=[64, 64, 64])

    # Check if model exists
    model_path = "PPOAcrobot.zip"
    if not os.path.exists(model_path):
        print("Model file not found. Training a new model...")
        # PPO Configuration
        print("Training PPO...")
        ppo_model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=0.001,
            batch_size=256,
            n_steps=256,
            gamma=0.99,
            gae_lambda=0.94,
            ent_coef=0.0,
            n_epochs=4,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        ppo_model.learn(total_timesteps=total_timesteps, callback=callback)

        ppo_model.save(model_path)

    else:
        print("Loading existing model...")
        ppo_model = PPO.load(model_path)


    # # DQN Configuration
    # print("Training DQN...")
    # dqn_model = DQN(
    #     "MlpPolicy",
    #     vec_env,
    #     learning_rate=0.00063,
    #     batch_size=128,
    #     buffer_size=50000,
    #     gamma=0.99,
    #     exploration_fraction=0.12,
    #     exploration_final_eps=0.1,
    #     target_update_interval=250,
    #     train_freq=4,
    #     verbose=1
    # )
    # dqn_model.learn(total_timesteps=total_timesteps, callback=callback)

    # # A2C Configuration
    # print("Training A2C...")
    # a2c_model = A2C(
    #     "MlpPolicy",
    #     vec_env,
    #     learning_rate=0.0007,
    #     n_steps=5,
    #     gamma=0.99,
    #     vf_coef=0.25,
    #     ent_coef=0.01,
    #     max_grad_norm=0.5,
    #     verbose=1
    # )
    # a2c_model.learn(total_timesteps=total_timesteps, callback=callback)


    # # Load the trained PPO model
    # print("Loading trained PPO model...")
    # ppo_model = PPO.load("PPOAcrobot.zip")


    # Render Environment After Training
    print("Rendering trained agent...")
    render_env = gym.make(env_id, render_mode="human")  # Specify render mode
    obs, _ = render_env.reset()  # Unpack the tuple returned by reset()
    for _ in range(3000):  # Adjust the number of steps as needed
        action, _states = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)  # Unpack all values
        if terminated or truncated:  # Check if the episode has ended
            obs, _ = render_env.reset()  # Reset and unpack
    render_env.close()
