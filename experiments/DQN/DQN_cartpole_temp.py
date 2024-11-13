import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class MeanReturnCallback(BaseCallback):
    def __init__(self, verbose=0, plot_interval=100):
        super(MeanReturnCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_returns = []
        self.plot_interval = plot_interval

    def _on_step(self) -> bool:
        self.episode_rewards.append(self.locals["rewards"])

        if np.any(self.locals["dones"]):
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    episode_return = np.sum(self.episode_rewards)
                    self.episode_returns.append(episode_return)
                    self.episode_rewards = []
                    # Plot periodically
                    if len(self.episode_returns) % self.plot_interval == 0:
                        self.plot_returns()

        return True

    def plot_returns(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_returns, label="Episode Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Episode Return During Training")
        plt.legend()
        plt.grid()
        plt.show()

    def _on_training_end(self) -> None:
        mean_return = np.mean(self.episode_returns)
        print(f"Mean return over all episodes: {mean_return}")

        # Calculate AUC using the trapezoidal rule
        auc = trapz(self.episode_returns, dx=1)
        print(f"Area Under the Curve (AUC) of Returns: {auc}")

        # Final Performance: Plot all returns and AUC
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_returns, label="Episode Return")
        plt.fill_between(range(len(self.episode_returns)), self.episode_returns, alpha=0.2)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f"Final Performance: Mean Return={mean_return:.2f}, AUC={auc:.2f}")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    env_id = "CartPole-v1"
    num_cpu = 24
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = DQN("MlpPolicy", vec_env, verbose=1)
    mean_return_callback = MeanReturnCallback(plot_interval=100)
    model.learn(total_timesteps=int(1e5), callback=mean_return_callback)




# import gymnasium as gym

# from stable_baselines3 import PPO
# from stable_baselines3 import DQN

# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed

# from stable_baselines3.common.callbacks import BaseCallback
# import numpy as np


# def make_env(env_id: str, rank: int, seed: int = 0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: the environment ID
#     :param num_env: the number of environments you wish to have in subprocesses
#     :param seed: the initial seed for RNG
#     :param rank: index of the subprocess
#     """
#     def _init():
#         env = gym.make(env_id, render_mode="human")
#         env.reset(seed=seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init

# class MeanReturnCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(MeanReturnCallback, self).__init__(verbose)
#         self.episode_rewards = []
#         self.episode_returns = []

#     def _on_step(self) -> bool:
#         # Gather rewards at each step
#         self.episode_rewards.append(self.locals["rewards"])

#         # Check if the episode has ended
#         if np.any(self.locals["dones"]):
#             # Compute episode return for each environment in vectorized env
#             for idx, done in enumerate(self.locals["dones"]):
#                 if done:
#                     episode_return = np.sum(self.episode_rewards)
#                     self.episode_returns.append(episode_return)
#                     self.episode_rewards = []  # Reset rewards for new episode

#         return True

#     def _on_training_end(self) -> None:
#         # Calculate mean return over all episodes
#         mean_return = np.mean(self.episode_returns)
#         print(f"Mean return over all episodes: {mean_return}")

# if __name__ == "__main__":
#     # env_id = "CartPole-v1"
#     # num_cpu = 24  # Number of processes to use
#     # Create the vectorized environment
#     # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

#     # Stable Baselines provides you with make_vec_env() helper
#     # which does exactly the previous steps for you.
#     # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
#     # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

#     # model = PPO("MlpPolicy", vec_env, verbose=1)
#     # model.learn(total_timesteps=25_000)

#     # Instantiate the agent
#     # model = DQN("MlpPolicy", vec_env, verbose=1)
#     # Train the agent and display a progress bar
#     # model.learn(total_timesteps=int(2e5), progress_bar=True)


#     # ---------------
#     # Instantiate the environment and the model
#     env_id = "CartPole-v1"
#     num_cpu = 24
#     vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

#     model = DQN("MlpPolicy", vec_env, verbose=1)

#     # Train the agent with the custom callback
#     mean_return_callback = MeanReturnCallback()
#     model.learn(total_timesteps=int(1e5), callback=mean_return_callback)

#     # obs = vec_env.reset()
#     # for _ in range(1000):
#     #     action, _states = model.predict(obs)
#     #     obs, rewards, dones, info = vec_env.step(action)
#     #     vec_env.render()








