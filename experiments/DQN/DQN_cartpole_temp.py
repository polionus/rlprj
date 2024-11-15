import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id)
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
    env_id = "CartPole-v1"
    num_cpu = 32
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # model = DQN(
    #     "MlpPolicy",
    #     vec_env,
    #     learning_rate=0.001,
    #     gamma=0.99,
    #     buffer_size=100000,
    #     batch_size=64,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     target_update_interval=500,
    #     train_freq=1,
    #     gradient_steps=1,
    #     verbose=1
    # )

    model = PPO("MlpPolicy", vec_env, verbose=1)

    mean_return_callback = MeanReturnCallback()
    model.learn(total_timesteps=int(4e4), callback=mean_return_callback)

    # model.save("CartPole")


    # obs = vec_env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     vec_env.render()
