import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback



gym.register(
    id="CustomCartPole-v0",
    entry_point="custom_cartpole:CustomCartPoleEnv",
    reward_threshold = 475.0,
    max_episode_steps = 500, 
)

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
        # plt.ion()
        # self.fig, self.ax = plt.subplots(figsize=(10, 5))
        # self.line, = self.ax.plot([], [], label='Return')
        # self.ax.set_xlabel('Episode')
        # self.ax.set_ylabel('Return')
        # self.ax.set_title('Return over Episodes')
        # self.ax.grid(True)
        # self.ax.legend()
        # plt.show()

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

                # # Update the plot
                # self.line.set_xdata(range(len(self.episode_returns)))
                # self.line.set_ydata(self.episode_returns)
                # self.ax.relim()
                # self.ax.autoscale_view()
                # self.fig.canvas.draw()
                # self.fig.canvas.flush_events()

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

   

    T = 1000
    dt_multip = 0.3
    max_episode_steps = 500/dt_multip

    # callback =  RewardCallback()
    callback = MeanReturnCallback()

    env = gym.make("CustomCartPole-v0",dt_multip = dt_multip, max_episode_steps=max_episode_steps)
    
    #env = gym.make("CartPole-v1")
    policy_kwargs = dict(net_arch=[64, 64, 64])

    model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1)
    model.learn(total_timesteps=1e5/dt_multip, callback=callback)
    