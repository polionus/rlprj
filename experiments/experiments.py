import argparse
import zipfile
import gymnasium as gym
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, DQN, A2C
import pandas as pd


#TODO: 1. Make a virtual envrionment
#2. Have all the custom enviornment code files in the venv
#3. Have a requirements.txt file so that people can just easily install ali.


#TODO: What to do in the callback
class RewardCallback(BaseCallback):

    def __init__(self, model_id, env_id, seed, delta_t, alpha, path, task_ID, verbose=0):
        super().__init__(verbose)
        self.model_id = model_id
        self.env_id = env_id
        self.seed = seed
        self.delta_t = delta_t
        self.alpha = alpha
        self.path = path
        self.task_ID = task_ID  # Add task_ID
        self.eps_returns_list = []
        self.eps_return = 0
        self.steps_rewards = []
        self.done_status = []
    
    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        #WTF Episodes??!!
        self.steps_rewards.append(self.locals['rewards'])
        self.done_status.append(self.locals['dones'])
        self.eps_return += self.locals['rewards']
        if self.locals['dones'] == True:
            self.eps_returns_list.append(self.eps_return[0])
            self.eps_return =0
       
        return True

    def _on_training_end(self) -> None:
        filename_returns = f"Alg{self.model_id}_env{self.env_id}_seed{self.seed}_tmultiplier{self.delta_t}_alpha{self.alpha}_RETURNS.npy"
       
        full_returns_path = os.path.join(self.path, filename_returns)
        

        np.savez_compressed(full_returns_path, returns = self.eps_returns_list, rewards =np.column_stack((self.steps_rewards, self.done_status)) )

        return True
     

class Experiment:
    def __init__(self, 
                 model_id:str,
                 env_id:str, 
                 delta_t = 1,
                 gamma = 1,
                 epsilon = 0.3,
                 learning_rate = 0.001,
                 #steps_per_update = 1,
                 policy_kwargs = dict(net_arch=[64,64,64]),
                 device = 'cpu',
                 seed = 0, #TODO: Please think about this?
                #  callback = RewardCallback(),
                 total_timesteps = 200_000,
                 #max_episode_steps = None, 
                 save_path = None,
                 batch_size = 16,
                 buffer_size = 500, 
                 target_update = 100, #TODO: Find out the default values of these parameters.
                 task_ID="01",  # Add task_ID parameter
                 ):
        
        self.returns = 0
        self.delta_t = delta_t #TODO: How can we compare graphs if we make the delta t as a multiple of the default time step?
        self.gamma = gamma
        # self.callback = callback
        self.save_path = save_path
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.learning_rate = learning_rate
        #self.steps_per_update = steps_per_update
        self.policy_kwargs = policy_kwargs
        self.device = device
        self.seed = seed

        self.buffer_size = buffer_size
        self.target_update = target_update

        self.model_id = model_id
        #self.max_episode_steps = max_episode_steps
        self.env_id = env_id
        self.total_timesteps = total_timesteps/self.delta_t
        self.task_ID = task_ID  # Store task_ID

        # Initialize the callback with required parameters
        self.callback = RewardCallback(
            model_id=self.model_id,
            env_id=self.env_id,
            seed=self.seed,
            delta_t=self.delta_t,
            alpha=self.learning_rate,
            path=self.save_path,
            task_ID=self.task_ID,  # Pass task_ID to the callback
        )

        #make the environment
        self.make_env()

        #define the model
        self.define_model()

    def make_env(self):

        #First register the environment:
        if self.env_id == "CartPole":
            self.env_id = "CustomCartPole-v0"
            gym.register(
            id="CustomCartPole-v0",
            entry_point="custom_cartpole:CustomCartPoleEnv",
            reward_threshold = 500,
            max_episode_steps = 500/self.delta_t, 
            )
        elif self.env_id == "Acrobot":
            self.env_id = "Acrobot-v1"
            pass
            self.env_id ="CustomAcroBot-v1"
            gym.register(
            id="CustomAcroBot-v1",
            entry_point="custom_acrobot:CustomAcrobotEnv",
            reward_threshold = -100,
            max_episode_steps = 500/self.delta_t, 
             )
            #TODO: change to MC
        elif self.env_id == "MountainCar":
            self.env_id = "CustomMountainCar-v0" #TODO: Create CustomLunarLander
            gym.register(
            id="CustomMountainCar-v0",
            entry_point="custom_mountain_car:CustomMountainCarEnv",
            reward_threshold = 200,
            max_episode_steps = 200/self.delta_t, 
            )

        #self.env = gym.make(self.env_id, 
                        #dt_multip = self.delta_t, 
                        #max_episode_steps= self.max_episode_steps, 
                        #)
        self.env = gym.make(self.env_id,max_episode_steps= 500)
        self.env.reset(seed = self.seed)

    def run(self):
        self.model.learn(total_timesteps =self.total_timesteps, callback = self.callback)

    def save_data(self):
        self.model.save(self.save_path) #TODO: Deal with the path after you realize how to run on compute canada.
  
    

    #Please first run make env
    def define_model(self):
        
        if self.model_id == 'PPO':
            self.model = PPO("MlpPolicy", 
                        env=self.env, 
                        verbose=1, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        gamma = self.gamma,
                        device= self.device,
                        batch_size=self.batch_size,
                        # exploration_initial_eps = self.epsilon,
                        # exploration_final_eps = self.epsilon,
                        )
        elif self.model_id == 'DQN':
            self.model = DQN("MlpPolicy", 
                        env = self.env, 
                        verbose=1, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        buffer_size=self.buffer_size,
                        gamma = self.gamma,
                        device=self.device, 
                        batch_size= self.batch_size,
                        target_update_interval = self.target_update,
                        exploration_initial_eps = 0.99, #self.epsilon,
                        exploration_final_eps = 0.1, #self.epsilon,
                        exploration_fraction = 0.2 #self.epsilon,
                        )
        elif self.model_id == 'A2C':
            self.model = A2C("MlpPolicy", 
                        env = self.env, 
                        verbose=1, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        gamma = self.gamma,
                        device=self.device, 
                        )


def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run experiments with different configurations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the experiment")
    parser.add_argument("--alg", type=str, default="PPO", choices=["PPO", "DQN","A2C"], help="Algorithm to use")
    parser.add_argument("--env", type=str, default="CartPole", help="Environment ID")
    parser.add_argument("--t_multip", type=float, default=1.0, help="Time step multiplier (delta_t)")
    #parser.add_argument("--no_t_step", type=float, default=10000, help="No. of tiem steps")
    parser.add_argument("--alph", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--path", type=str, default="./", help="Path to save results")
    parser.add_argument("--task_ID", type=str, default="01", help="Task_ID")
    args = parser.parse_args()



    exp = Experiment(model_id = args.alg,
                 env_id = args.env, 
                 delta_t = args.t_multip,
                 gamma = 0.95,
                 epsilon = 0.2,
                 learning_rate = args.alph,
                 #steps_per_update = 1,
                 policy_kwargs = dict(net_arch=[512,256,64]),
                 device = 'cpu',
                 seed = args.seed, #TODO: Please think about this?
                #  callback = RewardCallback(),
                 total_timesteps = 200_000,
                 #max_episode_steps = None, 
                 save_path = args.path,
                 batch_size = 16,
                 buffer_size = 500, 
                 target_update = 100, #TODO: Find out the default values of these parameters.
                 task_ID=args.task_ID,  # Pass task_ID here
                 )

    # Step 3: Run the experiment
    exp.run()

    # Step 4: Save the results
    exp.save_data()


if __name__ == "__main__":
    main()
