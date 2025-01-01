from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

class RewardCallback(BaseCallback):

    def __init__(self, model_id, env_id, seed, delta_t, alpha, path, task_ID, verbose=0):
        super().__init__(verbose)
        self.model_id = model_id
        self.env_id = env_id
        self.seed = seed
        self.delta_t = delta_t
        self.alpha = alpha
        self.path = path
        self.task_ID = task_ID  # For CC
        self.eps_returns_list = []
        self.eps_return = 0
        self.steps_rewards = []
        self.done_status = []
    
    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        self.steps_rewards.append(self.locals['rewards'])
        self.done_status.append(self.locals['dones'])
        self.eps_return += self.locals['rewards']
        if self.locals['dones'] == True:
            self.eps_returns_list.append(self.eps_return[0])
            self.eps_return =0
       
        return True

    def _on_training_end(self) -> None:
        filename_returns = f"Alg{self.model_id}_env{self.env_id}_seed{self.seed}_tmultiplier{self.delta_t}_alpha{self.alpha}_RETURNS"
       
        full_returns_path = os.path.join(self.path, filename_returns)
        

        np.savez_compressed(full_returns_path, returns = self.eps_returns_list, rewards =np.column_stack((self.steps_rewards, self.done_status)) )

        return True
     
