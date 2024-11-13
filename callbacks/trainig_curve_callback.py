from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


#a call back that accumulates rewards

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Log rewards
        if "episode" in self.locals:
            self.rewards.append(self.locals["episode"]["r"])
        return True 