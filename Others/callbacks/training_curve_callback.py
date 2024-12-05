from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

#a call back that accumulates rewards
class RewardCallback(BaseCallback):
    '''A Callback receives and keeps the important signals that a train function produces,
    and helps us with logging it.'''

    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.cum_reward = 0.
        #self.cum_done = 0.

    def _on_step(self) -> bool:
        reward = np.mean(self.locals["rewards"])  # average over all environments
        #done = np.mean(self.locals["dones"])

        self.cum_reward = self.cum_reward + reward
        #self.cum_done = self.cum_done + done

        self.logger.record('custom/cum_reward', self.cum_reward)
        #self.logger.record('custom/cum_done', self.cum_done)
        return True
