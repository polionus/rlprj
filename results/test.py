import zipfile
import numpy as np
import matplotlib.pyplot as plt

# filename_returns = f"AlgDQN_envCartPole_seed10_tmultiplier_alpha_RETURNS.npy"
# filename_rewards = f"AlgDQN_envCartPole_seed10_tmultiplier_alpha_REWARDS.npy"



# np.save(filename_returns, np.array([1,2]))
# np.save(filename_rewards, np.array([3,4]))

# zip_filename="sosisAli.zip"
# with zipfile.ZipFile(zip_filename, 'w') as zipf:
#             zipf.write(filename_returns)
#             zipf.write(filename_rewards)


A = np.load("AlgDQN_envCartPole_seed0_tmultiplier1.0_alpha0.01_RETURNS.npy")

plt.plot(A)
plt.show()