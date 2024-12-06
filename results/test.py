import zipfile
import numpy as np
import matplotlib.pyplot as plt
import glob
import time

configs = []
for seed in [0, 1, 2, 3, 4]: 
  for alg in ["PPO", "A2C"]:
    for env in ["CartPole", "Acrobot"]:
      for t in ["0.125", "0.25", "0.5", "1.0", "2.0", "4.0", "8.0"]:
        for alph in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
          configs.append(f"Alg{alg}_env{env}_seed{seed}_tmultiplier{t}_alpha{alph}_RETURNS.npz")

count=1          
for name in glob.glob("./results/results13/*.npz"):
  
    n = name.split('\\')[-1]
    if n in configs:
        ind = configs.index(n)
        configs.pop(ind)

    # A = np.load(name)
    # plt.plot(A['returns'])
    # plt.figure(f"Figure{count}")
    # plt.title(name)
    # plt.show()
    # count +=1
    # time.sleep(1)
    # plt.close()
configs.sort()
for conf in configs:
   print(conf)