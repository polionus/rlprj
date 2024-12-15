import zipfile
import numpy as np
import matplotlib.pyplot as plt
import glob
import time

configs = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]: 
  for alg in ["PPO", "A2C"]:
    for env in ["CartPole", "Acrobot"]:
      for t in ["16.0", "32.0", "64.0", "128.0"]:
        for alph in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
          configs.append(f"Alg{alg}_env{env}_seed{seed}_tmultiplier{t}_alpha{alph}_RETURNS.npz")



count=1          
for name in glob.glob("../Big t/results13/*.npz"):
  
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
def sort_key(file_name):
    parts = file_name.split("_")
    alg = parts[0]  # e.g., "AlgPPO"
    env = parts[1]  # e.g., "envAcrobot"
    seed = parts[2]  # e.g., "seed10"
    tmult = parts[3]  # e.g., "tmultiplier0.125"
    alpha = parts[4]  # e.g., "alpha0.0001"
    return alg, env, tmult, alpha, seed

# Sort the list
sorted_files = sorted(configs, key=sort_key)


# Print the sorted list
# t125 = [x for x in sorted_files if "tmultiplier0.125" in x]
# t25 = [x for x in sorted_files if "tmultiplier0.25" in x]
# t5 = [x for x in sorted_files if "tmultiplier0.5" in x]

# print(len(t125))
# print(len(t25))
# print(len(t5))

for name in sorted_files:
  params = name.split("_")
  alg = params[0][3:]
  env = params[1][3:]
  seed = params[2][4:]
  t_multip = params[3][11:]
  aplph = params[4][5:]

  print(f"--seed {seed} --alg {alg} --env {env} --t_multip {t_multip} --alph {alph} --path ./Big_t/left --time 00:30:00")


