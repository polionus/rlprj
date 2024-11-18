import numpy as np
import matplotlib.pyplot as plt

#Let's plot only one:

#different

for i in range(60):
    arr = np.load("run{}.npy".format(i))

    plt.plot(arr)
    plt.show()
    