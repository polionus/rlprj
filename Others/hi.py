import sys
import gymnasium as gym 

env =  gym.make("CartPole-v1")
env.reset(seed= 1)

print(sys.argv[0])