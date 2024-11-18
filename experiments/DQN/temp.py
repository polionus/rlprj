import gymnasium as gym
import inspect

env_id = "Acrobot-v1"
env = gym.make(env_id)
env_file_path = inspect.getfile(type(env.unwrapped))
print(f"The environment file is located at: {env_file_path}")


import os

os.system(f"code {env_file_path}")  # Open with VS Code
# or
# os.system(f"notepad {env_file_path}")  # Open with Notepad

