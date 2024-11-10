import gymnasium as gym


import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


n_timsteps = 500_000

policy_kwargs = dict(net_arch=[64, 64, 64])

# Parallel environments
#vec_env = make_vec_env("LunarLander-v2", n_envs=4)
#vec_env = make_vec_env("CartPole-v1", n_envs=4)
#env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0,
               #enable_wind=False, wind_power=15.0, turbulence_power=1.5,dt = 10)
env = gym.make("CartPole-v1", dt_multip = 1e-10)


# model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1)
# model.learn(total_timesteps=n_timsteps)
# model.save("ppo_cartpole")
# exit()



model = PPO.load("ppo_cartpole")

env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0,
              enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode = "human", dt = 10)


# obs, _ = env.reset()
# for step in range(500):
#     env.render()
#     action, _states = model.predict(obs)
#     obs,reward, terminated, truncated, _ = env.step(action)
#     print(reward)

# env.close()
# exit()

seeds = 100
#deltas = [5e-3,1e-2,3e-2,5e-2, 1e-1,2e-1, 3e-1, 4e-1, 5e-1, 7e-1, 1e0]

num_deltas = 30
start_delta = 1e-15
end_delta = 1

# increment = (end_delta-start_delta)/num_deltas
increment = 1e-1
deltas = [start_delta * increment**i for i in range(num_deltas)]
#deltas = []

returns_mean = []
for dt in tqdm(deltas):
    print(dt)
    #env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0,
               #enable_wind=False, wind_power=15.0, turbulence_power=1.5,dt = dt)
    env = gym.make("CartPole-v1", dt_multip = dt)
    
    returns = []

    for seed in tqdm(range(seeds)):
        _return = 0
        obs, _ = env.reset()


        for step in range(500):
            #env.render()
            action, _states = model.predict(obs, deterministic = True)
            obs,reward, terminated, truncated, _ = env.step(action)
            #obs,reward, terminated, trunctated, _ = env.step(env.action_space.sample())

            # if terminated or truncated:
            #     reward = 0


            _return += reward
        
        
        returns.append(_return)
    mean = np.mean(returns)
    returns_mean.append(mean)    


plt.plot(returns_mean)
plt.show()

#print(np.mean(returns))
env.close()    
# plt.plot(returns)
# plt.show()
