import gymnasium as gym
from Helpers.callback import RewardCallback
from stable_baselines3 import PPO, DQN, A2C


class Experiment:
    def __init__(self, 
                 model_id:str,
                 env_id:str, 
                 delta_t = 1,
                 gamma = 1,
                 epsilon = 0.3,
                 learning_rate = 0.001,
                 #steps_per_update = 1,
                 policy_kwargs = dict(net_arch=[64,64,64]),
                 device = 'cpu',
                 seed = 0, #TODO: Please think about this?
                #  callback = RewardCallback(),
                 total_timesteps = 200_000,
                 #max_episode_steps = None, 
                 save_path = None,
                 batch_size = 16,
                 buffer_size = 500, 
                 target_update = 100, #TODO: Find out the default values of these parameters.
                 task_ID="01",  # Add task_ID parameter
                 ):
        
        self.returns = 0
        self.delta_t = delta_t 
        self.gamma = gamma
        # self.callback = callback
        self.save_path = save_path
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.learning_rate = learning_rate
        #self.steps_per_update = steps_per_update
        self.policy_kwargs = policy_kwargs
        self.device = device
        self.seed = seed

        self.buffer_size = buffer_size
        self.target_update = target_update

        self.model_id = model_id
        #self.max_episode_steps = max_episode_steps
        self.env_id = env_id
        self.total_timesteps = total_timesteps/self.delta_t
        self.task_ID = task_ID  

        # Initialize the callback with required parameters
        self.callback = RewardCallback(
            model_id=self.model_id,
            env_id=self.env_id,
            seed=self.seed,
            delta_t=self.delta_t,
            alpha=self.learning_rate,
            path=self.save_path,
            task_ID=self.task_ID, 
        )

        #make the environment
        self.make_env()

        #define the model
        self.define_model()

    def make_env(self):

        #First register the environment:
        if self.env_id == "CartPole":
            self.env_id = "CustomCartPole-v0"
            gym.register(
            id="CustomCartPole-v0",
            entry_point="Environments.custom_cartpole:CustomCartPoleEnv",
            reward_threshold = 500,
            max_episode_steps = 500/self.delta_t, 
            )
        elif self.env_id == "Acrobot":
            self.env_id = "Acrobot-v1"
            pass
            self.env_id ="CustomAcroBot-v1"
            gym.register(
            id="CustomAcroBot-v1",
            entry_point="Environments.custom_acrobot:CustomAcrobotEnv",
            reward_threshold = -100,
            max_episode_steps = 500/self.delta_t, 
             )
            #TODO: change to MC
        elif self.env_id == "MountainCar":
            self.env_id = "CustomMountainCar-v0" 
            gym.register(
            id="CustomMountainCar-v0",
            entry_point="Environments.custom_mountain_car:CustomMountainCarEnv",
            reward_threshold = 200,
            max_episode_steps = 200/self.delta_t, 
            )

        self.env = gym.make(self.env_id,max_episode_steps= 500)
        self.env.reset(seed = self.seed)

    def run(self):
        self.model.learn(total_timesteps =self.total_timesteps, callback = self.callback)

    def save_data(self):
        self.model.save(self.save_path) 
  
    

    #Please first run make env
    def define_model(self):
        
        if self.model_id == 'PPO':
            self.model = PPO("MlpPolicy", 
                        env=self.env, 
                        verbose=1, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        gamma = self.gamma,
                        device= self.device,
                        batch_size=self.batch_size,
                        )
        elif self.model_id == 'DQN':
            self.model = DQN("MlpPolicy", 
                        env = self.env, 
                        verbose=1, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        buffer_size=self.buffer_size,
                        gamma = self.gamma,
                        device=self.device, 
                        batch_size= self.batch_size,
                        target_update_interval = self.target_update,
                        exploration_initial_eps = 0.99, #self.epsilon,
                        exploration_final_eps = 0.1, #self.epsilon,
                        exploration_fraction = 0.2 #self.epsilon,
                        )
        elif self.model_id == 'A2C':
            self.model = A2C("MlpPolicy", 
                        env = self.env, 
                        verbose=1, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        gamma = self.gamma,
                        device=self.device, 
                        )


