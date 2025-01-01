import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from experiments import Experiment




def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run experiments with different configurations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the experiment")
    parser.add_argument("--alg", type=str, default="PPO", choices=["PPO", "DQN","A2C"], help="Algorithm to use")
    parser.add_argument("--env", type=str, default="CartPole", help="Environment ID")
    parser.add_argument("--t_multip", type=float, default=1.0, help="Time step multiplier (delta_t)")
    #parser.add_argument("--no_t_step", type=float, default=10000, help="No. of tiem steps")
    parser.add_argument("--alph", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--path", type=str, default="./", help="Path to save results")
    parser.add_argument("--task_ID", type=str, default="01", help="Task_ID")
    parser.add_argument("--time", type=str, default="01:00:00", help="Dummy argument!")
    args = parser.parse_args()

   




    # Step 2: Initialize from Experiment class
    exp = Experiment(model_id = args.alg,
                 env_id = args.env, 
                 delta_t = args.t_multip,
                 gamma = 1,
                 epsilon = 0.2,
                 learning_rate = args.alph,
                 #steps_per_update = 1,
                 policy_kwargs = dict(net_arch=[512,256,64]),
                 device = 'cpu',
                 seed = args.seed, 
                #  callback = RewardCallback(),
                 total_timesteps = 200_000,
                 #max_episode_steps = None, 
                 save_path = args.path,
                 batch_size = 16,
                 buffer_size = 500, 
                 target_update = 100, 
                 task_ID=args.task_ID,  
                 )

    # Step 3: Run the experiment
    exp.run()

    # Step 4: Save the results
    exp.save_data()


if __name__ == "__main__":
    
    main()
