#Calculate the number of time steps
from utils.utils import clock_to_time_steps

n_timesteps = clock_to_time_steps()



clock_settings = {
    "wall-clock-time": 10,
}


experiment_settings = {      
    "n_timesteps": clock_to_time_steps(clock_settings["wall-clock-time"]),
    "policy_kwargs": dict(net_arch=[64, 64, 64]),
    "seeds":100,
    "start_delta_exponent":-40,
    "end_delta_exponent":-10,
    "num_deltas":50,
    "training_delta":1e-70,
    "checker_delta":1e-10,
    "desc_color":"\033[92m",
    "reset_color":"\033[0m",
    "path_to_google_drive":"path/to/desktop/google/drive/RL Project (Fall 2024)/Results/CartPole_Single",
}


