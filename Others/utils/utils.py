from math import ceil


def clock_to_time_steps(wall_clock_time: float, delta_t: float) ->  int:
    '''
    - Inputs: 
       -  Wall clock time: The actual physical time passed in the environment
       -  Delta t: The time discretization width. 

    - Output:
        - Time steps: The number of time steps needed to train the agent for
   
    '''
    return ceil(wall_clock_time/delta_t)
