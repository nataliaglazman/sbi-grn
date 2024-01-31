""" Soring a parameter set, based on steady states and the Jacobian. """

from typing import Dict, List, Tuple

import numpy as np


import matplotlib.pyplot as plt

from scipy.integrate import odeint

Vec3 = Tuple[float, float, float]

def score(par: Dict[str, float]) -> float:
    """ Score a parametrisation, specified by a parameter dict,
    evaluated at four levels of external signal.
    We want pure DC behaviour at S=1 and S=10'000,
    pure AC behaviour at S=100,
    and a coexistence of both at S=1'000. """
    new_pars = par.items()
 
    # Convert object to a list
    new_pars = list(new_pars)
 
    # Convert list to an array
    new_pars = np.array(new_pars)

    true_params = np.array([
    3.0, 0.0, 2.0, 0.0,  # first set of odes
    3.0, 0.0, 2.0, 0.0,  # second set of odes
    3.0, 0.0, 2.0, 0.0  # third set of odes
])
    epsilon = 5  # Error threshold
    num_samples = 10000  # Number of samples to draw
    num_timesteps = 1000  # Number of time steps for simulation

    t = np.linspace(0, 30, num_timesteps)

    true_data = solve_ode(true_params, t)
    new_data = solve_ode(new_pars, t)


    return euclidean_distance_multiple_trajectories(true_data, new_data)



def model(variables, t, params):

    m1, p1, m2, p2, m3, p3 = variables
    a1, b1, n1, gamma1, a2, b2, n2, gamma2, a3, b3, n3, gamma3 = params

    dm1dt = -m1 + (10 ** a1 / (1 + p2 ** n1)) + 10 ** gamma1
    dp1dt = -10 ** b1 * (p1 - m1)

    dm2dt = -m2 + (10 ** a2 / (1 + p3 ** n2)) + 10 ** gamma2
    dp2dt = -10 ** b2 * (p2 - m2)
    dm3dt = -m3 + (10 ** a3 / (1 + p1 ** n3)) + 10 ** gamma3
    dp3dt = -10 ** b3 * (p3 - m3)

    return [dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt]



def solve_ode(params, t):
    initial_conditions = np.array([0, 1, 0, 3, 0, 2])
    solution = odeint(model, initial_conditions, t, args=(params,))
    return solution



def euclidean_distance_multiple_trajectories(observed_trajectories, simulated_trajectories):

    mean_squared_difference = np.mean((observed_trajectories - simulated_trajectories)**2)
    return mean_squared_difference