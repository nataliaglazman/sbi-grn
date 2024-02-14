""" Soring a parameter set, based on steady states and the Jacobian. """

from typing import Dict, List, Tuple

import numpy as np
import subprocess
import math

import ast
import matplotlib.pyplot as plt

from scipy.integrate import odeint

Vec3 = Tuple[float, float, float]

def score(par) -> float:

    """ Score a parametrisation, specified by a parameter dict"""
    new_pars = np.array(list(par.values()))
    true_params = np.array([
    0, 0,  # first set of odes
    0, 3,
      3, 3  # third set of odes
])
    num_timesteps = 100  # Number of time steps for simulation

    t = np.linspace(0, 100, num_timesteps)

    true_data = solve_ode(true_params, t)
    new_data = solve_ode(new_pars, t)
    
    distance = euclidean_distance_multiple_trajectories(true_data, new_data)

    if math.isnan(distance):
        print(new_pars)

    return distance



def model(variables, t, params):

    m1, p1, m2, p2, m3, p3 = variables
    k1, k2, k3, a1, a2, a3 = params

    dm1dt = -m1 + (10 ** a1 / (1 + (10 ** k1 * p2)**2)) + 1
    dp1dt = -10 ** 0 * (p1 - m1)    
    dm2dt = -m2 + (10 ** a2 / (1 + (10 ** k2 * p3)**2)) + 1
    dp2dt = -10 ** 0 * (p2 - m2)    
    dm3dt = -m3 + (10 ** a3 / (1 + (10 ** k3 * p1)**2)) + 1
    dp3dt = -10 ** 0 * (p3 - m3)    
    
    return [dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt]

def solve_ode(params, t):
    initial_conditions = np.array([0.0, 2.0, 0.0, 1.0, 0.0, 3.0])
    solution = odeint(model, initial_conditions, t, args=(params,))
    return solution

#from julia import Julia
#from julia import Main

# Start Julia
#jl = Julia(compiled_modules=False)

# Load and execute the Julia script
#jl.include("julia_solver.jl")

# Example usage
#params_array = [0]  # Example array of parameter sets
#output_folder = "output_trajectories"  # Specify the output folder

# Call the Julia function from Python
#Main.solve_and_save_trajectories(params_array, output_folder)



def euclidean_distance_multiple_trajectories(truth, simulation):

    num_trajectories = len(truth)
    total_distance = 0.0

    for i in range(num_trajectories):
        observed_data = truth[i]
        simulated_data = simulation[i]

        # Calculate the Euclidean distance between observed and simulated data
        euclidean_distance = np.linalg.norm(observed_data - simulated_data)

        # Accumulate the distances
        total_distance += euclidean_distance

    # Average the distances over all trajectories
    average_distance = total_distance / num_trajectories

    return average_distance

