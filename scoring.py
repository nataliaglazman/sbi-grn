""" Soring a parameter set, based on steady states and the Jacobian. """

from typing import Dict, List, Tuple

import numpy as np
import subprocess

import ast
import matplotlib.pyplot as plt

from scipy.integrate import odeint

Vec3 = Tuple[float, float, float]

def score(par) -> float:

    """ Score a parametrisation, specified by a parameter dict"""
    new_pars = np.array(list(par.values()))

    true_params = np.array([
    3.0, 2.0,  # first set of odes
    3.0, 2.0, # second set of odes
    3.0, 2.0  # third set of odes
])
    epsilon = 5  # Error threshold
    num_samples = 10000  # Number of samples to draw
    num_timesteps = 10  # Number of time steps for simulation

    t = np.linspace(0, 100, num_timesteps)

    true_data = solve_ode(true_params, t)
    new_data = solve_ode(new_pars, t)
    
    return euclidean_distance_multiple_trajectories(true_data, new_data)



def model(variables, t, params):

    m1, p1, m2, p2, m3, p3 = variables
    b1,k1,b2,k2,b3,k3 = params

    dm1dt = -m1 + (10 ** 3 / (1 + (10 ** k1 * p2)**2)) + 10 ** 0
    dp1dt = -10 ** b1 * (p1 - m1)    
    dm2dt = -m2 + (10 ** 3 / (1 + (10 ** k2 * p3)**2)) + 10 ** 0
    dp2dt = -10 ** b2 * (p2 - m2)    
    dm3dt = -m3 + (10 ** 3 / (1 + (10 ** k3 * p1)**2)) + 10 ** 0
    dp3dt = -10 ** b3 * (p3 - m3)    
    return [dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt]

    return [dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt]



def solve_ode(params, t):
    initial_conditions = np.array([0.0, 2.0, 0.0, 1.0, 0.0, 3.0])
    solution = odeint(model, initial_conditions, t, args=(params,))
    return solution

def solve_ode_julia(parameters, num_timesteps):
    # Run Julia
    julia_script_path = "/project/home23/ng319/Desktop/sbi-grn/solve_ode.jl_fewer_params.txt"
    command = ["/project/home23/ng319/.juliaup/bin/julia", julia_script_path] + [str(param) for param in parameters] + [str(num_timesteps)]    # Use subprocess.run
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    solution = result.stdout.decode("utf-8")
    solution_list = ast.literal_eval(solution) 

    return solution_list


def euclidean_distance_multiple_trajectories(truth, simulation):

    truth = np.array(truth)
    simulation = np.array(simulation)    
    if truth.shape != simulation.shape:
        raise ValueError("Input arrays must have the same shape")    
    squared_differences = np.square(truth - simulation)
    mean_squared_difference = np.mean(squared_differences)  

    return mean_squared_difference
