import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
from math import prod
from scipy.stats import uniform



# Define the system of ODEs
def model(variables, t, params):
    m1, p1, m2, p2, m3, p3 = variables
    k1,k2,k3=params
    #for now we are setting a, g, n, b dm, dp to be constant
    a1 = a2 = a3 = 24.78485282457379
    g1 = g2 = g3 = 0.024884149937163258
    n1 = n2 = n3 = 5.
    b1 = b2 = b3 = 33.82307682700831
    dm1 = dm2 = dm3 =1.143402097500176
    dp1 = dp2 = dp3 = 0.7833664565550977
    dm1dt = -dm1*m1 + (a1 / (1 + ((1/k1) * p2)**n1)) + g1
    dp1dt = (b1*m1) - (dp1*p1)
    dm2dt = -dm2*m2 + (a2 / (1 + ((1/k2) * p3)**n2)) + g2
    dp2dt = (b2*m2) - (dp2*p2)
    dm3dt = -dm3*m3 + (a3 / (1 + ((1/k3) * p1)**n3)) + g3
    dp3dt = (b3*m3)-(dp3*p3)
    return np.array([dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt]).flatten()




def solve_ode(params):
    initial_conditions = np.array([0, 2, 0, 1, 0, 3])
    solution = odeint(model, initial_conditions, t=np.linspace(0,100,100), args=(params,)) # The initial value point should be the first element of this sequence
    return solution


true_params=[200.,200.,200.]
true_data = solve_ode(true_params)



rng = np.random.default_rng()
def multivariate_gaussian_kernel(cov_params,scale=1):
    gk=scale*rng.multivariate_normal(mean=np.zeros(len(true_params)),cov=cov_params) #using symmetrical kernels for now
    return(gk)
 #need to take care in future in not getting out of bounds

def euclidean_distance_multiple_trajectories(observed_trajectories, simulated_trajectories):
    num_trajectories = len(observed_trajectories)
    total_distance = 0.0

    for i in range(num_trajectories):
        observed_data = observed_trajectories[i]
        simulated_data = simulated_trajectories[i]

        # Calculate the Euclidean distance between observed and simulated data
        euclidean_distance = np.linalg.norm(observed_data - simulated_data)

        # Accumulate the distances
        total_distance += euclidean_distance

    # Average the distances over all trajectories
    average_distance = total_distance / num_trajectories

    return average_distance



def run_chain(args):
    prior=[0.,250.]
    true_params, num_iterations, chain_index = args
    true_data = solve_ode(true_params)
    init_cov=30.*np.identity(len(true_params)) ## initial covariance matrix should be large
    accepted_params = np.zeros((num_iterations, len(true_params)))
    distance_arr= np.zeros(num_iterations+1)
    count_arr = []
    count=0
    cov=init_cov
    # Initialized to random parameters
    sampled_params = uniform.rvs(0.,250.,size=3) #from -3 to 3, six params
     #np.random: high to 3 low=-3
    accepted_params[0] = sampled_params
    sampled_data = solve_ode(sampled_params)
    threshold=1000 #initial threshold
    distance=euclidean_distance_multiple_trajectories(true_data, sampled_data)
    set_distance = distance
    #round=-1
    for i in range(1, num_iterations):
        if i % 10000 == 0:
            #round+=1
            #threshold=threshold_arr[round]
            count_arr.append(count)
            acceptance_rate=count/10000 #acceptance rate for the past 5000 iterations
            count=0 #reset count
            #if the acceptance rate really low <1% 
            #keep current threshold
            #else: decrease threshold to median of previously accepted parameters
            cov=0.66*np.cov(accepted_params[i-10000:i,:],rowvar=False) #get covariance matrix of accepted parameters, after 1000 burnins
            if acceptance_rate > 0.01:
                dis_seg=distance_arr[i-10000:i]
                threshold=np.median(np.unique(dis_seg)) #new threshold median of previous round
                #decreasingf threshold every 10000 iterations
                #arbitrary at the moment
                #only look at the last 10000 samples for now
            print(f"{i}th iterations done, previously acceptance rate = {acceptance_rate}, threshold is set to {threshold}")
        # Using Gaussian kernel to sample for next model parameters

        perturbation=multivariate_gaussian_kernel(cov)
        new_sampled_params=sampled_params+perturbation
        if max(new_sampled_params)>prior[1] or min(new_sampled_params)<prior[0]:
            while  max(new_sampled_params)>prior[1] or min(new_sampled_params)<prior[0]:
                perturbation = multivariate_gaussian_kernel(cov) #perturb again
                new_sampled_params=sampled_params+perturbation 
    # Generate synthetic data using samples
        new_sampled_data = solve_ode(new_sampled_params)
        distance = euclidean_distance_multiple_trajectories(true_data, new_sampled_data)
        if distance < threshold:
            count += 1
            #accepted_data[i] = sampled_data
            #prior_prob = prior_prob_new  # Keep track of prior probs for efficiency
            sampled_params = new_sampled_params 
            set_distance = distance
        distance_arr[i]=set_distance
        accepted_params[i] = sampled_params
    return accepted_params, count_arr, distance_arr


def run_chain1(args):
    true_params, num_iterations = args
    true_data = solve_ode(true_params)
    init_cov=50.*np.identity(len(true_params)) ## initial covariance matrix should be large
    accepted_params = np.zeros((num_iterations, len(true_params)))
    distance_arr= np.zeros(num_iterations+1)
    count_arr = []
    count=0
    cov=init_cov
    prior=[0.,250.]
    # Initialized to random parameters
    sampled_params = uniform.rvs(0.,250.,size=3) #from -3 to 3, six params
     #np.random: high to 3 low=-3
    accepted_params[0] = sampled_params
    sampled_data = solve_ode(sampled_params)
    threshold=600 #initial threshold
    distance=euclidean_distance_multiple_trajectories(true_data, sampled_data)
    set_distance = distance
    #round=-1
    for i in range(1, num_iterations):
        if i % 10000 == 0:
            #round+=1
            #threshold=threshold_arr[round]
            count_arr.append(count)
            acceptance_rate=count/10000 #acceptance rate for the past 5000 iterations
            count=0 #reset count
            #if the acceptance rate really low <1% 
            #keep current threshold
            #else: decrease threshold to median of previously accepted parameters
            cov=0.6*np.cov(accepted_params[i-10000:i,:],rowvar=False) #get covariance matrix of accepted parameters, after 1000 burnins
            if acceptance_rate > 0.05:
                dis_seg=distance_arr[i-10000:i]
                threshold=np.median(np.unique(dis_seg)) #new threshold median of previous round
                #decreasingf threshold every 10000 iterations
                #arbitrary at the moment
                #only look at the last 10000 samples for now
            print(f"{i}th iterations done, previously acceptance rate = {acceptance_rate}, threshold is set to {threshold}")
        # Using Gaussian kernel to sample for next model parameters
        perturbation=multivariate_gaussian_kernel(cov)
        new_sampled_params=sampled_params+perturbation
        if max(new_sampled_params)>prior[1] or min(new_sampled_params)<prior[0]:
            while  max(new_sampled_params)>prior[1] or min(new_sampled_params)<prior[0]:
                perturbation = multivariate_gaussian_kernel(cov) #perturb again
                new_sampled_params=sampled_params+perturbation 
    # Generate synthetic data using samples
        new_sampled_data = solve_ode(new_sampled_params)
        distance = euclidean_distance_multiple_trajectories(true_data, new_sampled_data)
        if distance < threshold:
            count += 1
            #accepted_data[i] = sampled_data
            #prior_prob = prior_prob_new  # Keep track of prior probs for efficiency
            sampled_params = new_sampled_params 
            set_distance = distance
        distance_arr[i]=set_distance
        accepted_params[i] = sampled_params
    return accepted_params, count_arr, distance_arr