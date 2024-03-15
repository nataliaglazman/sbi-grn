
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
from math import prod
from scipy.stats import uniform
import scipy
import time
import numpy.fft as fft
import scipy.signal as signal
import peakutils


def model_6p(variables,t, params,fixed_values):
    m1, p1, m2, p2, m3, p3 = variables
    k1,k2,k3,a1,a2,a3=params
    g1,g2,g3,n1,n2,n3,b1,b2,b3,dm1,dm2,dm3,dp1,dp2,dp3=fixed_values
    dm1dt = -dm1*m1 + (a1 / (1 + ((1/k1) * p2)**n1)) + g1
    dp1dt = (b1*m1) - (dp1*p1)
    dm2dt = -dm2*m2 + (a2 / (1 + ((1/k2) * p3)**n2)) + g2
    dp2dt = (b2*m2) - (dp2*p2)
    dm3dt = -dm3*m3 + (a3 / (1 + ((1/k3) * p1)**n3)) + g3
    dp3dt = (b3*m3)-(dp3*p3)
    return np.array([dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt]).flatten()

def solve_ode(params,fixed_values):
    initial_conditions = np.array([0, 2, 0, 1, 0, 3])
    solution = odeint(model_6p, initial_conditions, t=np.linspace(0,100,100), args=(params,fixed_values)) # The initial value point should be the first element of this sequence
    return solution

rng = np.random.default_rng()

def multivariate_gaussian_kernel(cov_params,scale=1):
    gk=scale*rng.multivariate_normal(mean=np.zeros(6),cov=cov_params) #using symmetrical kernels for now
    return(gk)


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

priors_a=np.array([[10.**-2, 250],[10.**-2, 250],[10.**-2, 250],[20.,40.,],[20.,40.,],[20.,40.]])

#format prior low,high,low,high
def generate_init_cov(priors,scale):
    """generate initial covariance matrix for the mcmc sampler"""
    init_cov=np.zeros((6,6))
    p1=priors[:,0]
    p2=priors[:,1]
    dif=np.subtract(p2,p1)/np.sqrt(12)
    np.fill_diagonal(init_cov, dif*scale)
    return(init_cov)

# Only what we need
def getDif(indexes, arrayData):	
    arrLen = len(indexes)
    sum = 0
    for i, ind in enumerate(indexes):
        if i == arrLen - 1:
            break
        sum += arrayData[ind] - arrayData[indexes[i + 1]]
    #add last peak - same as substracting it from zero
    sum += arrayData[indexes[-1:]]
    return sum

#gets standard deviation
def getSTD(indexes, arrayData, window):
    numPeaks = len(indexes)
    arrLen = len(arrayData)
    sum = 0
    for ind in indexes:
        minInd = max(0, ind - window)
        maxInd = min(arrLen, ind + window)
        sum += np.std(arrayData[minInd:maxInd])
    sum = sum/numPeaks 	#The 1/P factor
    return sum

def getFrequencies(y):
    res = abs(fft.rfft(y))  #Real FT
    #normalize the amplitudes
    #res = res/math.ceil(1/2) #Normalise with a factor of 1/2
    return res

def costTwo(Y, getAmplitude = False): #Yes
    p1 = Y[:,1]  #Get the first column
    fftData = getFrequencies(p1)    #Get frequencies of FFT of the first column
    fftData = np.array(fftData)
    #find peaks using very low threshold and minimum distance
    indexes = peakutils.indexes(fftData, thres=0.02/max(fftData), min_dist=1)  #Just find peaks
    #in case of no oscillations return 0
    if len(indexes) == 0:
        return 0
    #if amplitude is greater than 400nM
    #global amp
    #amp = np.max(fftData[indexes])
    #if amp > 400: #If bigger than 400, then cost is 0, not viable
      #  return 0,
    fitSamples = fftData[indexes]  			
    std = getSTD(indexes, fftData, 1)  #get sd of peaks at a window of 1 (previous peak)
    diff = getDif(indexes, fftData)  #Get differences in peaks
    cost = std + diff #Sum them
    #print(cost)
    if getAmplitude:
        return cost, amp
    return int(cost)

def combined_distance(observed, simulated):
    euclidean_distance = euclidean_distance_multiple_trajectories(observed, simulated)
    penalising_factor = np.abs(np.abs(costTwo(simulated)) - 200)
    if costTwo(simulated) >= 200:
        return euclidean_distance
    else:
        if penalising_factor < 1:
            penalising_factor = 1
        return euclidean_distance * penalising_factor
    

def run_chain1(args):
    rndint=np.random.randint(low=0,high=1e7)
    timeseed=time.time_ns()%2**16
    np.random.seed(rndint+timeseed)
    true_params, num_iterations,fixed_values,kernel_size,init_kernel_size = args
    true_data = solve_ode(true_params,fixed_values)
    init_cov=generate_init_cov(priors_a,init_kernel_size)
    accepted_params = np.zeros((num_iterations, len(true_params)))
    distance_arr= np.zeros(num_iterations+1)
    count_arr = []
    count=0
    cov=init_cov
    # Initialized to random parameters
    sampled_params = np.concatenate([np.random.uniform(0.,250.,size=3),np.random.uniform(20.,40.,size=3)])
     #np.random: high to 3 low=-3
    accepted_params[0] = sampled_params
    sampled_data = solve_ode(sampled_params,fixed_values)
    threshold=700 #initial threshold
    distance=combined_distance(true_data, sampled_data)
    distance_arr[0]=distance
    set_distance = distance
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
            cov=kernel_size*np.cov(accepted_params[i-10000:i,:],rowvar=False) #get covariance matrix of accepted parameters, after 1000 burnins
            if acceptance_rate > 0.02:
                dis_seg=distance_arr[i-10000:i]
                threshold=np.min(dis_seg)+0.95*(np.median(np.unique(dis_seg))-np.min(dis_seg)) #new threshold median of previous round
                #decreasingf threshold every 10000 iterations
                #arbitrary at the moment
                #only look at the last 10000 samples for now
            print(f"{i}th iterations done, previously acceptance rate = {acceptance_rate}, threshold is set to {threshold}")
        # Using Gaussian kernel to sample for next model parameters
        perturbation=multivariate_gaussian_kernel(cov)
        new_sampled_params=sampled_params+perturbation
        if not (np.all(new_sampled_params > priors_a[:,0]) and np.all(new_sampled_params < priors_a[:,1])):
            while not (np.all(new_sampled_params > priors_a[:,0]) and np.all(new_sampled_params < priors_a[:,1])):
                perturbation = multivariate_gaussian_kernel(cov) #perturb again
                new_sampled_params=sampled_params+perturbation
        new_sampled_data = solve_ode(new_sampled_params,fixed_values)
        distance = combined_distance(true_data, new_sampled_data)
        if distance < threshold:
            count += 1
            #accepted_data[i] = sampled_data
            #prior_prob = prior_prob_new  # Keep track of prior probs for efficiency
            sampled_params = new_sampled_params
            set_distance = distance
            # while  max(new_sampled_params)>prior[1] or min(new_sampled_params)<prior[0]:
            #     perturbation = multivariate_gaussian_kernel(cov) #perturb again
            #     new_sampled_params=sampled_params+perturbation
    # Generate synthetic data using samples
        distance_arr[i]=set_distance
        accepted_params[i] = sampled_params
    return accepted_params, count_arr, distance_arr