""" Soring a parameter set, based on steady states and the Jacobian. """

from typing import Dict, List, Tuple

import numpy as np
import subprocess
import math
import peakutils
import ast
import numpy.fft as fft
import matplotlib.pyplot as plt

from scipy.integrate import odeint

Vec3 = Tuple[float, float, float]

def score(par) -> float:

    """ Score a parametrisation, specified by a parameter dict"""
    new_pars = np.array(list(par.values()))
    true_params = np.array([
    104.02468968707345, 104.02468968707345, 104.02468968707345, 1.3515127830534523,1.3515127830534523,1.3515127830534523,36.81797811695671,36.81797811695671,36.81797811695671,3,3,3 # second set of odes
#    0, 0  # third set of odes
])
    num_timesteps = 100  # Number of time steps for simulation

    t = np.linspace(0, 100, num_timesteps)

    true_data = solve_ode(true_params, t)
    new_data = solve_ode(new_pars, t)
    
    distance = combined_distance(true_data, new_data)

    if math.isnan(distance):
        print(new_pars)

    return distance



def model(variables, t, params):

    m1, p1, m2, p2, m3, p3 = variables
    k1, k2, k3, g1, g2, g3,a1,a2,a3,n1,n2,n3 = params
    b1=b2=b3 = 43.43698380995929
    dm1=dm2=dm3 = 2.2491024418676977
    dp1=dp2=dp3 = 0.7747671591505249
    
    dm1dt = -dm1*m1 + (a1 / (1 + ((1/k1) * p2)**n1)) + g1
    dp1dt = (b1*m1) - (dp1*p1)
    dm2dt = -dm2*m2 + (a2 / (1 + ((1/k2) * p3)**n2)) + g2
    dp2dt = (b2*m2) - (dp2*p2)
    dm3dt = -dm3*m3 + (a3 / (1 + ((1/k3) * p1)**n3)) + g3
    dp3dt = (b3*m3)-(dp3*p3)
    
    return [dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt]

def solve_ode(params, t):
    initial_conditions = np.array([0.0, 2.0, 0.0, 1.0, 0.0, 3.0])
    solution = odeint(model, initial_conditions, t, args=(params,))
    return solution


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
        return cost
    return int(cost)

def combined_distance(observed_trajectory, simulated_trajectory):
    timepoints = int(len(observed_trajectory))
    third = int(timepoints / 3)
    observed = observed_trajectory[third:timepoints]
    simulated = simulated_trajectory[third:timepoints] #Discard the first third
    euclidean_distance = euclidean_distance_multiple_trajectories(observed, simulated)
    penalising_factor = np.abs(np.abs(costTwo(simulated_trajectory)) - 200)
    if costTwo(simulated_trajectory) >= 200:
        return euclidean_distance
    else:
        if penalising_factor < 1:
            penalising_factor = 1
        return euclidean_distance * penalising_factor