#!/usr/bin/env python
# coding: utf-8

# # Rejection Approximate Bayesian Computation (REJ-ABC) for (3) parameter inference of a repressilator

# First we define the distance function, in this case Euclidean distance - it's great

# In[1]:


import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np 
import peakutils
import numpy.fft as fft
import matplotlib.pyplot as plt


#Distance function
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

def costTwo(Y): #Yes
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
    # fitSamples = fftData[indexes]  			
    std = getSTD(indexes, fftData, 1)  #get sd of peaks at a window of 1 (previous peak)
    diff = getDif(indexes, fftData)  #Get differences in peaks
    cost = std + diff 
    return int(cost)

def combined_distance(observed_trajectory, simulated_trajectory):
    euclidean_distance = euclidean_distance_multiple_trajectories(observed_trajectory, simulated_trajectory)
    penalising_factor = np.abs(np.abs(costTwo(simulated_trajectory)) - 200)
    if costTwo(simulated_trajectory) >= 200:
        return euclidean_distance
    else:
        if penalising_factor < 1:
            penalising_factor = 1
        return euclidean_distance * penalising_factor


# Secondly we specify a smoothing kernel as a function of $\epsilon$ as an alternative to a regular distance threshold cutoff

# In[2]:


#Visualise the smoothing kernel wrt epsilon
       #We have a 0.135 chance of accepting above the threshold!
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
epsilon = 150
x = np.linspace(-epsilon, epsilon, 1000)
normalising_factor = 1/norm.pdf(0, loc=0, scale = epsilon)
ax.plot(x, norm.pdf(x, loc=0, scale=epsilon/2)*normalising_factor/2,

       'r-', lw=5, alpha=0.6, label='norm pdf')


# In[3]:


epsilon = 150 #Results independent of epsilon
norm.pdf(-epsilon, loc=0, scale=epsilon/2)*normalising_factor/2


# # Try for three parameters (k1, k2, k3) (use a bigger epsilon $\epsilon = 150$)

# In[5]:


def model(variables, t, params):
    m1, p1, m2, p2, m3, p3 = variables
    k1, k2 =params
    k3 = 52.24130352947285
    a1 = 30.52123007653054
    a2 = 39.231588311273846
    a3 = 34.86007101736975
    g1 = 0.1902661176643755
    g2 = 0.28113643328037285
    g3 = 0.8181176651249633
    n1 = 4.758094367378883
    n2 = 1.985417765709296
    n3 = 4.910007465597671
    b1 = 24.68491191280538
    b2 = 29.42387320898578
    b3 = 41.04934603816582
    dm1 = 1.1977430229572492
    dm2 = 1.5302375124759988
    dm3 = 1.5608364378206137
    dp1 = 0.7747339528650133
    dp2 = 0.7511393265314563
    dp3 = 0.7528339378453786
    dm1dt = -dm1*m1 + (a1 / (1 + ((1/k1) * p2)**n1)) + g1
    dp1dt = (b1*m1) - (dp1*p1)
    dm2dt = -dm2*m2 + (a2 / (1 + ((1/k2) * p3)**n2)) + g2
    dp2dt = (b2*m2) - (dp2*p2)
    dm3dt = -dm3*m3 + (a3 / (1 + ((1/k3) * p1)**n3)) + g3
    dp3dt = (b3*m3)-(dp3*p3)
    return np.array([dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt])

true_params = np.array([
    38.94801652652866, 193.4015439096185  # first set of odes
])

def solve_ode(params):
    initial_conditions = np.array([0, 2, 0, 1, 0, 3])
    solution = odeint(model, initial_conditions, t=np.linspace(0,100,100), args=(params,)) # The initial value point should be the first element of this sequence
    return solution

num_timesteps = 100  # Number of time steps for simulation
t = np.linspace(0, 100, num_timesteps) #Range of time of simulation
true_data = solve_ode(true_params) #True trajectories


# In[25]:


# ABC rejection
from scipy.stats import norm
def abc_rejection(true_params, epsilon, num_samples):
    accepted_params = []
    accepted_data = []  # Added to store simulated data for accepted parameters
   
    for s in range(num_samples):
        # Define prior and sample
        sampled_params = np.random.uniform(low=0.01, high=250, size=len(true_params)) #Wide priors
       
        # Generate synthetic data using samples
        sampled_data = solve_ode(sampled_params)
        
        distance = combined_distance(true_data, sampled_data)
        normalising_factor = 1/norm.pdf(0, loc=0, scale = epsilon)
        smooth_threshold = norm.pdf(distance, loc=0, scale=epsilon/2)*normalising_factor/2
        rdm_tmp = np.random.uniform(low=0, high=1, size=1)
        if rdm_tmp < smooth_threshold: #If the random number is below the threshold, we accept it
            accepted_params.append(sampled_params)
            accepted_data.append(sampled_data)
        
        if s % 1000 == 0:
            print(f"Simulated {s} samples")
   
    return np.array(accepted_params), np.array(accepted_data)



# In[24]:


import pandas as pd
epsilon = 50 # Error threshold used in kernel, higher than 2 params
num_samples = 150000  # Number of samples to draw

#Algorithm
accepted_parameters_2p, accepted_data_2p = abc_rejection(true_params, epsilon, num_samples)

### Save csv...
column_name = ['k1','k2']
rejabc_2p = pd.DataFrame(data=accepted_parameters_2p, columns=column_name)
rejabc_2p.to_csv('rejabc_2p.csv')


# # Now do 3p

# In[26]:


def model(variables, t, params):
    m1, p1, m2, p2, m3, p3 = variables
    k1, k2, k3 =params
    a1 = 30.52123007653054
    a2 = 39.231588311273846
    a3 = 34.86007101736975
    g1 = 0.1902661176643755
    g2 = 0.28113643328037285
    g3 = 0.8181176651249633
    n1 = 4.758094367378883
    n2 = 1.985417765709296
    n3 = 4.910007465597671
    b1 = 24.68491191280538
    b2 = 29.42387320898578
    b3 = 41.04934603816582
    dm1 = 1.1977430229572492
    dm2 = 1.5302375124759988
    dm3 = 1.5608364378206137
    dp1 = 0.7747339528650133
    dp2 = 0.7511393265314563
    dp3 = 0.7528339378453786
    dm1dt = -dm1*m1 + (a1 / (1 + ((1/k1) * p2)**n1)) + g1
    dp1dt = (b1*m1) - (dp1*p1)
    dm2dt = -dm2*m2 + (a2 / (1 + ((1/k2) * p3)**n2)) + g2
    dp2dt = (b2*m2) - (dp2*p2)
    dm3dt = -dm3*m3 + (a3 / (1 + ((1/k3) * p1)**n3)) + g3
    dp3dt = (b3*m3)-(dp3*p3)
    return np.array([dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt])

true_params = np.array([
    38.94801652652866, 193.4015439096185, 52.24130352947285  # first set of odes
])

def solve_ode(params):
    initial_conditions = np.array([0, 2, 0, 1, 0, 3])
    solution = odeint(model, initial_conditions, t=np.linspace(0,100,100), args=(params,)) # The initial value point should be the first element of this sequence
    return solution

num_timesteps = 100  # Number of time steps for simulation
t = np.linspace(0, 100, num_timesteps) #Range of time of simulation
true_data = solve_ode(true_params) #True trajectories


# In[29]:


epsilon = 50 # Error threshold used in kernel, higher than 2 params
num_samples = 150000  # Number of samples to draw

#Algorithm
accepted_parameters_3p, accepted_data_3p = abc_rejection(true_params, epsilon, num_samples)

### Save csv...
column_name = ['k1','k2','k3']
rejabc_3p = pd.DataFrame(data=accepted_parameters_3p, columns=column_name)
rejabc_3p.to_csv('rejabc_3p.csv')

