import numpy as np 
import math
import peakutils
import numpy.fft as fft
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib import cm  
from scipy.integrate import odeint 
import scipy.signal as signal 
	
'''
The deterministic model of biological repressilator 
''' 
class Repressilator: 
	
	def __init__(self, parameter_values, params, initial_conditions, dt = 0.001, mode = 0): 
		self.nParams = len(params)   
		self.params = params #model parameters
		self.parameter_values = parameter_values #allowed parameter ranges  
		self.y0 = initial_conditions 	
		self.dt = dt
		self.T = 48 #hours
		self.N = int(self.T/self.dt) #how many samples, 1000 per hour, 480000
		self.ts = np.linspace(0, self.T, self.N) 
		self.amp = 300 #[nM] 		
		self.per = self.T/8 	
		self.sample_rate 		= 0.0033333333 #Hz 
		self.samples_per_hour 	= (1/self.dt)	#1000	
		self.jump 				= int(self.samples_per_hour/(self.sample_rate*3600)) #number of samples 	
		self.ideal = self.amp*(np.sin(math.pi*(self.ts)/self.per - math.pi/2) + 1) 
		#number of samples for FFT		
		self.nS = self.N/self.jump 
		self.dF = self.sample_rate/self.nS  
		self.idealF = self.getFrequencies(self.ideal) 		 	
		thresholdOne = -(self.nS/2)*100 #10nM -+ from ideal signal harmonics       
		thresholdTwo = 200  
		self.minAmp = 200
		self.maxAmp = 400 
		self.mode = mode    			
		self.modes = [self.eval]       
		self.threshold = thresholdOne  
		self.omega = 1 #nm^-1 
		if self.mode == 1:
			self.threshold = thresholdTwo
	
	#gets sumed difderence of arrayData
	@staticmethod 	
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
	@staticmethod 
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
	
	def getFrequencies(self, y):
		#fft sample rate: 1 sample per 5 minutes
		y = y[0::self.jump]  #Get y at every jump
		res = abs(fft.rfft(y))  #Real FT
		#normalize the amplitudes 
		res = res/math.ceil(self.nS/2) #nS is N/T, just normalise with this factor
		return res

	def costOne(self, Y): #X
		p1 = Y[:,1]   
		fftData = self.getFrequencies(p1)     
		
		diff = fftData - self.idealF         
		cost = -np.dot(diff, diff) 		
		return cost,	
		
	def costTwo(self, Y, getAmplitude = False): #Yes
		p1 = Y[:,1]  #Get the first column
		fftData = self.getFrequencies(p1)    #Get frequencies of FFT of the first column  
		fftData = np.array(fftData) 
		#find peaks using very low threshold and minimum distance
		indexes = peakutils.indexes(fftData, thres=0.02/max(fftData), min_dist=1)  #Just find peaks
		#in case of no oscillations return 0 
		if len(indexes) == 0:     
			return 0,  
		#if amplitude is greater than 400nM
		amp = np.max(fftData[indexes])
		if amp > self.maxAmp: #If bigger than 400, then cost is 0, not viable
			return 0, 
		fitSamples = fftData[indexes]  			
		std = self.getSTD(indexes, fftData, 1)  #get sd of peaks at a window of 1 (previous peak)
		diff = self.getDif(indexes, fftData)  #Get differences in peaks
		cost = std + diff #Sum them
		#print(cost)   
		if getAmplitude:
			return cost, amp
		return cost, 
		
	def isViableFitness(self, fit):
		return fit >= self.threshold #If above 200, then yes it produces oscillations
		
	def isViable(self, point): 
		fitness = self.eval(point, getAmplitude=True)  
		if self.mode == 0: #Mode is determined beforehand...
			return self.isViableFitness(fitness[0]) 
			
		fit = fitness[0] 
		amp = 0
		if fit > 0:
			amp = fitness[1] 
		return self.isViableFitness(fit) and amp >= self.minAmp and amp <= self.maxAmp   
		
	#evaluates a candidate  
	def eval(self, candidate, getAmplitude = False): 
		Y = np.array(self.simulate(candidate)) #store the simulation and return costs
		if self.mode == 0:
			return self.costOne(Y)  
		else:
			return self.costTwo(Y, getAmplitude)      
	
	#simulates a candidate
	def simulate(self, candidate): 
		return odeint(self.repressilatorModelOde, self.y0, self.ts, args=(candidate,))   		
		
	def plotModel(self, subject, mode="ode", show=True):     		
		if mode == "ode":
			ts = np.linspace(0, self.T, self.N)
			Y = self.simulate(subject) 			
		else:
			#ssa simulation
			ts,Y = self.represilatorStochastic(subject)
			
		Y = np.array(Y) 
		
		p0 = Y[:,0] 
		p1 = Y[:,1]  
		p2 = Y[:,2]   
		p3 = Y[:,3]     
		p4 = Y[:,4] 
		p5 = Y[:,5] 
		
		lines = plt.plot(ts, p1, ts, p3, ts, p5)  
		plt.setp(lines[0], linewidth=1.5, c="#15A357")
		plt.setp(lines[1], linewidth=1.5, c="#0E74C8")
		plt.setp(lines[2], linewidth=1.5, c="#A62B21")     		 
		plt.ylabel('Concentration [$nM$]')   
		plt.xlabel(r"Time [$h$]")   
		plt.legend(('X', 'Y', 'Z'), loc='upper right')      		
		if show: 	
			plt.show() 
			 				

	def getTotalVolume(self):
		vol = 1.0
		for param in self.params:		
			vol = vol*(self.parameter_values[param]["max"] - self.parameter_values[param]["min"])
		return vol 

	def repressilatorModelOde(self, Y, t, can):  
		alpha = can[0]
		alpha0 = can[1]
		n = can[2]
		beta = can[3]
		deltaRNA = can[4] 
		deltaP = can[5]
		kd = can[6] 
		mx = Y.item(0)
		my = Y.item(2) 
		mz = Y.item(4)
		x = Y.item(1)
		y = Y.item(3) 
		z = Y.item(5) 
		
		#in case of math range error
		try:
			dmx = -deltaRNA*mx + alpha/(1 + math.pow(z/kd, n)) + alpha0 
			dmy = -deltaRNA*my + alpha/(1 + math.pow(x/kd, n)) + alpha0  
			dmz = -deltaRNA*mz + alpha/(1 + math.pow(y/kd, n)) + alpha0  
		except (OverflowError, ValueError):
			dmx = -deltaRNA*mx + alpha + alpha0
			dmy = -deltaRNA*my + alpha + alpha0
			dmz = -deltaRNA*mz + alpha + alpha0 
			
		dpx = beta*mx - deltaP*x
		dpy = beta*my - deltaP*y 
		dpz = beta*mz - deltaP*z
		
		return np.array([dmx, dpx, dmy, dpy, dmz, dpz])
	
	
	def getPerAmp(self, subject, mode="ode", indx=0): 
		if mode == "ode":
			ts = np.linspace(0, self.T, self.N) 
			Y = self.simulate(subject)    				
		else:
			ts,Y = self.represilatorStochastic(subject) 
		ts = np.array(ts) 
		Y = np.array(Y) 
		sig = Y[:, indx]
		indx_max, properties = signal.find_peaks(sig, prominence = (np.max(sig) - np.min(sig))/4, distance = len(ts)/100)      
		indx_min, properties = signal.find_peaks(sig*-1, prominence = (np.max(sig) - np.min(sig))/4, distance = len(ts)/100)     

		amps = [] 
		pers = []   
		for i in range(min(len(indx_max), len(indx_min))):
			amps.append((sig[indx_max[i]] - sig[indx_min[i]])/2) 			
			if i + 1 < len(indx_max):
				pers.append(ts[indx_max[i + 1]] - ts[indx_max[i]])
			if i + 1 < len(indx_min):
				pers.append(ts[indx_min[i + 1]] - ts[indx_min[i]])
		
		if len(amps) > 0 and len(pers) > 0:
			amps = np.array(amps)   	
			pers = np.array(pers)  
			
			#print(amps)
			amp = np.mean(amps)	
			#print(pers) 
			per = np.mean(pers) 
		else:
			amp = 0
			per = 0  
		
		print("amp" + str(amp)) 
		print("per" + str(per))   	
		
		return per, amp 
			
	def represilatorStochastic(self, can):
		omega = self.omega 
		y_conc = np.array(self.y0*omega).astype(int) 
		Y_total = []
		Y_total.append(y_conc)
		t = 0 
		T = []   
		T.append(t)
		
		#get kinetic rates 
		alpha = can[0]
		alpha0 = can[1]
		n = can[2]
		beta = can[3]
		deltaRNA = can[4] 
		deltaP = can[5]
		kd = can[6] 
		
		N = np.zeros((6,15)) #6 species, 15 reactions
		N[0,0] = -1
		N[0,1] = 1 
		N[0,2] = 1 

		N[1,9] = 1
		N[1,10] = -1
		
		N[2,3] = -1 
		N[2,4] = 1
		N[2,5] = 1

		N[3,11] = 1
		N[3,12] = -1 
		
		N[4,6] = -1
		N[4,7] = 1
		N[4,8] = 1

		N[5,13] = 1
		N[5,14] = -1 
		
		
		while t < self.T:
			#choose two random numbers 
			r = np.random.uniform(size=2)
			r1 = r[0] 
			r2 = r[1] 					
			
			#get propensities
			a = np.zeros(15)
			a[0] = deltaRNA*y_conc[0] 
			a[1] = alpha/(1.0 + math.pow(y_conc[5]/(kd*omega), n))*omega 
			a[2] = alpha0*omega  
			
			a[3] = deltaRNA*y_conc[2] 
			a[4] = alpha/(1.0 + math.pow(y_conc[1]/(kd*omega), n))*omega
			a[5] = alpha0*omega   
			
			a[6] = deltaRNA*y_conc[4] 
			a[7] = alpha/(1.0 + math.pow(y_conc[3]/(kd*omega), n))*omega
			a[8] = alpha0*omega          
			
			a[9] = beta*y_conc[0]
			a[10] = deltaP*y_conc[1]

			a[11] = beta*y_conc[2]
			a[12] = deltaP*y_conc[3]

			a[13] = beta*y_conc[4] 
			a[14] = deltaP*y_conc[5]
			

			asum = np.cumsum(a)
			a0 = np.sum(a)  
			#get tau
			tau = (1.0/a0)*np.log(1.0/r1)     
		
			#select reaction 
			reaction_number = np.argwhere(asum > r2*a0)[0,0] #get first element			
		
			#update concentrations
			y_conc = y_conc + N[:,reaction_number]   	
			Y_total.append(y_conc) 
			#update time
			t = t + tau  
			T.append(t)
		T = np.array(T) 
		Y_total = np.array(Y_total)   
		return T, Y_total 
	
	

