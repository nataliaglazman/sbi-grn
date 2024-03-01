from scipy.signal import find_peaks

def summarise(solution):
    data=solution[:,0:6:2]
    """Calculate summary statistics
        requires solution to be the result of calling solve_ode[parameters]
        e.g. true_data
        Returns
        -------
        a 1D array of summary statistics containing the mean of solution, the log variance of solution, the autocorrelation at lag 10, cross correlation of mRNA and protein species separately, number of peaks
        """
    mean=np.mean(data, axis=0) #mean of mRNA species
    var=np.var(data, axis=0) #var
    auto_cor=np.corrcoef(data[10:,0],solution[:-10,0])[1,0] #autocorrelation at lag 10 for m1
    cor=np.corrcoef(data,data, rowvar=False)[:3,1:4] #cross correlation
    cor_coef=np.diag(cor)
    #returning cross correlation coefficients of mRNA to mRNA species, protein to protein species
    truncated_trajectory=data[third:timepoints,:] 
    peaks,_=zip(*(find_peaks(truncated_trajectory[:,i], height=0) for i in range(3))) 
    peak_count=np.array([len(p) for p in peaks]) #number of peaks 
    return(np.concatenate([mean, var, auto_cor.flatten(), cor_coef, peak_count.flatten()]))