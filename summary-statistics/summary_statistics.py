def summarise(solution):
    """Calculate summary statistics
        requires solution to be the result of calling solve_ode[parameters]
        e.g. true_data
        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
    mean=np.mean(solution, axis=0)
    log_var=np.log(np.var(solution, axis=0))
    #The autocorrelation coefficient of each time series at lag 10 and lag 5
    #can be altered
    auto_cor=np.zeros(12)
    for s in range(6):
        auto_cor[2*s]=np.corrcoef(solution[10:,s],solution[:-10,s])[0][1]
        auto_cor[2*s+1]=np.corrcoef(solution[5:,s],solution[:-5,s])[0][1]
    cor_coef=[]
    for i in range(3): #cross correlation for mRNA only
        for j in range(i+1,3):
            cross_cor=np.corrcoef(solution[:,i*2],solution[:,j*2])[1,0]
            cor_coef.append(cross_cor)
    return(np.concatenate([mean, log_var, auto_cor, cor_coef]))
    