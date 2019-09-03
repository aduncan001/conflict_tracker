def online_changepoint_detection(data, hazard_func, observation_likelihood,p):
    #code taken initially taken from
    #https://github.com/hildensia/bayesian_changepoint_detection
    maxes = np.zeros(len(data) -p)
    params = np.zeros(shape = (2, len(data)-p-1))
    predict = np.zeros(shape = (2, len(data)-p-1))
    R = np.zeros((len(data) -p, len(data) -p))
    R[0, 0] = 1
    for m, x in enumerate(data):
        t = m-p-1 #shift everything to exclude the first points
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        #only save the run before you have enough data point to fit the time series 
        if(m < p+1):
            observation_likelihood.update_run(x)
            continue
        #step 3-4
        # Evaluate the predictive distribution for the new datum for all runlength
        predprobs, expected_params, predicted_params = observation_likelihood.pdf(x,R[0:t+1, t])
        
        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)))
        
        # Evaluate the growth probabilities 
        #step 5 
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        #step 6
        # Evaluate the probability that there was a changepoint 
        
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        # Renormalize the run length probabilities 
        #step 7
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        print(R[:,t+1])
        
        # Update the run.
        #step 8
        observation_likelihood.update_run(x)
        
        #save parameters for all runlength
        #step 9
        maxes[t] = R[:, t].argmax()
        params[:,t] = expected_params[:,int(maxes[t])]
        predict[:,t] = predicted_params[:,int(maxes[t])]
    #return estimated parameters and prediction (only for time dependency case)
    return R, params, predict
