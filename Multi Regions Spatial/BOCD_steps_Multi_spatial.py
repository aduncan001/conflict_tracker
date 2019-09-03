def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)
    params = np.zeros(shape = (2,len(data))).tolist()

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    for t, x in enumerate(data):
        #step 3
        # Evaluate the predictive distribution for the new datum for all runlength
        #Save mean and std of posterior parameters 
        predprobs, expected_params = observation_likelihood.pdf(x,R[0:t+1, t])
        
        # Evaluate the hazard function 
        H = hazard_func(np.array(range(t+1)))
       
        # Evaluate the growth probabilities 
        #step 4
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        
        # Evaluate the probability that there was a changepoint and we're
        # accumulating the mass back down at r = 0.
        #step 5
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        # Renormalize the run length probabilities 
        #step 6-7
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        
        # Update the parameter sets for each possible run length.
        #step 8
        observation_likelihood.update_run(x)
                
        #save only posterior parameters estimation for runlength with highest probability.
        maxes[t] = R[:, t].argmax()
        params[0][t] = expected_params[0][int(maxes[t])]
        params[1][t] =  expected_params[1][int(maxes[t])] 
    params = np.concatenate(params,axis = 0).reshape(2*len(data),len(data[0]))
    print(params.shape)
    return R, params
