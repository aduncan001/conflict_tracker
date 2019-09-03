def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)
    params = np.zeros(shape = (2,len(data))).tolist()
    predicted = np.zeros(shape = (2,len(data))).tolist()
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    for t, x in enumerate(data):
        #step 3-4
        # Evaluate the predictive distribution for the new datum for all runlength
        predprobs, expected_params, predicted_params = observation_likelihood.pdf(x,R[0:t+1, t])
        
        # Evaluate the hazard function
        H = hazard_func(np.array(range(t+1)))
       
        # Evaluate the growth probabilities 
        #step 5
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        
        # Evaluate the probability that there was a changepoint and we reset the runlength.
        #step 6
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        # Renormalize the run length probabilities 
        #step 7
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        # Update the parameter sets for each possible run length.
        #step 8
        observation_likelihood.update_run(x)
                
        #save parameters correspoding to the runlength with the highest probability
        #step 9
        maxes[t] = R[:, t].argmax()
        params[0][t] = expected_params[0][int(maxes[t])]
        params[1][t] =  expected_params[1][int(maxes[t])]
        predicted[0][t] = predicted_params[0][int(maxes[t])]
        predicted[1][t] =  predicted_params[1][int(maxes[t])]

    params = np.concatenate(params,axis = 0).reshape(2*len(data),len(data[0]))
    predicted = np.concatenate(predicted,axis = 0).reshape(2*len(data),len(data[0]))
    return R, params, predicted
