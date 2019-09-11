#!/usr/bin/python3
from __future__ import division
import numpy as np
from scipy.stats import stats, poisson, nbinom, invwishart, cauchy
import pystan
from functools import partial
import math
import pandas as pd

data_africa = pd.read_csv("1900-01-01-2019-08-07-Eastern_Africa-Middle_Africa-Northern_Africa-Southern_Africa-Western_Africa.csv")
data_VAC = data_africa[data_africa['event_type'] == 'Violence against civilians']
VAC = data_VAC[['event_date','fatalities']]
VAC['timestamp'] =pd.to_datetime(data_VAC.event_date, infer_datetime_format= True)
VAC['month'] = pd.DatetimeIndex(VAC['event_date']).month
VAC['year'] = pd.DatetimeIndex(VAC['event_date']).year

VAC_togroup = VAC[['year', 'month']]
VAC_togroup['count'] = 1

filtered = VAC_togroup[VAC_togroup['year']>=2009]
filtered = filtered[filtered['year']<2017]

filtered = filtered.groupby(['year','month']).sum()

D = filtered['count'].tolist()


Single_Region_code = """
data {
    int N; //number of observations in block
    int y[N]; //number of data points
    //define the prior parameters
    real mean_a;
    real sigma_a;
    real mean_sigma;
    real sigma_sigma;
    real mean_mu;
    real sigma_mu;
    real mean_gamma1;
    real sigma_gamma1;
}

parameters {
    real a; //coefficients of AR(1)
    real mu; //constant term in AR
    real sigma; //variance of error term
    real gamma_prior;
    vector[N] err;
}

transformed parameters{
    vector[N] gamma;
    real new_gamma;
    gamma[1] = gamma_prior;
    if (N>1){ //if you have enogh datapoint define time series
    for (i in 2:(N)){
        gamma[i] = mu +  a*gamma[i-1]+ sigma*err[i-1];
        }
    }
    new_gamma = mu + a*gamma[N]+sigma*err[N]; //prediction
}

model {
    a ~ normal(mean_a, sigma_a);
    sigma ~ normal(mean_sigma,sigma_sigma);
    mu ~ normal(mean_mu, sigma_mu);
    gamma_prior ~ normal(mean_gamma1,sigma_gamma1);
    err ~ normal(0,1);
    y ~ poisson(exp(gamma));
}
"""

Single_Reg_time = pystan.model.StanModel(model_code=Single_Region_code, model_name ='Single_Region_code')

class Single_Region_time(): 
    #the function _init_ is the builder
    def __init__(self,mean_a,sigma_a,mean_sigma, sigma_sigma, mean_mu, sigma_mu, mean_gamma1, sigma_gamma1): 
        #initialise an attribute for the elements of the run
        self.r = []
        self.mean_a = mean_a
        self.sigma_a = sigma_a
        self.mean_sigma =mean_sigma
        self.sigma_sigma = sigma_sigma
        self.mean_mu = mean_mu
        self.sigma_mu = sigma_mu
        self.mean_gamma1 = mean_gamma1
        self.sigma_gamma1 = sigma_gamma1

       
    def update_run(self, x):
        #save the data collected up to this point
        #you save the most recent element in position 0
        self.r.insert(0,x)

    def pdf(self, data, R):
        t = len(self.r)+1
        lik = []
        expected_param = np.zeros(shape=(2,t))
        predict_param = np.zeros(shape=(2,t))
        gamma = np.random.normal(self.mean_gamma1,self.sigma_gamma1,1000)
        lik.append(np.mean(poisson.pmf(data, np.exp(gamma))))
        expected_param[:,0] = [np.mean(np.exp(gamma)), np.std(np.exp(gamma))]
        predict_param[:,0] = [np.mean(np.exp(gamma)), np.std(np.exp(gamma))]
        print("t=0  ", expected_param[:,0])
        print("predicted param ", predict_param[:,0])
         
        if(t>1):
            for i in range(1,t):
                if(R[i] < 10**(-4)*5):
                    lik.append(0)               
                    continue
                x_run = self.r[0:(i)]
                x_run.reverse()
                #generate the model with this data
                dat = {'N': i,
                    'y': np.array(x_run).T.tolist(), #re-organise the data 
                    'mean_a': self.mean_a,
                    'sigma_a': self.sigma_a,
                    'mean_sigma': self.mean_sigma,
                    'sigma_sigma': self.sigma_sigma,
                    'mean_mu': self.mean_mu,
                    'sigma_mu': self.sigma_mu,
                    'mean_gamma1': self.mean_gamma1,
                    'sigma_gamma1': self.sigma_gamma1,
                     }

                trace = Single_Reg_time.sampling(data=dat, iter=5000,warmup = 3000, init = '0', seed = 2341,chains=6, control=dict(adapt_delta=0.99,max_treedepth=15))
                la = trace.extract()  # return the chains
                new_gamma = (la['new_gamma'])
                gamma = trace.extract()['gamma']
                expected_param[:,i] = [np.mean(np.exp(gamma[:,-1])), np.std(np.exp(gamma[:,-1]))]
                predict_param[:,i] = [np.mean(np.exp(new_gamma)), np.std(np.exp(new_gamma))]
                lik.append(np.mean(poisson.pmf(data, mu = np.exp(new_gamma))))
             
        return lik, expected_param, predict_param

def online_changepoint_detection(data, hazard_func, observation_likelihood):

	#code originally borrowed from 
	#https://github.com/hildensia/bayesian_changepoint_detection.git

    maxes = np.zeros(len(data) + 1)
    params = np.zeros(shape = (2, len(data)))
    predict =  np.zeros(shape = (2, len(data)))
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    for t, x in enumerate(data):
        #Evaluate the predictive distribution for the new datum for all runlength
        #save predictions and posterior parameter.
        #step 3-4
        predprobs, expected_params, predict_params = observation_likelihood.pdf(x,R[0:t+1, t])
        
        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)),t = t+1)
        
        # Evaluate the growth probabilities 
        #step 5 
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        
        # Evaluate the probability that there was a changepoint and we reset the runlength
        #step 6
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        # Renormalize the runlength probabilities
        #step 7
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        print(R[0:t+2,t+1])
        # Update the parameter sets for each possible run length.
        #step 8

        observation_likelihood.update_run(x)

        maxes[t] = R[:, t].argmax()
        params[:,t] = expected_params[:,int(maxes[t])]
        predict[:,t] = predict_params[:,int(maxes[t])]
    
    return R, params, predict


def constant_hazard(lam,r):
    return 1/lam * np.ones(r.shape)

R_Single_Reg_time, rates, predict  = online_changepoint_detection(D, partial(constant_hazard, 250), Single_Region_time(0,.25,0,1,2.5,.2,5.5,.5))
np.savetxt('R_VAC_mod.txt', R_Single_Reg_time, delimiter=',')

np.savetxt('expected_params_VAC_mod.txt', rates, delimiter =',')

np.savetxt('predicted_params_VAC_mod.txt', predict, delimiter =',')
