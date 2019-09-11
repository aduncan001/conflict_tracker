#!/usr/bin/python3
from __future__ import division
import numpy as np
from scipy.stats import stats, poisson, nbinom, wishart
import pystan
from functools import partial
import math
import pandas as pd


data_libya = pd.read_csv("2010-01-01-2011-12-31-Libya.csv")
data_ASP = data_libya
ASP = data_ASP[['event_date', 'admin2']]
ASP['timestamp'] =pd.to_datetime(data_ASP.event_date, infer_datetime_format= True)
ASP['month'] = pd.DatetimeIndex(ASP['event_date']).month
ASP['year'] = pd.DatetimeIndex(ASP['event_date']).year

#each row represents an event
ASP['count'] = 1
l1=ASP['admin2'].unique().tolist()
l2=ASP['year'].unique().tolist()
l3=ASP['month'].unique().tolist()
l2.reverse()
l3.reverse()

import itertools
new_df =  pd.DataFrame(list(itertools.product(l1,l2,l3))).rename(columns={0:'admin2',1:'year',2:'month'})
new_df=pd.merge(new_df,ASP,on=['admin2','year','month'],how='left')

counts = new_df.groupby(['admin2','year','month']).sum()
reg = len(l1)

obs = []
D= []
for year in l2:
    for month in l3:
        for zone in l1:
            obs.append(int(counts['count'][zone][year][month]))
        D.append(obs)
        obs = []

Multi_Region_code_time = """
data {
    int K; //number of regions
    int N; //number of points
    int y[K,N]; //observations
    matrix[K,K] Id; // identity matrix
    vector[K] Zeros; //vector of Zeros
    real  mean_A;
    real  sigma_A;
    real mean_Sigma;
    real sigma_Sigma;
    vector[K] Mean_mu; 
    matrix[K,K] Sigma_mu;
    vector[K] Mean_gamma1;
    matrix[K,K] Sigma_gamma1;
}
parameters {
    matrix[K,K]  A; // coefficients
    vector[K] Mu;
    matrix[K,K] Sigma;
    matrix[K, N] Err;
    vector[K] Gamma_Prior;
}
transformed parameters {
    matrix[K,N] Gamma;
    vector[K] New_Gamma;
    Gamma[,1] = Gamma_Prior;
    if(N>1){ //if you have enogh points fit time series
        for (i in 2:N) {
            Gamma[,i] = Mu+A*Gamma[,i-1]+ Sigma*Err[,i-1];
        }
    }
    New_Gamma = Mu + A*Gamma[,N]+Sigma*Err[,N];
}
model {
    Mu ~ multi_normal(Mean_mu, Sigma_mu);
    to_vector(A) ~ normal (mean_A,sigma_A); //priors on parameters
    to_vector(Sigma) ~ normal(mean_Sigma, sigma_Sigma);
    Gamma_Prior ~ multi_normal( Mean_gamma1, Sigma_gamma1);
    for(time in 1:N){   Err[,time] ~ multi_normal(Zeros, Id);
        y[,time] ~ poisson(exp(Gamma[,time]));}
}
"""
 
Multi_Reg_time = pystan.model.StanModel(model_code=Multi_Region_code_time, model_name ='Multi_reg_time')

class Multi_Region_time(): #class for likelihood of data
    #the function _init_ is the builder
    def __init__(self, n, mean_A, sigma_A, mean_Sigma, sigma_Sigma, Mean_mu, Sigma_mu, Mean_gamma1, Sigma_gamma1): 
        #initialise an attribute for the elements of the run
        self.r = []
        self.dim = n
        self.mean_A = mean_A
        self.sigma_A = sigma_A
        self.mean_Sigma = mean_Sigma
        self.sigma_Sigma = sigma_Sigma
        self.Mean_mu = Mean_mu
        self.Sigma_mu = Sigma_mu
        self.Mean_gamma1 = Mean_gamma1
        self.Sigma_gamma1 = Sigma_gamma1
    def update_run(self, x):
        #save the data collected up to this point
        #you save the most recent element in position 0
        self.r.insert(0,x)
       
    def pdf(self, data, R):
        #this is a vector which includes the probabilities for ALL runlength. 
        #if the runlength < p assign 0
        reg = self.dim
        Id = np.identity(reg)
        Zero = np.zeros(reg)
         #initiliase prior on mean 
         #initialise prior on covariance
        t = len(self.r)+1
        probs = []
        lik = []
        expected_param = np.zeros(shape =(2,t)).tolist()
        predicted_param = np.zeros(shape =(2,t)).tolist()
        gamma = np.random.multivariate_normal(self.Mean_gamma1, self.Sigma_gamma1, 1000)
        for zone in range(0,reg):
            probs.append(math.log(np.mean(poisson.pmf(k=data[zone], mu= np.exp(gamma[:,zone])))))
        lik.append(math.e**np.sum(probs))            
        expected_param[0][0]=[np.mean(np.exp(gamma), axis = 0)]
        expected_param[1][0]=[np.std(np.exp(gamma),axis = 0)]
        predicted_param[0][0]=[np.mean(np.exp(gamma), axis = 0)]
        predicted_param[1][0]=[np.std(np.exp(gamma),axis = 0)]
        if(t >1) :
            for s in range(1,t):
                if(R[s] < 10**(-4)*5):
                    lik.append(0)               
                    continue
                x_run = self.r[0:(s)]
                #reverse in order to preserve time arrival
                x_run.reverse()
                #from the most recent x_(t-1) to x_(t-i)
                #generate the model with this data
                dat = {'K': self.dim,
                        'N': s,
                        'y': np.array(x_run).T.tolist(), #re-organise the data so you have on each line the same region
                        'Id' : Id,
                        'Zeros' : np.zeros(self.dim),
                        'mean_A':self.mean_A,
                        'sigma_A': self.sigma_A,
                        'mean_Sigma': self.mean_Sigma,
                        'sigma_Sigma' : self.sigma_Sigma,
                        'Mean_mu': self.Mean_mu,
                        'Sigma_mu': self.Sigma_mu,
                        'Mean_gamma1': self.Mean_gamma1,
                        'Sigma_gamma1': self.Sigma_gamma1                     
                       }
                #now sample from posterior distribution
                trace = Multi_Reg_time.sampling(data=dat,warmup = 3000, iter=5000,init = '0', seed = 2356, chains=5, control=dict(adapt_delta=0.95, max_treedepth =15))
                la = trace.extract()  # return the chains
                New_Gamma = la['New_Gamma']
                Gamma = la['Gamma']
                print(Gamma.shape)
                #select only the estimation of the last gamma
                expected_param[0][s]=[np.mean(np.exp(Gamma),axis=0)[:,-1]]
                expected_param[1][s]=[np.std(np.exp(Gamma),axis=0)[:,-1]]
                predicted_param[0][s]=[np.mean(np.exp(New_Gamma),axis=0)]
                predicted_param[1][s]=[np.std(np.exp(New_Gamma),axis=0)]
                probs =[]
                for i in range(reg):
                    probs.append(math.log(np.mean(poisson.pmf(k=data[i], mu = np.exp(New_Gamma[:,i])))))
                lik.append(math.e**np.sum(probs))

        return lik, expected_param, predicted_param



def online_changepoint_detection(data, hazard_func, observation_likelihood):

    #code originally borrowed from 
    #https://github.com/hildensia/bayesian_changepoint_detection.git

    maxes = np.zeros(len(data) + 1)
    params = np.zeros(shape = (2,len(data))).tolist()
    predicted = np.zeros(shape = (2,len(data))).tolist()
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    for t, x in enumerate(data):
        #Evaluate the predictive distribution for the new datum for all runlength
        #save predictions and posterior parameter.
        predprobs, expected_params, predicted_params = observation_likelihood.pdf(x,R[0:t+1, t])
        
        # Evaluate the hazard function
        H = hazard_func(np.array(range(t+1)))
       
        # Evaluate the growth probabilities
        #step 5 
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        
        # Evaluate the probability that there was a changepoint and we reset the runlength
        #step 6
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        # Renormalize the runlength probabilities
        #step 7
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        # Update the parameter sets for each possible run length.
        #step 8
        observation_likelihood.update_run(x)
        print(R[0:t+2,t+1])
       
        maxes[t] = R[:, t].argmax()
        params[0][t] = expected_params[0][int(maxes[t])]
        params[1][t] =  expected_params[1][int(maxes[t])]
        predicted[0][t] = predicted_params[0][int(maxes[t])]
        predicted[1][t] =  predicted_params[1][int(maxes[t])]
    params = np.concatenate(params,axis = 0).reshape(2*len(data),len(data[0]))
    predicted = np.concatenate(predicted,axis = 0).reshape(2*len(data),len(data[0]))
    return R, params, predicted


def constant_hazard(lam,r):
    return 1/lam * np.ones(r.shape)

R_Multi_Reg_time, expected, predicted  = online_changepoint_detection(D, partial(constant_hazard, 250), Multi_Region_time(reg,0,0.1,0,0.1,np.zeros(reg), np.identity(reg), np.zeros(reg), np.identity(reg)*0.4))

np.savetxt('ASP_mod.txt', R_Multi_Reg_time, delimiter=',')

np.savetxt('params_mod.txt', expected, delimiter = ',')

np.savetxt('predicted_mod.txt', predicted, delimiter = ',')

