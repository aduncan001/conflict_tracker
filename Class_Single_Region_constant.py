class Single_Region_constant():
    #the function _init_ is the builder
    def __init__(self, mean_prior,sd_prior):
        #initialise an attribute for the elements of the run
        self.r = []
        self.mean_prior_mean = mean_prior
        self.mean_prior_sd = sd_prior
    def update_run(self, x):
        #save the data collected up to this point
        #you save the most recent element in position 0
        self.r.insert(0,x)
        
        
    def pdf(self, data, R):
        #sample from posterior distribution given data (for all possible run length)
        #this is a vector which includes the probabilities for ALL runlength. 
        t = len(self.r)+1
        mean = self.mean_prior_mean
        std = self.mean_prior_sd
        expected_param = np.zeros(shape =(2,t))
        lik = []
        #changepoint happens: initialize the rate as a random normally distributed number
        param = np.random.normal(np.random.normal(mean,std,1000), np.ones(1000), size = (100,1000))
        expected_param[:,0]=[(np.mean(param)), np.std(param)]
        lik.append(np.mean(poisson.pmf(k=data, mu= param**2)))
        if(t>1):
            for i in range(1,t):
                if(R[i]<0.001):
                    lik.append(0) #do not sample if the probability of r_t is too small
                    continue
                #save in x_run the elements in the current run
                	#this will be the data for our model 
                x_run = self.r[0:(i)] #from the most recent x_(t-1) to x_(t-i)
                #generate the model with this data
                
                with pm.Model() as Poisson_Norm:
                    #initialise prior distribution on mean mu
                    mu = pm.Normal('mu', mu = mean, sd = std)
                    #initialise parameter gamma 
                    gamma = pm.Normal('gamma', mu = mu, sd = 1)
                    #now initialise data distributed as poisson gamma^2
                    x_obs = pm.Poisson('x_obs', mu = gamma**2, observed = x_run)
                        
                    #now sample from posterior distribution
                    trace = pm.sample(1500, tune=200, target_accept=0.95)
                    #mean and std of RATES
                    expected_param[:,i]=[(np.mean(trace['gamma']**2)), np.std(trace['gamma']**2)]
                #evaluate predictive probability
                lik.append(np.mean(poisson.pmf(k=data, mu = trace['gamma']**2)))
        return lik, expected_param
