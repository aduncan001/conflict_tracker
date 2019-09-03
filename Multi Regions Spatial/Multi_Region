class Multi_Region(): 
    #the function _init_ is the builder
    def __init__(self,n,mu_mean,mu_sigma,sigma_expected_val,sigma_std,omega_k): 
        #initialise an attribute for the elements of the run
        self.r = []
        self.dim = n
        self.Mu_Mean = mu_mean
        self.Mu_Sigma = mu_sigma
        self.sigma_mean = sigma_expected_val
        self.sigma_sd = sigma_std
        self.Omega_k = omega_k

    def update_run(self, x):
        #save the data collected up to this point
        #you save the most recent element in position 0
        self.r.insert(0,x)
            
    def pdf(self, data, R):
        #sample from posterior distribution given data (for all possible run length)
        #this is a vector which includes the probabilities for ALL runlength. 
        
        Mu_Mean = self.Mu_Mean
        Mu_Sigma= self.Mu_Sigma
        sigma_mean = self.sigma_mean 
        sigma_sd = self.sigma_sd 
        Omega_k= self.Omega_k 

        t = len(self.r)+1
        probs = []
        lik = []
        expected_param = np.zeros(shape =(2,t)).tolist()
        #changepoint happens: initialize the rate as a random normally distributed number
        #build the matrix Sigma as done in the Stan model 
        Mu = np.random.multivariate_normal(Mu_Mean, Mu_Sigma, 1000)       
        sigma  = abs(np.random.normal(sigma_mean,sigma_sd, size =(100,self.dim)))
        Omega = pm.LKJCorr.dist(eta = Omega_k, n = self.dim).random(size = 100)
	      #Omega is a vector: you need to rebuild the matrix
        Gamma =[]
        for elem in range(100):
            New = np.diag(np.zeros(self.dim))
            #transform Omega into upper triangular matrix
            New[np.triu_indices(self.dim,k=1)] = Omega[elem]
            #create Correlation matrix
            New = New+New.T+ np.identity(self.dim)
            #create Sigma
            Sigma = np.diag(sigma[elem,:]) @ New @np.diag(sigma[elem,:]) 
            Gamma.append(np.random.multivariate_normal(Mu[elem], Sigma ,100))
        Gamma = np.concatenate(Gamma, axis= 0)
        expected_param[0][0]=[np.mean(Gamma**2, axis = 0)]
        expected_param[1][0]=[np.std(Gamma**2,axis = 0)]
        Rate= Gamma**2
        #evaluate likelihood in each region
        for i in range(self.dim):
            probs.append(np.mean((poisson.pmf(k=data[i], mu = Rate[:,i]))))
        #evaluate predictive probability of new datum
        lik.append(np.prod(probs))
        probs = []
        if(t>1):
            #evaluate for only the useful information
            for i in range(1,t):
                if(R[i] < 10^(-3)):
                    lik.append(0)               
                    continue
                #save in x_run the elements in the current run
                #this will be the data for our model 
                x_run = self.r[0:(i)] 
                #from the most recent x_(t-1) to x_(t-i)
                #generate the model with this data
                dat = {'K': self.dim,
                       'N': i,
                       'y': np.array(x_run).T.tolist(), #re-organise the data so you have on each line the same region
                       'Mu_Mean' : Mu_Mean,
                       'Mu_Sigma' : Mu_Sigma,
                       'sigma_mean': sigma_mean,
                       'sigma_sd': sigma_sd,
                       'Omega_k': Omega_k,
                      }
                
                    #now sample from posterior distribution
                trace = Multi_Reg.sampling(data=dat, iter=2000, chains=8 , control=dict(adapt_delta=0.95),warmup = 300)
                la = trace.extract()  # return the chains
                Gamma = la['Gamma']
                expected_param[0][i]=[ np.mean(trace['Gamma']**2,axis = 0)]
                expected_param[1][i] = [np.std(trace['Gamma']**2, axis = 0)] 
                for i in range(self.dim):
                    probs.append(np.mean(poisson.pmf(k=data[i], mu = Gamma[:,i]**2))) 
                lik.append(np.prod(probs))
                probs = []
        return lik, expected_param
