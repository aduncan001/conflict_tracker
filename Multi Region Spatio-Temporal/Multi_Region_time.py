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
        #if the runlength < p assign 1
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
            probs.append(np.mean(poisson.pmf(k=data[zone], mu= np.exp(gamma[:,zone]))))
        lik.append(np.prod(probs))
        expected_param[0][0]=[np.mean(np.exp(gamma), axis = 0)]
        expected_param[1][0]=[np.std(np.exp(gamma),axis = 0)]
        predicted_param[0][0]=[np.mean(np.exp(gamma), axis = 0)]
        predicted_param[1][0]=[np.std(np.exp(gamma),axis = 0)]
        if(t >1) :
            for s in range(1,t):
                if(R[s] < 10**(-3)*5):
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
                        #'Mean' : np.zeros(self.dim),
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
                    probs.append(np.mean(poisson.pmf(k=data[i], mu = np.exp(New_Gamma[:,i])))) 
                lik.append(np.prod(probs))
        print('Expected param:', expected_param)
        return lik, expected_param, predicted_param
