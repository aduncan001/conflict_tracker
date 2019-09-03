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
        #not enough information on data
        gamma = np.random.normal(1,1,1000)
        lik.append(np.mean(poisson.pmf(data, np.exp(gamma))))
        expected_param[:,0]=[np.mean(np.exp(gamma)),np.std(np.exp(gamma))]
        predict_param[:,0]=[np.mean(np.exp(gamma)),np.std(np.exp(gamma))]
        print("t=0  ", expected_param[:,0])
        if(t>1):
            for i in range(1,t):
                if(R[i] < 10**(-3)*5):
                    lik.append(0)               
                    continue
                x_run = self.r[0:(i)]
                x_run.reverse()
                #generate the model with this data
                dat = {'N': i,
                    'y': np.array(x_run).T.tolist()[0], #organise the data 
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
                #lik.append(np.mean(poisson.pmf(data, mu = np.exp(gamma[:,-1]))))
        return lik, expected_param, predict_param
