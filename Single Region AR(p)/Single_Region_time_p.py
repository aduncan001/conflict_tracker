class Single_Region_time_p():
    #the function _init_ is the builder
    def __init__(self,p, mean_a, sigma_a, mean_sigma, sigma_sigma, mean_mu, sigma_mu, mean_gamma1, sigma_gamma1):
        #initialise an attribute for the elements of the run
        self.r = []
        self.p = p
        self.mean_a = mean_a
        self.sigma_a = sigma_a
        self.mean_sigma = mean_sigma
        self.sigma_sigma = sigma_sigma
        self.mean_mu = mean_mu
        self.sigma_mu = sigma_mu
        self.mean_gamma1 = mean_gamma1
        self.sigma_gamma1 = sigma_gamma1

    def update_run(self, x):
        #save the data collected up to this point
        #you save the most recent element in position 0
        self.r.insert(0,x)
        print(self.r)
    def pdf(self, data, R):
        p = self.p
        t = len(self.r)-p
        lik = []
        expected_param = np.zeros(shape =(2,t))
        predicted_param = np.zeros(shape =(2,t))
        for i in range(t):
            print(R[i])           
            if(R[i]< 10**(-4)*5):
                lik.append(0)
                continue
            #select elements in the run and reverse to preserve order
            x_run = self.r[0:(i+self.p)]
            x_run.reverse()
            #generate the model with this data
            dat = {'N': i+self.p,
                   'y': np.array(x_run).T.tolist()[0], #re-organise the data
                   'p': p,
                    'mean_a': self.mean_a, 
                    'sigma_a': self.sigma_a, 
                    'mean_sigma': self.mean_sigma,
                    'sigma_sigma': self.sigma_sigma,
                    'mean_mu': self.mean_mu,
                    'sigma_mu': self.sigma_mu,
                    'mean_gamma1': self.mean_gamma1,
                    'sigma_gamma1': self.sigma_gamma1,               
                  }
            trace = Single_Reg_time.sampling(data=dat, iter=5000,warmup =3000, chains=5,init = '0', seed =2532, control=dict(adapt_delta=0.99,max_treedepth=15))
            la = trace.extract()  # return the chains

            new_gamma = la['new_gamma'] #predicted
            gamma = la['gamma']
            expected_param[:,i] = [np.mean(np.exp(gamma)[:,-1]), np.std(np.exp(gamma)[:,-1])]
            predicted_param[:,i] = [np.mean(np.exp(new_gamma)), np.std(np.exp(new_gamma))]
            lik.append(np.mean(poisson.pmf(data, mu = np.exp(new_gamma))))
        return lik,expected_param, predicted_param
