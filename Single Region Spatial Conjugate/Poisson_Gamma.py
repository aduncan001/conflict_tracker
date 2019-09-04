class Poisson_Gamma:
    #code initially taken from
    #https://jeremy9959.net/Blog/bayesian-online-changepoint-fixed/
    def __init__(self, k, theta):
        #initialise prior parameters
        self.k_prior = self.k = np.array([k])
        self.theta_prior = self.theta = np.array([theta])

    def pdf(self, data,R):
        #evaluate predictive probability for all posterior parameters
        return nbinom.pmf(data,self.k, 1/(1+self.theta))

    def update_run(self, data):
        #add information gained from new point
        self.k = np.concatenate((self.k_prior, self.k+data))
        self.theta = np.concatenate((self.theta_prior, self.theta/(1+self.theta)))
