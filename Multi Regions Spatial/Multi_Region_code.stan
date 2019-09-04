data {
    int  K; //number of regions
    int N; //number of points
    int y[K,N]; //observations
    vector[K] Mu_Mean; //the mean of Mu
    matrix[K,K] Mu_Sigma; //Covariance matrix of M 
    real sigma_mean; //mean of sigma
    real sigma_sd; //standard deviation of sigma
    real Omega_k; //parameter of LKJ corr
}

parameters {
    vector[K] Mu; //mean of normal distribution
    vector[K] Gamma; //vector of gammas for the regions
    corr_matrix[K] Omega; //correlation matrix
    vector<lower=0>[K] sigma; //vector to diagonal matrix
}

transformed parameters {
    cov_matrix[K] Sigma; 
    Sigma = quad_form_diag(Omega, sigma); //Sigma=diag(sigma)*Omega*diag(sigma)
}
model {
    real k = K;
     //prior on the mean
    Mu ~ multi_normal(Mu_Mean, Mu_Sigma);
    sigma ~ normal(sigma_mean,sigma_sd);
    // LKJ prior on the correlation matrix 
    Omega ~ lkj_corr(Omega_k); 
    Gamma ~ multi_normal(Mu, Sigma);
    //Likelihood
    for (i in 1:K) y[i,] ~ poisson(square(Gamma[i]));
}
