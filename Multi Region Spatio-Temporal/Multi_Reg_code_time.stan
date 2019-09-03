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
