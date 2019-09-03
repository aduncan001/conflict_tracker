data {
    int N; //number of observations in block
    int y[N]; //number of data points
    //prior parameter
    int p;
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
    vector[p] a; //coefficients of AR
    real mu; //constant term in AR
    real sigma; //variance of error term
    real new_err;
    vector[N-p] err;
    vector[p] gamma_prior;
}

transformed parameters{
    vector[N] gamma;
    real temp_new = 0;
    real temp;
    real new_gamma;
    //define prior on first p gammas
    for (z in 1:p){
        gamma[z] = gamma_prior[z];
    }
    //define the time series AR(p)
    for (i in (p+1):(N)){
        temp = mu;
        for(j in 1:p){
            temp += a[j]*gamma[i-p+j-1];
        }
        gamma[i] = temp + sigma*err[i-p];
        }
    for (k in 1:p){
        temp_new += a[k]*gamma[N-p+k];
        }
    new_gamma = mu + temp_new + sigma*new_err;
}

model {
    //specify distributions on parameters
    a ~ normal(mean_a, sigma_a);
    sigma ~ normal(mean_sigma,sigma_sigma);
    mu ~ normal(mean_mu,sigma_mu);
    gamma_prior ~ normal(mean_gamma1,sigma_gamma1);
    err ~ normal(0,1);
    y ~ poisson(exp(gamma));
    new_err ~ normal(0,1);
}
