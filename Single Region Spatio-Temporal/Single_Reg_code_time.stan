data {
    int N; //number of observations in block
    int y[N]; //number of data points
    //define the prior parameters
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
    real a; //coefficients of AR(1)
    real mu; //constant term in AR
    real sigma; //variance of error term
    real gamma_prior;
    vector[N] err;
}

transformed parameters{
    vector[N] gamma;
    real new_gamma;
    gamma[1] = gamma_prior;
    if (N>1){ //if you have enogh datapoint define time series
    for (i in 2:(N)){
        gamma[i] = mu +  a*gamma[i-1]+ sigma*err[i-1];
        }
    }
    new_gamma = mu + a*gamma[N]+sigma*err[N]; //prediction
}

model {
    a ~ normal(mean_a, sigma_a);
    sigma ~ normal(mean_sigma,sigma_sigma);
    mu ~ normal(mean_mu, sigma_mu);
    gamma_prior ~ normal(mean_gamma1,sigma_gamma1);
    err ~ normal(0,1);
    y ~ poisson(exp(gamma));
}
