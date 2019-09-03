# BOCD-Gaus_Cox_P

We will perform Bayesian online changepoint detection (BOCD) on spatio-temporal point processes. The spatio-temporal models we shall consider are Cox processes, which we will fit to the data at each time step, to evaluate the posterior parameters distribution for every single block (sequence with the same generative parameters) and then apply a more general version of the algorithm in Adams et al. to detect changepoints in streaming data.

