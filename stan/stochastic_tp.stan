data {
    int<lower=0> n; // Number of time series
    int<lower=0> t_max; // Duration of time series
    real<lower=0> x[n,t_max]; // State
}
parameters {
    real<lower=0> x1; // Unstable Equilibrium
    real<lower=0> x2; // Stable equilibrium
    real<lower=0> sigma; // Noise
}
transformed parameters {
    real<lower=0> mu[n,t_max]; // Mean State
    for (i in 1:n) {
        mu[i,1] = x[i,1];
        for (j in 2:t_max) {
          mu[i,j] = x[i,j-1] + (x1 - x[i,j-1])^2 * (x2 - x[i,j-1]);
        }
    }
}
model {
    // Priors
    x1 ~ beta(500, 130);
    x2 ~ beta(80, 300);
    sigma ~ beta(2, 50);
    
    for (i in 1:n) {
    // Likelihood
        for (j in 2:t_max) {
            x[i,j] ~ normal(mu[i,j], sigma);
        }
    }
}
generated quantities {
    real<lower=0> x_ppc[t_max]; 
    
    x_ppc[1] = 0.75;
    for (j in 2:t_max) {
      x_ppc[j] = normal_rng(x_ppc[j-1] + (x1 - x_ppc[j-1])^2 * (x2 - x_ppc[j-1]), sigma);
    }
}