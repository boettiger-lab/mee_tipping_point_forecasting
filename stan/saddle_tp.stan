data {
    int<lower=0> n; // Number of time series
    int<lower=0> t_max; // Duration of time series
    real<lower=0> x[n,t_max]; // State
}
parameters {
    real<lower=0> sigma; // Noise
    real<lower=0> r; // Growth rate
    real<lower=0> K; // Carrying Capacity
    real<lower=0> s; // Saturation term
    real<lower=0> h; // initial state degradation
    real<lower=0> a; // Constant degrad increase
}
transformed parameters {
    real<lower=0> mu[n,t_max]; // Mean State
    real<lower=0> alpha; //Degradation
    for (i in 1:n) {
        mu[i,1] = x[i,1];
        alpha = h;
        for (j in 2:t_max) {
          mu[i,j] = x[i,j-1] + r * x[i,j-1] * (1 - x[i,j-1] / K)  - alpha * (x[i,j-1]^2 / (s^2 + x[i,j-1]));
          alpha = alpha + a;
        }
    }
}
model {
    // Priors
    r ~ normal(0, 1);
    K ~ normal(0, 1);
    sigma ~ normal(0, 0.2);
    s ~ normal(0, 0.25);
    h ~ normal(0, 0.25);
    a ~ normal(0, 0.1);
    
    for (i in 1:n) {
    // Likelihood
        for (j in 2:t_max) {
            x[i,j] ~ normal(mu[i,j], sigma);
        }
    }
}

