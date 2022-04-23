library(tidyverse)
library(greta)
library(bayesplot)
library(scoringRules)



np.clip <- function(x, a, b) {
  if(x < a) return(a)
  if(x > b) return(b)
  x
}

step_sn <- function(N, 
                 eta,
                 t,
                 pars) {
  h <- np.clip(pars$h_init + pars$alpha * (pars$t_init + t), 0, 0.27)
  N <- N + pars$r * N * (1 - N / pars$K) - 
    h * (N**2 / (pars$s**2 + N**2)) + eta
  N <- np.clip(N, 0, 100)
  N
}
# simulate
simulate_sn <- function(N_init = 0.75,
                     t_max = 250L,
                     pars = list(
                       r = 1,
                       K = 1,
                       s = 0.1,
                       h_init = .15,
                       alpha = 0.0015,
                       mu = 0,
                       sigma = 0.00,
                       t_init = 0
                     )
) {

  eta <- rnorm(t_max, pars$mu, pars$sigma) # mu = 0, no drift
  N   <- numeric(t_max)
  N[1] <- N_init
  for (t in 1:(t_max-1)) {
    N[t+1] <- step_sn(N[t], eta[t], t, pars)
  }
  tibble::tibble(t = pars$t_init + 1:t_max, N = N)
}


greta_model_sn <- function(train) {
  gsims <- train |> 
    group_by(i) |> 
    mutate(xt1 = lead(N)) |>
    filter(t<max(t))
  x_t <- gsims$N
  x_t1 <- gsims$xt1
  t <- gsims$t
  
  r <- uniform(0, 10)
  K <- uniform(0, 10)
  s <- uniform(0, 10)
  h_init <- uniform(0, 10)
  alpha <- uniform(0, 10)
  sigma <- uniform(0, 10)
  h <- h_init + alpha * t
  mean <-  x_t + r*x_t*(1 - x_t / K) - h*(x_t^2 / (s^2 + x_t^2))
  distribution(x_t1) <- normal(mean, sigma)
  m <- model(r, K, s, h_init, alpha, sigma)
  
}


