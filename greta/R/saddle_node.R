library(tidyverse, quietly = TRUE)
library(greta, quietly = TRUE)
library(bayesplot, quietly = TRUE)
library(scoringRules)

step_sn <- function(N, 
                 eta,
                 t,
                 p) {
  h <- p$ho + p$alpha * (p$t_init + t)
  N <- N + p$r * N * (1 - N / p$K) - 
    h * (N**2 / (p$s**2 + N**2)) + eta
  N <- np.clip(N, 0, 100)
  N
}
# simulate
simulate_sn <- function(t_max = 250L, p) {

  eta <- rnorm(t_max, 0, p$sigma) # mu = 0, no drift
  N   <- numeric(t_max)
  N[1] <- p$N_init
  for (t in 1:(t_max-1)) {
    N[t+1] <- step_sn(N[t], eta[t], t, p)
  }
  tibble::tibble(t = p$t_init + 1:t_max, N = N)
}


greta_model_sn <- function(train,
                           r     = uniform(0, 2),
                           K     = uniform(0, 2),
                           s     = uniform(0.0, 0.2),
                           ho    = uniform(0, 1),
                           alpha = uniform(0, 0.01),
                           sigma = uniform(0, 0.05)
                           ) {
  gsims <- train |> 
    group_by(i) |> 
    mutate(xt1 = lead(N)) |>
    filter(t<max(t))
  x_t <- gsims$N
  x_t1 <- gsims$xt1
  t <- gsims$t
  

  h <- ho + alpha * t
  mean <-  x_t + r*x_t*(1 - x_t / K) - h*(x_t^2 / (s^2 + x_t^2))
  distribution(x_t1) <- normal(mean, sigma)
  m <- model(r, K, s, ho, alpha, sigma)
  m
}


