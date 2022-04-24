library(tidyverse, quietly = TRUE)
library(greta, quietly = TRUE)
library(bayesplot, quietly = TRUE)

# View the bistable curve: 
#curve( (0.75 - x)**2 * (0.25 - x), 0.749999, 0.750001)
#curve(0*x,0,1,add = TRUE)

#curve((0.25 - x)**2 * (0.75 - x), 0, 1)
#curve(0*x,0,1,add = TRUE)

step_stoch <- function(N, eta,  a = 0.75, b = 0.25) {
  N <- N + (a - N)**2 * (b - N) + eta
  N <- np.clip(N, 0, 1)
  N
}

# simulate
simulate_stoch <- function(t_max = 250L,
                           p = list(
                             sigma = 1e-2,
                             a = 0.75, 
                             b = 0.25,
                             N_init = 0.75)) {
  eta <- rnorm(t_max, 0, p$sigma) # mu = 0, no drift
  N   <- numeric(t_max)
  N[1] <- p$N_init
  for (t in 1:(t_max-1)) {
    N[t+1] <- step_stoch(N[t], eta[t], a = p$a, b = p$b)
  }
  tibble::tibble(t = 1:t_max, N = N)
}


greta_model_stoch <- function(train) {
  gsims <- train |> group_by(i) |> mutate(xt1 = lead(N)) |> filter(t<max(t))
  
  
  x_t <- gsims$N
  x_t1 <- gsims$xt1
  
  library(greta)
  a <- uniform(0, 10)
  b <- uniform(0, 1)
  sigma <- uniform(0, 10)
  mean <-  x_t + (a - x_t) ^ 2 * (b - x_t)
  distribution(x_t1) <- normal(mean, sigma)
  m <- model(a, b, sigma)
  m
}