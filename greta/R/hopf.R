library(tidyverse, quietly = TRUE)
library(greta, quietly = TRUE)
library(bayesplot, quietly = TRUE)

step_hopf <- function(H, P, eta,  t=0, p) {
  K <- p$Ko + p$delta*(t+p$t_init)
  N <- numeric(2)
  N[1] <- H * exp(p$r * (1 - H/K) - p$c * P + eta[1])
  N[2] <- H * exp(p$r * (1 - H/K) ) * (1 - exp(-p$c * P+ eta[2]) )
  N[1] <- np.clip(N[1], 0, 100)
  N[2] <- np.clip(N[2], 0, 100)
  N
}

simulate_hopf <- function(t_max = 100, p) {
  eta <- array(c(rnorm(t_max, 0, p$sigma_H), 
                 rnorm(t_max, 0, p$sigma_P)),
               dim=c(t_max,2))
  N   <- array(NA, c(t_max,2))
  
  N[1,] <- c(p$H_init, p$P_init)
  for (t in 1:(t_max-1)) {
    N[t+1,] <- step_hopf(N[t,1], N[t,2], eta[t,], t=t, p = p)
  }
  tibble::tibble(t = p$t_init + 1:t_max, H = N[,1], P = N[,2])
}


greta_model_hopf <- function(train) {
  gsims <- train |> 
    group_by(i) |> 
    mutate(H1 = lead(H),
           P1 = lead(P)) |> 
    filter(t<max(t)) |> 
    ungroup()
  stopifnot(all(gsims$P1 > 0))
  
  H_t  <- gsims$H 
  H_t1 <- log( gsims$H1 ) 
  P_t  <- gsims$P
  P_t1 <- log( gsims$P1 )
  t <- gsims$t
  
  
  
  r <- lognormal(0, 1)
  c <- lognormal(0, 1)
  Ko <- uniform(0, 50)
  delta <- uniform(-1,1)
  sigma_H <- lognormal(0, 1)
  sigma_P <- lognormal(0, 1)
  
  K <- Ko + delta * t
  mu_H <- log( H_t ) +  (r * (1 - H_t/K) - c * P_t) 
  distribution(H_t1) <- normal(mu_H, sigma_H )
  
  
  mu_P <- log( H_t ) +  (r * (1 - H_t/K) ) + log(1 - exp(-c * P_t) )
  distribution(P_t1) <- normal(mu_P, sigma_P )
  m <- model(r, c, Ko, delta, sigma_H, sigma_P)
  m
  
}