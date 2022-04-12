library(tidyverse)
set.seed(1234567)

np.clip <- function(x, a, b) {
  if(x < a) return(a)
  if(x > b) return(b)
  x
}

# View the bistable curve: 
# curve( (0.75 - x)**2 * (0.25 - x), 0, 1)

step <- function(N, eta,  a = 0.75, b = 0.25) {
  N <- N + (a - N)**2 * (b - N) + eta
  N <- np.clip(N, 0, 1)
  N
}

# simulate
simulate <- function(N_init = 0.75,
                     t_max = 250L,
                     mu =0, # -5e-4,
                     sigma = 1e-2,
                     a = 0.75, 
                     b = 0.25) {
  eta <- rnorm(t_max, mu, sigma)
  N   <- numeric(t_max)
  N[1] <- N_init
  for (t in 1:(t_max-1)) {
    N[t+1] <- step(N[t], eta[t], a = a, b = b)
  }
  tibble::tibble(t = 1:t_max, N = N)
}

sims <- purrr::map_dfr(1:100, \(i) simulate(), .id = "i")
sims |> ggplot(aes(t, N, group=i)) + geom_line(alpha=0.06)

sims <- sims |> group_by(i) |> mutate(xt1 = lead(N)) | filter(t<max(t))

x_t <- sims$N
x_t1 <- sims$xt1

library(greta)
a <- uniform(0, 2)
b <- uniform(0, 2)
sigma <- uniform(0, 1)

# Model   (mean <-  may(x_t, p))
mean <-  x_t + (a - x_t) ^ 2 * (b - x_t)
distribution(x_t1) <- normal(mean, sigma )

m <- model(a, b, sigma)

mmcmc <- memoise::memoise(mcmc, cache = memoise::cache_filesystem("mcmc_cache"))

bench::bench_time({                 
  draws <- mmcmc(m, n_samples = 10000, warmup = 5000, chains = 4, verbose = FALSE)
})

