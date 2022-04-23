library(tidyverse)
library(memoise)
library(greta)
library(scoringRules)

mmcmc <- memoise::memoise(greta::mcmc, cache = memoise::cache_filesystem("cache"))

plot_posteriors <- function(draws, pars) {
  true <- as_tibble(pars) |> gather(variable, value)
  bind_rows(map(draws, as_tibble)) |>
    gather(variable, value) |> ggplot() + 
    geom_histogram(aes(value), bins = 30)  +
    geom_vline(data = true, aes(xintercept = value), col = "red", lwd = 1) + 
    facet_wrap(~variable, scales = "free")
}


forecast <- function(draws, train, test, simulate) {
  train_t_max <- max(train$t)
  
  # get initial conditions for forecast from final state distribution in training samples
  left_off <- train |> 
    filter(t == train_t_max) |> 
    select(N) |> 
    rename(N_init = N) |>
    slice_sample(n = test_reps, replace=TRUE)
  
  # sample from posteriors
  posterior_samples <- 
    bind_rows(map(draws, as_tibble)) |> 
    sample_n(test_reps) |>
    mutate(t_init = train_t_max) |>
    bind_cols(left_off)
  
  # run simulate() with each row of parameters
  posterior_sims <- 
    posterior_samples |>
    mutate(mu=0) |>
    purrr::transpose() |>
    map_dfr(function(q) 
      simulate(t_max = test_t_max, pars = q, N_init = q$N_init),
      .id = "i")
  
  # Combine as single data.frame
  bind_rows(
    mutate(train, model="historical"),
    mutate(test, model="true"), 
    mutate(posterior_sims, model="predicted"))
}




scores <- function(observed, dat) {
  logsscore <- scoringRules::logs_sample(observed, dat)
  crpsscore <- scoringRules::crps_sample(observed, dat)
  data.frame(logs = mean(logsscore[-1]), crps =  mean(crpsscore[-1]))
  
}
