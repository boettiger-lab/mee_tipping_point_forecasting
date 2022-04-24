library(tidyverse)
library(tidyselect)
library(memoise)
library(greta)
library(scoringRules)



np.clip <- function(x, a, b) {
  if(x < a) return(a)
  if(x > b) return(b)
  x
}

mmcmc <- memoise::memoise(greta::mcmc, cache = memoise::cache_filesystem("cache"))

plot_posteriors <- function(draws, pars) {
  true <- as_tibble(pars) |> 
    select(-contains("init")) |> 
    gather(variable, value)
  
  bind_rows(map(draws, as_tibble)) |>
    gather(variable, value) |> ggplot() + 
    geom_histogram(aes(value), bins = 30)  +
    geom_vline(data = true, aes(xintercept = value), col = "red", lwd = 1) + 
    facet_wrap(~variable, scales = "free")
}



get_inits <- function(train, vars = "N") {
  train_t_max <- max(train$t)
  
  # initial conditions for forecast using final state distribution in training 
  left_off <- train |> 
    filter(t == train_t_max) |> 
    select(all_of(vars)) |> 
    slice_sample(n = test_reps, replace=TRUE) 
  names(left_off) <- paste0(names(left_off), "_init")
  left_off |>
    mutate(t_init = train_t_max)
}

sample_posteriors <- function(draws, inits = NULL, test_reps = 100) {  
    bind_rows(map(draws, as_tibble)) |> 
    sample_n(test_reps) |>
    bind_cols(inits)
}

forecast_dist <- function(posterior_samples, simulate) {
  # run simulate() with each row of parameters

    posterior_samples |>
    purrr::transpose() |>
    map_dfr(function(q) 
      simulate(t_max = test_t_max, p = q),
      .id = "i")
}


compare_forecast <- function(draws, train, test, simulate, vars, test_reps = 100) {
  inits <- get_inits(train, vars)
  posterior_sims <-
    sample_posteriors(draws, inits, test_reps=test_reps) |>
    forecast_dist(simulate)

  bind_rows(
    mutate(train, type="historical"),
    mutate(test, type="true"), 
    mutate(posterior_sims, type="predicted")) |>
    pivot_longer(vars, values_to="value", names_to="variable")
}




compute_scores <- function(observed, dat) {
  logsscore <- scoringRules::logs_sample(observed, dat)
  crpsscore <- scoringRules::crps_sample(observed, dat)
  data.frame(logs = logsscore[-1], crps =  crpsscore[-1])
  
}

rep_scores <- function(combined, var) {
  obs <- 
    combined |> 
    filter(type == "predicted") |>
    filter(variable == var) |>
    pivot_wider(id_cols = "t", 
                names_from="i",
                values_from = "value") |> 
    select(-t) |> as.matrix()
  
  combined |> 
    filter(type == "true", variable == var) |> 
    group_by(i) |> 
    group_modify(~ compute_scores(.x$value, obs)) |>
    mutate(variable = var)
  
}

