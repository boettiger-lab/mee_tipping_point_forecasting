library(arrow)
library(tidyverse)

bound <- function(x, percentile = 0.975) {
  dplyr::nth(x, percentile * length(x), order_by = x)
}

reference <- arrow::open_dataset("greta/forecasts/", format="csv") |> 
  filter(type != "predicted") |> 
  collect()


sims <- reference |>
  group_by(simulation, variable, type, t) |> 
  summarise(mean = mean(value), 
            upper = bound(value, 0.975), 
            lower = bound(value, 0.025), 
            .groups = "drop") |>
  mutate(variable = forcats::fct_recode(variable, 
                                        host = "H", 
                                        parasitoid = "P", 
                                        X = "N")) 


darts <- arrow::open_dataset("darts/forecasts/", format="csv")
darts <- darts |> filter(type == "predicted")
greta <- arrow::open_dataset("greta/forecasts/", format="csv") |> mutate(forecasting_model = "MCMC", reps=1, group=1)
greta <- greta |>  filter(type == "predicted") |> rename(iter = i)

  
  
## Compute summary stats in arrow before reading into R
darts_df <- darts |>  
  collect() |> 
  group_by(simulation, variable, type, t, reps, group, forecasting_model) |> 
  summarise(mean = mean(value), 
            upper = bound(value, 0.975), 
            lower = bound(value, 0.025), 
            .groups = "drop")
greta_df <- greta |>
  collect() |> 
  group_by(simulation, variable, type, t, reps, group, forecasting_model) |> 
  summarise(mean = mean(value), 
            upper = bound(value, 0.975), 
            lower = bound(value, 0.025), 
            .groups = "drop")

combined <- bind_rows(darts_df, greta_df) |> 
  mutate(variable = forcats::fct_recode(variable, 
                                        host = "H", 
                                        parasitoid = "P", 
                                        X = "N")) 

hopf_sims <- sims |> filter(simulation=="hopf")
stochastic_sims <- sims |> filter(simulation=="stochastic")
saddle_sims <- sims |> filter(simulation=="saddle")

combined |> filter(simulation=="hopf", reps == 1, group == 1) |> 
  ggplot(aes(t, col=type, fill=type)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha=0.5) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha=0.5, data = hopf_sims) +
  geom_line(aes(y=mean)) +
  geom_line(aes(y=mean), data = hopf_sims) +
  facet_grid(forecasting_model ~ variable, scales = "free") +
  theme_bw() + 
  ggtitle("A. Hopf bifurcation")




combined |> filter(simulation=="saddle", group == 1) |> 
  ggplot(aes(t, col=type, fill=type)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha=0.5) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha=0.5, data = saddle_sims) +
  geom_line(aes(y=mean)) +
  geom_line(aes(y=mean), data = saddle_sims) +
  facet_grid(forecasting_model ~ reps, scales = "free") +
  theme_bw() + 
  ggtitle("B. Saddle-Node bifurcation")

combined |> filter(simulation=="stochastic", group == 1) |> 
  ggplot(aes(t, col=type, fill=type)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha=0.5) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha=0.5, data = stochastic_sims) +
  geom_line(aes(y=mean)) +
  geom_line(aes(y=mean), data = stochastic_sims) +
  facet_grid(forecasting_model ~ reps, scales = "free") +
  theme_bw() + 
  ggtitle("C. Stochastic transition")


#####################




compute_scores <- function(observed, dat) {
  logsscore <- scoringRules::logs_sample(observed, dat)
  crpsscore <- scoringRules::crps_sample(observed, dat)
  data.frame(logs = logsscore, crps =  crpsscore)
  
}

rep_scores <- function(predicted, sim) {
  obs <- 
    predicted |> 
    pivot_wider(id_cols = "t", 
                names_from="iter",
                values_from = "value") |> 
    select(-t) |> 
    as.matrix()

  scores <- sim |> 
    group_by(iter) |> 
    group_modify(~ compute_scores(.x$value, obs), .keep = TRUE) 
  
  ## crude way to restore metadata
  scores <- sim |> 
    select(t, variable, simulation) |> 
    bind_cols(scores)
  scores
}


score_it <- function(scenario, model, var, darts, reference) {
  predicted <- darts |> 
    filter(simulation=={{scenario}}, 
           forecasting_model == {{model}}, 
           variable == {{var}},
           group==1, 
           reps==1,
           type == "predicted",
           ) |>
    collect() 
  sim <- reference |> 
    filter(simulation=={{scenario}},
           variable=={{var}},
           type == "true") |>
    rename(iter = i) |> 
    collect()
  scores <- rep_scores(predicted, sim)

}
scores <- score_it(scenario = "hopf", "gru", "H", darts, reference) 

scores |> 
  ggplot(aes(t, crps, group=iter)) + geom_point()
