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


darts <- arrow::open_dataset("darts/forecasts/", format="csv") |>
  filter(type == "predicted")
greta <- arrow::open_dataset("greta/forecasts/", format="csv") |>
  mutate(forecasting_model = "MCMC", reps=1, group=1) |>
  filter(type == "predicted") |> 
  rename(iter = i)

  
  
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
arrow::write_parquet(combined, "manuscript/data/forecasts.parquet")
arrow::write_parquet(sims, "manuscript/data/observations.parquet")


forecasts <- read_parquet("manuscript/data/forecasts.parquet")
observations <- read_parquet("manuscript/data/observations.parquet")

hopf_obs <- observations |> filter(simulation=="hopf")
stochastic_obs <- observations |> filter(simulation=="stochastic")
saddle_obs <- observations |> filter(simulation=="saddle")

forecasts |> filter(simulation=="hopf", reps == 1, group == 1) |> 
  ggplot(aes(t, col=type, fill=type)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha=0.5) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha=0.5, data = hopf_sims) +
  geom_line(aes(y=mean)) +
  geom_line(aes(y=mean), data = hopf_obs) +
  facet_grid(forecasting_model ~ variable, scales = "free") +
  theme_bw() + 
  ggtitle("A. Hopf bifurcation")

forecasts |> filter(simulation=="saddle", group == 1, reps == 1) |> 
  ggplot(aes(t, col=type, fill=type)) + 
  geom_ribbon(aes(ymin = pmax(lower,0), ymax = upper), alpha=0.5) +
  geom_ribbon(aes(ymin = pmax(lower,0), ymax = upper), alpha=0.5, data = saddle_sims) +
  geom_line(aes(y=mean)) +
  geom_line(aes(y=mean), data = saddle_obs) +
  facet_wrap(~forecasting_model, scales = "free") +
  #facet_grid(forecasting_model ~ reps, scales = "free") +
  theme_bw() + 
  ggtitle("B. Saddle-Node bifurcation")

forecasts |> filter(simulation=="stochastic", group == 1, reps==1) |> 
  ggplot(aes(t, col=type, fill=type)) + 
  geom_ribbon(aes(ymin = pmax(lower,0), ymax = upper), alpha=0.5) +
  geom_ribbon(aes(ymin = pmax(lower,0), ymax = upper), alpha=0.5, data = stochastic_sims) +
  geom_line(aes(y=mean)) +
  geom_line(aes(y=mean), data = stochastic_obs) +
  facet_wrap(~forecasting_model, scales = "free") +
  #facet_grid(forecasting_model ~ reps, scales = "free") +
  theme_bw() + 
  ggtitle("C. Stochastic transition")


#####################

