library(arrow)
library(tidyverse)
library(forcats)
bound <- function(x, percentile = 0.975) {
  dplyr::nth(x, percentile * length(x), order_by = x)
}

reference <- arrow::open_dataset("greta/forecasts/", format="csv") |> 
  filter(type != "predicted") |> 
  collect() |>
  mutate(variable = as.character(fct_recode(variable, 
                                        host = "H", 
                                        parasitoid = "P", 
                                        X = "N")),
         type = as.character(fct_recode(type, observed = "true"))) |>
  rename(iter = i)


# group indicates the identifier for training data (0:5)
# reps indicates how many replicate training data sets were used (1 or 10)
# here we focus only on the case of a single training series.
# note that the MCMC data only consider rep=1, group=1 case anyway.
darts <- arrow::open_dataset("darts/forecasts/", format="csv") |>
  filter(type == "predicted") |> 
  filter(group == 1, reps == 1)
greta <- arrow::open_dataset("greta/forecasts/", format="csv") |>
  mutate(forecasting_model = "MCMC", reps=1, group=1) |>
  filter(type == "predicted") |> 
  rename(iter = i)


combined <- 
  bind_rows(collect(darts), collect(greta)) |> 
  mutate(variable = as_character(fct_recode(variable, 
                                        host = "H", 
                                        parasitoid = "P", 
                                        X = "N"))
         )


arrow::write_parquet(combined, "manuscript/data/forecasts.parquet")
arrow::write_parquet(reference, "manuscript/data/observations.parquet")


hopf_obs <- reference |> mutate(variable = as.character(variable)) |> filter(iter %in% 1:15, simulation == "hopf", variable !="X")
saddle_obs <- reference |> filter(iter %in% 1:15, simulation == "saddle")
stochastic_obs <- reference |> filter(iter %in% 1:15, simulation == "stochastic")


combined  |> mutate(variable = as.character(variable)) |> filter(iter %in% 1:15, simulation == "hopf", variable !="X") |> 
  ggplot(aes(x=t, y=value, group = interaction(iter, variable), col=type)) + 
  geom_line(alpha=0.1, data = hopf_obs) +
  geom_line(alpha=0.1) +
  geom_line(data = filter(hopf_obs, type=="historical")) +
  facet_grid(forecasting_model ~ variable, scales = "free") +
  theme_bw()  + coord_cartesian(ylim = c(0,15), xlim=c(75, 200))


combined |> filter(iter %in% 1:15, simulation == "saddle") |> 
  ggplot(aes(x=t, y=value, group = interaction(iter, variable), col=type)) + 
  geom_line(alpha=0.1, data = saddle_obs) +
  geom_line(alpha=0.1) +
  geom_line(data = filter(saddle_obs, type=="historical")) +
  facet_wrap(~forecasting_model, scales = "free") +
  theme_bw()  + coord_cartesian(ylim = c(0,1))


combined |> filter(iter %in% 1:15, simulation == "stochastic") |> 
  ggplot(aes(x=t, y=value, group = interaction(iter, variable), col=type)) + 
  geom_line(alpha=0.1, data = filter(stochastic_obs, type=="observed")) +
  geom_line(alpha=0.1) +
  geom_line(data = filter(stochastic_obs, type=="historical")) +
  facet_wrap(~forecasting_model, scales = "free") +
  theme_bw()  + coord_cartesian(ylim = c(0,0.8))







