library(arrow)
library(tidyverse)

bound <- function(x, percentile = 0.975) {
  dplyr::nth(x, percentile * length(x), order_by = x)
}

reference <- arrow::open_dataset("greta/forecasts/", format="csv") |> 
  filter(type != "predicted") |> 
  collect() |> rename(iter = i) |>
  mutate(variable = forcats::fct_recode(variable, 
                                        X = "N")) 


darts <- arrow::open_dataset("darts/forecasts/", format="csv")|>
  filter(type == "predicted")
ensemble <- darts |> 
  mutate(iter = paste0(iter,"-", forecasting_model), 
                       forecasting_model="ml_ensemble") |>
  collect()

## eek big
#darts <- bind_rows(collect(darts), collect(ensemble))


greta <- arrow::open_dataset("greta/forecasts/", format="csv") |>
  mutate(forecasting_model = "MCMC", reps=1, group=1) |> 
  filter(type == "predicted") |> rename(iter = i) |> collect() |>
  mutate(variable = forcats::fct_recode(variable,  X = "N")) 


compute_scores <- function(observed, dat) {
  logsscore <- scoringRules::logs_sample(observed, dat)
  crpsscore <- scoringRules::crps_sample(observed, dat)
  data.frame(logs = logsscore, crps =  crpsscore)
  
}

rep_scores <- function(predicted, sim) {

  ## ensure corresponding time interval in both
  match_sim <- sim |> filter(t %in% unique(predicted$t)) 
  match_predicted <- predicted |> filter(t %in% unique(match_sim$t)) 
  
  
    
  # T x iter
  obs <- 
    match_predicted |> 
    pivot_wider(id_cols = "t", 
                names_from="iter",
                values_from = "value") |> 
    select(-t) |> 
    as.matrix()


  scores <- match_sim |>
    group_by(iter) |> 
    group_modify(~ compute_scores(.x$value, obs), .keep = TRUE) 
  
  ## crude way to restore metadata
  scores <- match_sim |> 
    select(t, variable, simulation) |> 
    bind_cols(scores)
  scores
}

score_it <- function(scenario, model, var, darts, reference) {
  predicted <- darts |> 
    filter(simulation=={{scenario}}, 
           forecasting_model =={{model}}, 
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
    collect()
  scores <- rep_scores(predicted, sim) |> 
    mutate(forecasting_model = model,
           variable = var)
  
}

cases <- darts |> 
  select(simulation, forecasting_model, variable) |>
  distinct() |> 
  collect()
for (i in 1:nrow(cases)) {
      scenario <- cases$simulation[[i]]
      model <- cases$forecasting_model[[i]]
      var <- cases$variable[[i]]
      df <- score_it(scenario, model, var, darts, reference)
      write_csv(df, paste0("scores/", paste(scenario, model, var,sep="-"), ".csv.gz"))
}



cases <- greta |> 
  select(simulation, forecasting_model, variable) |>
  distinct()
for (i in 1:nrow(cases)) {
  scenario <- cases$simulation[[i]]
  model <- cases$forecasting_model[[i]]
  var <- cases$variable[[i]]
  df <- score_it(scenario, model, var, greta, reference)
  write_csv(df, paste0("scores/", paste(scenario, model, var,sep="-"), ".csv.gz"))
}

cases <- ensemble |> 
  select(simulation, forecasting_model, variable) |>
  distinct() |> 
  collect()
for (i in 1:nrow(cases)) {
  scenario <- cases$simulation[[i]]
  model <- cases$forecasting_model[[i]]
  var <- cases$variable[[i]]
  df <- score_it(scenario, model, var, ensemble, reference)
  write_csv(df, paste0("scores/", paste(scenario, model, var,sep="-"), ".csv.gz"))
}
















scores <- arrow::open_dataset("../scores", format="csv") |> collect()

# shift logs score to strictly positive, non-infinite values so we can use a log-scale
shift <- scores |> 
  filter(!is.infinite(logs)) |> 
  mutate(logs = logs - min(logs) +.01) |>
  pivot_longer(c(logs, crps), 
               names_to = "metric", 
               values_to = "score")

shift |>
  ggplot(aes(forecasting_model, score, color=forecasting_model)) + 
  geom_jitter(alpha = 0.006, show.legend = FALSE) + 
  coord_flip() +
  facet_grid(simulation ~ metric, scales = "free") + 
  scale_y_log10() +
  theme_bw()

p1 <- over_time |> filter(metric == "crps") |>
  ggplot(aes(t, score, col = forecasting_model, fill=forecasting_model)) + 
  geom_ribbon(aes(y = score, ymin = ymin, ymax =ymax),
              shape=".", alpha=0.4, show.legend=FALSE) +
  geom_line(show.legend=FALSE) +
  facet_wrap(~simulation, scales="free", ncol=1) + 
  theme_bw() 

p2 <- over_time |> filter(metric == "logs") |>
  ggplot(aes(t, score, col = forecasting_model, fill=forecasting_model)) + 
  geom_ribbon(aes(y = score, ymin = ymin, ymax = ymax),
              shape=".", alpha=0.4, show.legend=FALSE) +
  geom_line() +
  facet_wrap(~simulation, scales="free", ncol=1) + 
  scale_y_log10() +
  theme_bw() 

p1 + p2
