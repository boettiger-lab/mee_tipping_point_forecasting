
``` r
library(tidyverse, quietly = TRUE)
library(greta, quietly = TRUE)
library(bayesplot, quietly = TRUE)
source("R/utils.R")
source("R/saddle_node.R")
```

``` r
set.seed(4242)
train_reps <- 1
train_t_max <- 250
test_t_max <- 250
test_reps <- 100
simulate <- simulate_sn
```

``` r
p <- list(r = 1,
          K = 1,
          s = 0.1,
          ho = .15,
          alpha = 0.0015/4,
          sigma = 0.02,
          t_init = 0,
          N_init = 0.75
        )
t_max <- train_t_max + test_t_max

sim <- purrr::map_dfr(1:test_reps, 
                      \(i) simulate(t_max=t_max, p = p),
                      .id = "i")
train <- sim |> filter(t <= train_t_max, i <= train_reps)
test <- sim |> filter(t > train_t_max)

#sim |> ggplot(aes(t, N, group=i))+geom_line(alpha = 0.1) + geom_vline((aes(xintercept=train_t_max))) 
```

``` r
m <- greta_model_sn(train)
```

    ## ℹ Initialising python and checking dependencies, this may take a moment.

    ## ✓ Initialising python and checking dependencies ... done!

    ## 

``` r
bench::bench_time({                 
  draws <- mmcmc(m, n_samples = 700000, warmup = 600000,
                 chains = 6, verbose = FALSE)
})
```

    ## process    real 
    ##   1.55d  21.48h

``` r
## draw test_reps number of samples
combined <- compare_forecast(draws, train, test, simulate, vars = "N",
                              test_reps, test_t_max)
```

    ## Note: Using an external vector in selections is ambiguous.
    ## ℹ Use `all_of(vars)` instead of `vars` to silence this message.
    ## ℹ See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This message is displayed once per session.

``` r
write_csv(combined, "data/saddlenode.csv.gz")
```

``` r
scores <-
  rep_scores(combined, "N") |> 
  mutate(scenario="stochastic", 
         model="MCMC", 
         reps = train_reps) 

write_csv(scores, "data/scores_saddlenode.csv.gz")
```

``` r
bayesplot::mcmc_trace(draws)
```

![](sn_mcmc_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
plot_posteriors(draws, p)
```

![](sn_mcmc_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
combined |> 
  group_by(t,type,variable) |> 
  summarise(mean = mean(value), sd = sd(value), .groups = "drop") |> 
 ggplot(aes(t, col=type)) + 
  geom_ribbon(aes(ymin = pmax(mean-2*sd,0), ymax = mean+2*sd, fill=type), alpha=0.5) +
  geom_line(aes(y=mean)) +
  geom_vline(aes(xintercept = train_t_max)) + facet_wrap(~variable, ncol=1)
```

    ## Warning in max(ids, na.rm = TRUE): no non-missing arguments to max; returning
    ## -Inf

![](sn_mcmc_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
  combined |> 
  filter(i %in% 1:train_reps) |> 
  ggplot(aes(t, value, col=type, group=interaction(i,type))) + 
    geom_line() +
    geom_vline(aes(xintercept = train_t_max)) + facet_wrap(~variable, ncol=1)
```

![](sn_mcmc_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
n <- test_reps*10
prior_sims <- tibble(
    r = runif(n, 0, 2),
    K = runif(n, 0, 2), # runif(n, 0, 2),
    s = runif(n, 0, 0.2),
    ho = runif(n, 0, 0.2), # rlnorm(n, 0,1),
    alpha = runif(n, 0, 0.001),
    sigma = runif(n, 0, 0.1) 
    ) |>
  bind_cols(get_inits(train, "N", n)) |> 
  forecast_dist(simulate_sn, test_t_max) |>
  mutate(type = "prior") |>
  pivot_longer("N", values_to="value", names_to="variable")
prior_sims |>
  ggplot(aes(t, value, group=i)) + geom_line(alpha=0.01)
```

![](sn_mcmc_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->
