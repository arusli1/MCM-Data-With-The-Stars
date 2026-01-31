

## Data


``` r
data_summary <- read_csv(file.path(art, "data_summary.csv"), show_col_types = FALSE)
data_summary
```

```
## # A tibble: 1 × 6
##   n_rows n_seasons n_items min_season max_season n_missing_placement_final
##    <dbl>     <dbl>   <dbl>      <dbl>      <dbl>                     <dbl>
## 1    421        34     421          1         34                         0
```


``` r
train_seasons <- read_csv(file.path(art, "train_seasons.csv"), show_col_types = FALSE)
test_seasons <- read_csv(file.path(art, "test_seasons.csv"), show_col_types = FALSE)
list(
  n_train_seasons = nrow(train_seasons),
  n_test_seasons = nrow(test_seasons),
  test_seasons = test_seasons$test_season
)
```

```
## $n_train_seasons
## [1] 27
## 
## $n_test_seasons
## [1] 7
## 
## $test_seasons
## [1] 28 29 30 31 32 33 34
```

## Model

We fit a Plackett–Luce ranking model with item covariates. For an item \(i\) with covariate vector \(x_i\),
the model assigns a *worth*:

\[
w_i = \exp(x_i^\top \beta),
\]

and the probability of an observed full ranking is the product of stagewise multinomial choices (standard Plackett–Luce likelihood).

### Shrinkage / regularization mechanism

Because PlackettLuce does not provide separate per-variable penalties for factor levels, we implement shrinkage for sparse categorical variables via:

- **Pseudo-comparisons (`npseudo`)**: adds pseudo-information that regularizes the fit toward equal worths.
- **Rare-level collapsing (partner/state/country only)**: levels with low training support are mapped to `"Other"` (threshold tuned).

Industry and age are treated as **non-shrink** variables (industry levels are not collapsed).

## Hyperparameter selection (training only)


``` r
cv_grid <- read_csv(file.path(art, "cv_hyperparams.csv"), show_col_types = FALSE)
best <- read_csv(file.path(art, "best_hyperparams.csv"), show_col_types = FALSE)
best
```

```
## # A tibble: 1 × 10
##   npseudo min_partner min_state min_country lambda spearman_mean kendall_mean
##     <dbl>       <dbl>     <dbl>       <dbl>  <dbl>         <dbl>        <dbl>
## 1       2           1         1           3     10         0.402        0.291
## # ℹ 3 more variables: top1_mean <dbl>, top3_mean <dbl>, ndcg_mean <dbl>
```


``` r
cv_grid %>% arrange(desc(kendall_mean), desc(spearman_mean)) %>% head(10)
```

```
## # A tibble: 10 × 10
##    npseudo min_partner min_state min_country lambda spearman_mean kendall_mean
##      <dbl>       <dbl>     <dbl>       <dbl>  <dbl>         <dbl>        <dbl>
##  1       2           1         1           3     10         0.402        0.291
##  2       1           1         1           3     10         0.400        0.289
##  3       2           1         3           2     10         0.395        0.288
##  4       2           1         1           1     10         0.388        0.285
##  5       2           2         1           3     10         0.392        0.284
##  6       1           1         1           2     10         0.390        0.284
##  7       1           1         3           5     10         0.393        0.283
##  8       2           1         1           5     10         0.392        0.283
##  9       2           5         1           3     10         0.391        0.283
## 10       1           2         1           3     10         0.393        0.283
## # ℹ 3 more variables: top1_mean <dbl>, top3_mean <dbl>, ndcg_mean <dbl>
```

## Test-set performance (season heldout)


``` r
metrics_overall <- read_csv(file.path(art, "metrics_overall.csv"), show_col_types = FALSE)
metrics_overall
```

```
## # A tibble: 1 × 9
##   n_seasons spearman_mean spearman_sd kendall_mean kendall_sd  top1  top3
##       <dbl>         <dbl>       <dbl>        <dbl>      <dbl> <dbl> <dbl>
## 1         7         0.465       0.126        0.329      0.111 0.286 0.714
## # ℹ 2 more variables: ndcg_mean <dbl>, ndcg_sd <dbl>
```


``` r
metrics_by_season <- read_csv(file.path(art, "metrics_by_season.csv"), show_col_types = FALSE)
metrics_by_season
```

```
## # A tibble: 7 × 6
##   season spearman kendall_tau  top1  top3  ndcg
##    <dbl>    <dbl>       <dbl> <dbl> <dbl> <dbl>
## 1     28    0.587       0.424     0     1 0.736
## 2     29    0.471       0.352     0     0 0.585
## 3     30    0.493       0.333     0     1 0.653
## 4     31    0.303       0.2       1     1 0.908
## 5     32    0.376       0.253     1     1 0.914
## 6     33    0.372       0.234     0     0 0.611
## 7     34    0.656       0.508     0     1 0.864
```

## Predicted vs actual rankings


``` r
pred <- read_csv(file.path(art, "predictions_test_seasons.csv"), show_col_types = FALSE)
pred %>% arrange(season, predicted_rank) %>% head(20)
```

```
## # A tibble: 20 × 10
##    season celebrity_name       true_rank predicted_rank  score ballroom_partner 
##     <dbl> <chr>                    <dbl>          <dbl>  <dbl> <chr>            
##  1     28 Ally Brooke                  3              1  0.698 Sasha Farber     
##  2     28 Lauren Alaina                4              2  0.592 Gleb Savchenko   
##  3     28 Hannah Brown                 1              3  0.562 Alan Bersten     
##  4     28 Ray Lewis                   11              4  0.561 Cheryl Burke     
##  5     28 James Van Der Beek           5              5  0.556 Emma Slater      
##  6     28 Kate Flannery                7              6  0.449 Pasha Pashkov    
##  7     28 Kel Mitchell                 2              7  0.380 Witney Carson    
##  8     28 Lamar Odom                  10              8  0.378 Peta Murgatroyd  
##  9     28 Karamo Brown                 8              9  0.341 Jenna Johnson    
## 10     28 Sailor Brinkley-Cook         9             10  0.221 Valentin Chmerko…
## 11     28 Sean Spicer                  6             11 -0.199 Lindsay Arnold   
## 12     28 Mary Wilson                 12             12 -0.231 Brandon Armstrong
## 13     29 Skai Jackson                 5              1  0.802 Alan Bersten     
## 14     29 Johnny Weir                  6              2  0.614 Britt Stewart    
## 15     29 AJ McLean                    7              3  0.557 Cheryl Burke     
## 16     29 Veron Davis                 11              4  0.535 Peta Murgatroyd  
## 17     29 Nev Schulman                 2              5  0.465 Jenna Johnson    
## 18     29 Chrishell Stause             8              6  0.360 Gleb Savchenko   
## 19     29 Jesse Metcalfe              12              7  0.358 Sharna Burgess   
## 20     29 Nelly                        3              8  0.343 Daniella Karagach
## # ℹ 4 more variables: celebrity_homestate <chr>,
## #   celebrity_homecountry_region <chr>, celebrity_industry <chr>,
## #   celebrity_age_during_season <dbl>
```

## Variable impacts (multiplicative worth)


``` r
coef_tbl <- read_csv(file.path(art, "coefficients.csv"), show_col_types = FALSE)
coef_tbl %>% arrange(desc(abs(estimate)))
```

```
## # A tibble: 114 × 3
##    term                                 estimate worth_multiplier
##    <chr>                                   <dbl>            <dbl>
##  1 celebrity_industryMagician             -1.32             0.266
##  2 celebrity_industryBeauty Pagent        -1.04             0.354
##  3 celebrity_industryJournalist           -0.700            0.497
##  4 celebrity_industryMusician              0.677            1.97 
##  5 celebrity_industryPolitician           -0.611            0.543
##  6 celebrity_industryModel                -0.592            0.553
##  7 (Intercept)                             0.548            1.73 
##  8 celebrity_industrySports Broadcaster   -0.527            0.590
##  9 celebrity_industryRadio Personality    -0.470            0.625
## 10 celebrity_industryComedian             -0.398            0.672
## # ℹ 104 more rows
```

Interpretation: each coefficient \(\beta_j\) is a log-multiplicative effect on worth; \(\exp(\beta_j)\) is the worth multiplier per 1 unit change in the covariate (or relative to the baseline level for factors).

## Limitations

- This is still a *season-level* ranking model; it does not model week-by-week dynamics.
- Shrinkage here is operationalized via `npseudo` and rare-level collapsing (not true random effects).
- Some categorical levels are sparse and will be partially pooled into `"Other"` for stability.


