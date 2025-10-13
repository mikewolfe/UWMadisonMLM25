library(tidyverse)
library(here)
library(tidymodels)
library(textrecipes)
library(text2vec)
library(future)
theme_set(theme_classic())
tidymodels_prefer()

# Command line interface
args <- commandArgs(trailingOnly = TRUE)
outpre <- args[1]
grid_size <- args[2]
cores <- as.numeric(args[3])

plan("multisession", workers = cores)

outdir <- here("results", "mwolfe6", "2025-09-27_run_3mer_models")

d <- read_tsv(here(outdir, "train_with_seqs.tsv"))
# Zscore the data
d <- d %>% group_by(scoreset) %>% mutate(zscore = (score - mean(score))/sd(score)) %>% ungroup()

# Fit XGboost model

## preprocessor
three_mer_recipe <- function(data){
    out <- recipes::recipe(zscore ~ seq + ensp + scoreset, data = data)
    out |>
        recipes::step_naomit(zscore)  |>
        textrecipes::step_tokenize(seq, token = "characters") |>
        textrecipes::step_ngram(seq, num_tokens = 3, min_num_tokens = 3) |>
        textrecipes::step_tfidf(seq) |>
        recipes::step_novel(scoreset) |>
        recipes::step_novel(ensp) |>
        recipes::step_dummy(recipes::all_nominal_predictors())
}

## model
reg_xgboost_spec <-
    parsnip::boost_tree(
        trees = 500,
        tree_depth = tune(),
        min_n = tune(),
        loss_reduction = tune(),
        sample_size = tune(), mtry = tune(),
        learn_rate = tune()
    ) |>
    parsnip::set_engine("xgboost") |>
    parsnip::set_mode("regression")

# Split into test and train
d_split <- d %>% rsample::initial_split()

# split training into folds
d_folds <- rsample::vfold_cv(rsample::training(d_split), v = 3)

# set up workflow
xgboost_wf <- workflow() %>% add_recipe(three_mer_recipe(training(d_split))) %>%
    add_model(reg_xgboost_spec)

# get possible params
xgboost_params <- extract_parameter_set_dials(xgboost_wf) %>% dials::finalize(training(d_split)) %>%
    grid_random(size = grid_size)
xgboost_params %>% write_tsv(here(outdir, str_c(outpre, "_sampled_params.tsv")))

# Tuning
message("Tuning...")
grid_ctrl <- tune::control_grid(
        save_pred = FALSE,
        parallel_over = "resamples",
        save_workflow = FALSE,
        verbose = TRUE)

my_metrics <- metric_set(rmse, rsq, mae)
set.seed(42)
xgboost_reg_tune <-
    xgboost_wf %>%
    tune_grid(
        d_folds,
        grid = xgboost_params,
        metrics = my_metrics,
        control = grid_ctrl
    )

# write out tuning performance metrics
xgboost_reg_tune %>% collect_metrics(summarize = FALSE) %>%
    write_tsv(here(outdir, str_c(outpre, "_cv_metrics.tsv")))

# get best parameters
best_params <- xgboost_reg_tune %>% select_best(metric = "rmse")
best_params %>% write_tsv(here(outdir, str_c(outpre, "_best_params.tsv")))
final_wf <- xgboost_wf %>% finalize_workflow(best_params)

# fit on full training set
message("fitting full training set with best params...")
final_fit <-
    final_wf %>%
    fit(training(d_split))

# get performance on training and test set

message("getting train performance...")
train_perf <- final_fit %>% augment(model, new_data = training(d_split)) %>%
    mutate(split = "train")

message("getting test performance...")
test_perf <- final_fit %>% augment(model, new_data = testing(d_split)) %>%
    mutate(split = "test")

message("Writing out final performance...")
bind_rows(train_perf, test_perf) %>%
    write_tsv(here(outdir, str_c(outpre, "_train_test_perf.tsv")))
