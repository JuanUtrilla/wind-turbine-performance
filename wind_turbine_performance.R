# ==============================================================================
# PROJECT: Wind Turbine Yaw Misalignment Detection (End-to-End Demo)
# AUTHOR: Juan Peñas Utrilla
# DESCRIPTION: 
#   This script simulates realistic wind turbine SCADA data with induced 
#   yaw misalignment faults and trains an XGBoost model to detect them.
#   
#   It consists of two main parts:
#   1. Data Simulation (Synthetic Data Generation)
#   2. Machine Learning Pipeline (Feature Eng., Aggregation, XGBoost)
# ==============================================================================

# ------------------------------------------------------------------------------
# 00 - SETUP AND LIBRARIES
# ------------------------------------------------------------------------------
library(tidyverse)
library(xgboost)
library(Matrix)
library(pROC)
library(moments) # For skewness and kurtosis
library(caret)   # General ML utils

set.seed(123) # Ensure reproducibility

# ------------------------------------------------------------------------------
# 01 - DATA SIMULATION MODULE
# ------------------------------------------------------------------------------
# Function to generate realistic SCADA data with physical constraints and faults

generate_synthetic_wind_data <- function() {
  
  message(">> Starting Data Generation: 6000 Turbines with Fault Distribution...")
  
  # --- 1. General Parameters ---
  rated_power <- 2000   # kW (Vestas V90 equivalent)
  cut_in      <- 4      # m/s
  cut_out     <- 20     # m/s
  
  wb_shape    <- 2.0    # Weibull Shape
  wb_scale    <- 8.5    # Weibull Scale
  
  delta_ws    <- 0.5    # Tolerance for power curve bands
  n_turbines  <- 6000    # Number of turbines
  
  # --- 2. Theoretical Power Curve Definition ---
  ws_points <- c(0, 3, 3.5, 4, 5,  6,   7,   8,   9,   10,   11,   12,   13,   20, 25, 25.01)
  p_points  <- c(0, 0, 35, 80, 230, 480, 800, 1150, 1500, 1800, 1950, 1990, 2000, 2000, 2000, 0)
  
  power_curve_interp <- approxfun(ws_points, p_points, method = "linear", rule = 2)
  
  power_curve_calc <- function(ws) {
    p <- power_curve_interp(ws)
    p[ws > cut_out] <- 0
    return(p)
  }
  
  # --- 3. Temporal Structure ---
  start_time <- as.POSIXct("2025-12-01 00:00:00", tz = "UTC")
  step_sec <- 10 * 60             # 10-minute intervals
  n_points_per_turbine <- 7 * 24 * 6 # 1 week of data
  
  timestamps <- start_time + seq(0, by = step_sec, length.out = n_points_per_turbine)
  turbines   <- paste0("WT", sprintf("%03d", 1:n_turbines)) 
  
  # --- 4. Wind Speed Simulation (ARIMA + Spatial Correlation) ---
  # Generating a base wind signal
  ar_sim <- arima.sim(model = list(ar = 0.99), n = n_points_per_turbine)
  u_series <- pnorm(scale(as.numeric(ar_sim)))
  ws_base_weibull <- qweibull(u_series, shape = wb_shape, scale = wb_scale)
  
  df <- expand.grid(timestamp = timestamps, turbine_id = turbines, stringsAsFactors = FALSE)
  df <- df[order(df$timestamp, df$turbine_id), ] 
  
  df$ws_base <- rep(ws_base_weibull, each = n_turbines)
  
  # Adding spatial offsets and turbulence
  set.seed(124)
  spatial_offsets <- rnorm(n_turbines, 0, 0.6) 
  names(spatial_offsets) <- turbines
  
  turbulence <- rnorm(nrow(df), 0, 0.4) 
  
  df$ws <- df$ws_base + spatial_offsets[df$turbine_id] + turbulence
  df$ws <- pmax(0, df$ws)
  df$ws[df$ws > cut_out] <- cut_out
  
  # --- 5. Fault Injection (Yaw Misalignment / Degradation) ---
  set.seed(999)
  
  # Defining Severity Groups
  n_mid  <- 300   # Medium severity
  n_high <- 850  # High severity
  n_low  <- n_turbines - n_mid - n_high # Healthy/Low offset
  
  offsets_low  <- runif(n_low,  min = 0.05, max = 0.35)
  offsets_mid  <- runif(n_mid,  min = 0.35, max = 0.55)
  offsets_high <- runif(n_high, min = 0.55, max = 0.75)
  
  all_offsets <- c(offsets_low, offsets_mid, offsets_high)
  offsets_vec <- sample(all_offsets) # Shuffle assignments
  names(offsets_vec) <- turbines
  
  # Apply offsets to data
  df$base_offset <- offsets_vec[df$turbine_id]
  
  # Stochastic variability (Simulating dynamic yaw error)
  set.seed(888)
  offset_wobble <- rnorm(nrow(df), mean = 1, sd = 0.2)
  
  df$applied_offset <- df$base_offset * offset_wobble
  df$applied_offset <- pmax(0, df$applied_offset) 
  
  # Effective wind speed (The wind the turbine actually "sees")
  df$ws_effective <- pmax(0, df$ws - df$applied_offset)
  
  # --- 6. Power Generation Calculation ---
  df$p_theoretical <- power_curve_calc(df$ws)            
  df$p_physics     <- power_curve_calc(df$ws_effective) 
  
  # Adding Noise (variability depends on wind speed)
  sd_noise_vec <- rep(15, nrow(df))
  in_ramp <- df$ws > 4 & df$ws < 12
  sd_noise_vec[in_ramp] <- 60  
  
  noise <- rnorm(nrow(df), mean = 0, sd = sd_noise_vec)
  df$active_power <- df$p_physics + noise
  
  # Anomalies & Curtailment Injection
  # 1. Point Anomaly on WT050
  is_wt50 <- df$turbine_id == "WT050"
  df$active_power[is_wt50] <- df$active_power[is_wt50] + rnorm(sum(is_wt50), 0, 150)
  
  # 2. Curtailment Event
  limit_start <- as.POSIXct("2025-01-04 14:00:00", tz="UTC")
  limit_end   <- as.POSIXct("2025-01-04 23:00:00", tz="UTC")
  is_curtailed <- df$timestamp >= limit_start & df$timestamp <= limit_end
  
  df$active_power[is_curtailed] <- pmin(df$active_power[is_curtailed], 
                                        1100 + rnorm(sum(is_curtailed), 0, 25))
  
  # Final Cleanup
  df$active_power <- pmax(0, pmin(df$active_power, rated_power))
  
  # --- 7. Status & Bands ---
  df$p_lower_band <- power_curve_calc(pmax(df$ws - delta_ws, 0))
  df$p_upper_band <- power_curve_calc(df$ws + delta_ws)
  
  return(list(data = df, offsets = offsets_vec))
}

# ------------------------------------------------------------------------------
# 02 - HELPER FUNCTIONS
# ------------------------------------------------------------------------------

# Robust statistical functions (Handle NAs and empty vectors gracefully)
safe_min    <- function(x){ x <- x[!is.na(x)]; if (length(x)==0) NA_real_ else min(x) }
safe_max    <- function(x){ x <- x[!is.na(x)]; if (length(x)==0) NA_real_ else max(x) }
safe_mean   <- function(x){ x <- x[!is.na(x)]; if (length(x)==0) NA_real_ else mean(x) }
safe_median <- function(x){ x <- x[!is.na(x)]; if (length(x)==0) NA_real_ else median(x) }
safe_sd     <- function(x){ x <- x[!is.na(x)]; if (length(x)<=1) 0 else stats::sd(x) }

safe_cv <- function(x){
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(0)
  m <- mean(x); s <- sd(x)
  if (!is.finite(m) || abs(m) < 1e-9) return(NA_real_)
  s/m
}

safe_skew <- function(x){
  x <- x[is.finite(x)]
  if (length(x) < 3 || sd(x)==0) return(NA_real_)
  moments::skewness(x)
}

safe_kurt <- function(x){
  x <- x[is.finite(x)]
  if (length(x) < 4 || sd(x)==0) return(NA_real_)
  moments::kurtosis(x)
}

# Clean Infinite and NaN values
clean_inf_nan <- function(df){
  df %>% mutate(across(where(is.numeric),
                       ~ dplyr::case_when(
                         is.nan(.x)       ~ NA_real_,
                         is.infinite(.x)  ~ NA_real_,
                         TRUE             ~ .x
                       )))
}

# Imputation by Wind Farm (Fallback to global median)
impute_by_farm <- function(df){
  df %>%
    group_by(farm_name) %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x))) %>%
    ungroup() %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)))
}

# Aggregation function list
agg_fun <- list(
  mean = safe_mean, sd = safe_sd, cv = safe_cv,
  min = safe_min, max = safe_max, median = safe_median,
  skew = safe_skew, kurt = safe_kurt
)

# ------------------------------------------------------------------------------
# 03 - MAIN EXECUTION: DATA GENERATION
# ------------------------------------------------------------------------------

# Generate the data
sim_results <- generate_synthetic_wind_data()
raw_data    <- sim_results$data

# Define the Target Variable (Ground Truth)
# We define "Misaligned" if the average applied offset is >= 0.50 degrees/units
data_labeled <- raw_data %>%
  group_by(turbine_id) %>%
  mutate(is_misaligned = as.integer(mean(applied_offset, na.rm = TRUE) >= 0.50)) %>%
  ungroup()

data_labeled$farm_name <- "WindFarm_01_Madrid" 

# ------------------------------------------------------------------------------
# 04 - FEATURE ENGINEERING & CLEANING (ROW LEVEL)
# ------------------------------------------------------------------------------

max_pot <- max(data_labeled$active_power, na.rm = TRUE)

df_clean <- data_labeled %>%
  # Quality Filter: Remove low power or unrealistic outliers
  filter(
    active_power >= 0.1,
    active_power < max_pot * 0.95
  ) %>%
  # Feature Creation: Calculate deviations from theoretical curves
  mutate(
    bin_vel       = floor(ws / 0.5) * 0.5,
    delta_theo    = (active_power - p_theoretical),
    delta_low     = (active_power - p_lower_band),
    delta_high    = (active_power - p_upper_band),
    band_width    = (p_upper_band - p_lower_band),
    delta_theo_low= (p_theoretical - p_lower_band),
    delta_up_theo = (p_upper_band - p_theoretical)
  ) %>%
  # Select relevant columns for aggregation
  select(farm_name, turbine_id, ws, bin_vel, active_power, p_theoretical,
         delta_theo, delta_low, delta_high, band_width,delta_theo_low,
         delta_up_theo,is_misaligned)

# Handle NAs in target (just in case)
df_clean$is_misaligned[is.na(df_clean$is_misaligned)] <- 0

# ------------------------------------------------------------------------------
# 05 - TRAIN / TEST SPLIT (BY TURBINE ID)
# ------------------------------------------------------------------------------
# Crucial: Split by Turbine ID to prevent data leakage.
# The model must predict on NEW turbines it hasn't seen before.

ids_turbines <- unique(df_clean$turbine_id)
train_ids    <- sample(ids_turbines, size = floor(0.8 * length(ids_turbines)))
test_ids     <- setdiff(ids_turbines, train_ids)

df_train_rows <- df_clean %>% filter(turbine_id %in% train_ids)
df_test_rows  <- df_clean %>% filter(turbine_id %in% test_ids)

# ------------------------------------------------------------------------------
# 06 - AGGREGATION (TURBINE LEVEL PROFILE)
# ------------------------------------------------------------------------------
# Convert time-series data into a single row of statistics per turbine

cols_exclude <- c("is_misaligned", "bin_vel", "p_theoretical", "ws", "turbine_id", "farm_name")

aggregate_features <- function(df_input) {
  df_input %>%
    group_by(farm_name, turbine_id, is_misaligned) %>%
    summarise(
      across(
        .cols = where(is.numeric) & !any_of(cols_exclude),
        .fns  = agg_fun,
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    ) %>%
    clean_inf_nan() %>%
    impute_by_farm() %>%
    na.omit()
}

train_aggregated <- aggregate_features(df_train_rows)
test_aggregated  <- aggregate_features(df_test_rows)

message(paste("Training Set Size:", nrow(train_aggregated), "turbines"))
message(paste("Test Set Size:", nrow(test_aggregated), "turbines"))

# ------------------------------------------------------------------------------
# 07 - MODELING (XGBOOST)
# ------------------------------------------------------------------------------

# Prepare DMatrices
# Remove identifiers (Farm, ID) from the feature set
x_train_mtx <- model.matrix(is_misaligned ~ . -1, data = select(train_aggregated, -farm_name, -turbine_id))
x_test_mtx  <- model.matrix(is_misaligned ~ . -1, data = select(test_aggregated, -farm_name, -turbine_id))

y_train <- train_aggregated$is_misaligned
y_test  <- test_aggregated$is_misaligned

dtrain <- xgb.DMatrix(data = x_train_mtx, label = y_train)
dtest  <- xgb.DMatrix(data = x_test_mtx, label = y_test)

# Hyperparameters
params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,  # Lower learning rate for better generalization
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8
)

# --- Cross Validation (Stability Check) ---
message(">> Running Cross-Validation (5-Folds)...")

cv_results <- xgb.cv(
  params  = params,
  data    = dtrain,
  nrounds = 200,
  nfold   = 5,
  showsd  = TRUE,
  stratified = TRUE,
  print_every_n = 50,
  early_stopping_rounds = 20,
  maximize = TRUE
)

print(cv_results$evaluation_log[cv_results$best_iteration])

# --- Final Training ---
message(">> Training Final Model...")
final_model <- xgb.train(
  params    = params,
  data      = dtrain,
  nrounds   = 200,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20,
  print_every_n = 50
)

# ------------------------------------------------------------------------------
# 08 - EVALUATION
# ------------------------------------------------------------------------------

pred_prob  <- predict(final_model, dtest)
pred_class <- ifelse(pred_prob >= 0.5, 1, 0)

message("\n--- Confusion Matrix ---")
print(table(Predicted = pred_class, Actual = y_test))

if(length(unique(y_test)) > 1) {
  auc_score <- auc(y_test, pred_prob)
  message(paste("\nFinal AUC Score:", round(auc_score, 4)))
  
  # Optional: Plot ROC Curve
  plot(roc(y_test, pred_prob), print.auc = TRUE, col = "blue", main = "ROC Curve - XGBoost")
}

message("Process Completed Successfully.")
