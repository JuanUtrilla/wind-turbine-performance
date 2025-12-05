generate_synthetic_wind_data <- function() {
  
  message(">> Starting Data Generation: 6000 Turbines with Fault Distribution...")
  
  # --- 1. General Parameters ---
  rated_power <- 2000   # kW (Vestas V90 equivalent)
  cut_in      <- 4      # m/s
  cut_out     <- 20     # m/s
  
  wb_shape    <- 2.0    # Weibull Shape
  wb_scale    <- 8.5    # Weibull Scale
  
  delta_ws    <- 0.5    # Tolerance for power curve bands
  n_turbines  <- 600    # Number of turbines
  
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
  n_mid  <- 30   # Medium severity
  n_high <- 150  # High severity
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
