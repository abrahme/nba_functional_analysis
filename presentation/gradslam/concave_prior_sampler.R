library(shiny)
library(ggplot2)
library(dplyr)
library(tibble)

safe_cos_term <- function(divisor, x, L, eps = 1e-5) {
  divisor_safe <- ifelse(abs(divisor) < eps, eps, divisor)
  z <- divisor_safe * (x + L)
  denom <- 2 * L * (divisor_safe ^ 2)
  numer <- 1 - cos(z)

  base <- numer / denom
  limit <- ((x + L) ^ 2) / (4 * L)
  blend_weight <- exp(-((z / eps) ^ 2))

  blend_weight * limit + (1 - blend_weight) * base
}

safe_sin_term <- function(divisor, x, L, eps = 1e-5) {
  divisor_safe <- ifelse(abs(divisor) < eps, eps, divisor)
  z <- divisor_safe * (x + L)
  denom <- 2 * L * divisor_safe
  numer <- sin(z)

  base <- numer / denom
  limit <- (x + L) / (2 * L)
  blend_weight <- exp(-((z / eps) ^ 2))

  blend_weight * limit + (1 - blend_weight) * base
}

make_convex_phi <- function(x, L, M = 5, eps = 1e-5) {
  eig_vals <- (pi * (1:M)) / (2 * L)
  sum_eig_vals <- outer(eig_vals, eig_vals, "+")
  diff_eig_vals <- outer(eig_vals, eig_vals, "-")

  if (length(x) == 1) {
    cos_pos <- safe_cos_term(sum_eig_vals, x, L, eps)
    cos_neg <- safe_cos_term(diff_eig_vals, x, L, eps)
    return(cos_neg - cos_pos)
  }

  phi <- array(NA_real_, dim = c(length(x), M, M))
  for (idx in seq_along(x)) {
    cos_pos <- safe_cos_term(sum_eig_vals, x[idx], L, eps)
    cos_neg <- safe_cos_term(diff_eig_vals, x[idx], L, eps)
    phi[idx, , ] <- cos_neg - cos_pos
  }
  phi
}

make_convex_phi_prime <- function(x, L, M = 5, eps = 1e-5) {
  eig_vals <- (pi * (1:M)) / (2 * L)
  sum_eig_vals <- outer(eig_vals, eig_vals, "+")
  diff_eig_vals <- outer(eig_vals, eig_vals, "-")

  if (length(x) == 1) {
    sin_pos <- safe_sin_term(sum_eig_vals, x, L, eps)
    sin_neg <- safe_sin_term(diff_eig_vals, x, L, eps)
    return(sin_neg - sin_pos)
  }

  phi_prime <- array(NA_real_, dim = c(length(x), M, M))
  for (idx in seq_along(x)) {
    sin_pos <- safe_sin_term(sum_eig_vals, x[idx], L, eps)
    sin_neg <- safe_sin_term(diff_eig_vals, x[idx], L, eps)
    phi_prime[idx, , ] <- sin_neg - sin_pos
  }
  phi_prime
}

compute_convex_mu <- function(alpha_tilde,
                              phi_time,
                              shifted_x_time,
                              L_time,
                              t_max,
                              c_max,
                              M,
                              eps = 1e-5) {
  n <- nrow(alpha_tilde)
  t_len <- length(shifted_x_time)
  mu <- matrix(NA_real_, nrow = n, ncol = t_len)

  for (i in seq_len(n)) {
    phi_t_max_i <- make_convex_phi(t_max[i], L = L_time, M = M, eps = eps)
    phi_prime_t_max_i <- make_convex_phi_prime(t_max[i], L = L_time, M = M, eps = eps)

    mu[i, ] <- vapply(
      seq_len(t_len),
      function(t_idx) {
        core_t <- phi_t_max_i - phi_time[t_idx, , ] +
          phi_prime_t_max_i * (shifted_x_time[t_idx] - t_max[i])

        a_i <- alpha_tilde[i, , drop = FALSE]
        c_max[i] + sum((a_i %*% core_t) * a_i)
      },
      numeric(1)
    )
  }

  mu
}

sample_concave_prior <- function(lengthscale,
                                 variance,
                                 peak_age,
                                 peak_value,
                                 peak_age_variance,
                                 peak_value_variance,
                                 n_samples,
                                 num_basis,
                                 age_grid = seq(18, 38, by = .1)) {
  age_grid_shifted <- age_grid - mean(age_grid)
  L <- 2 * max(abs(age_grid_shifted))
  
  eigenvalues <- (pi * (1:num_basis) / (2 * L)) ^ 2
  spectral_density <- variance * exp(-0.5 * (sqrt(eigenvalues) / lengthscale) ^ 2)
  alpha <- matrix(rnorm(n_samples * num_basis), ncol = num_basis, nrow = n_samples)

  concave_phi <- make_convex_phi(age_grid_shifted, L = L, M = num_basis)
  alpha_tilde <- sweep(alpha, 2, sqrt(spectral_density), FUN = "*")

  t_max_raw <- rnorm(n_samples, mean = peak_age, sd = sqrt(peak_age_variance))
  t_max_raw <- pmin(pmax(t_max_raw, min(age_grid)), max(age_grid))
  t_max_shifted <- t_max_raw - mean(age_grid)

  c_max <- rnorm(n_samples, mean = peak_value, sd = sqrt(peak_value_variance))

  convex_mu_samples <- compute_convex_mu(
    alpha_tilde = alpha_tilde,
    phi_time = concave_phi,
    shifted_x_time = age_grid_shifted,
    L_time = L,
    t_max = t_max_shifted,
    c_max = c_max,
    M = num_basis
  )

  expand.grid(sample_id = seq_len(n_samples), age_idx = seq_along(age_grid)) |>
    as_tibble() |>
    mutate(
      age = age_grid[age_idx],
      value = convex_mu_samples[cbind(sample_id, age_idx)]
    ) |>
    select(sample_id, age, value)
}
