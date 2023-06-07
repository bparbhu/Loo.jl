using LinearAlgebra
using Statistics
using Distributions
using Distributed

function loo_moment_match_split(x, upars, cov, total_shift, total_scaling,
                     total_mapping, i, log_prob_upars,
                     log_lik_i_upars, r_eff_i, cores,
                     is_method, args...)

  S = size(upars, 1)
  S_half = convert(Int, 0.5 * S)
  mean_original = mean(upars, dims=1)

  # accumulated affine transformation
  upars_trans = (upars .- mean_original) .* total_scaling
  if cov
    upars_trans = upars_trans * total_mapping
  end
  upars_trans = upars_trans .+ total_shift .+ mean_original

  # inverse accumulated affine transformation
  upars_trans_inv = (upars .- mean_original)
  if cov
    upars_trans_inv = upars_trans_inv / total_mapping
  end
  upars_trans_inv = (upars_trans_inv ./ total_scaling) .+ mean_original .- total_shift

  # first half of upars_trans_half are T(theta)
  # second half are theta
  upars_trans_half = upars
  take = 1:S_half
  upars_trans_half[take, :] = upars_trans[take, :]

  # first half of upars_half_inv are theta
  # second half are T^-1 (theta)
  upars_trans_half_inv = upars
  take = S_half+1:S
  upars_trans_half_inv[take, :] = upars_trans_inv[take, :]

  # compute log likelihoods and log probabilities
  log_prob_half_trans = log_prob_upars(x, upars = upars_trans_half, args...)
  log_prob_half_trans_inv = log_prob_upars(x, upars = upars_trans_half_inv, args...)
  log_liki_half = log_lik_i_upars(x, upars = upars_trans_half, i = i, args...)

  # compute weights
  log_prob_half_trans_inv = log_prob_half_trans_inv - log(prod(total_scaling)) - log(det(total_mapping))
  stable_S = log_prob_half_trans .> log_prob_half_trans_inv

  lwi_half = -log_liki_half .+ log_prob_half_trans
  lwi_half[stable_S] = lwi_half[stable_S] - (log_prob_half_trans[stable_S] .+ log1pexp(log_prob_half_trans_inv[stable_S] .- log_prob_half_trans[stable_S]))

  lwi_half[.!stable_S] = lwi_half[.!stable_S] - (log_prob_half_trans_inv[.!stable_S] .+ log1pexp(log_prob_half_trans[.!stable_S] .- log_prob_half_trans_inv[.!stable_S]))

  is_obj_half = importance_sampling_default(lwi_half, method = is_method, r_eff = r_eff_i, cores = cores)
  lwi_half = weights(is_obj_half)

  is_obj_f_half = importance_sampling_default(lwi_half .+ log_liki_half, method = is_method, r_eff = r_eff_i, cores = cores)
  lwfi_half = weights(is_obj_f_half)

  # relative_eff recomputation
  # currently ignores chain information
  # since we have two proposal distributions
  # compute S_eff separately from both and take the smaller
  take = S_half+1:S
  log_liki_half_1 = log_liki_half[take]
  reshape(log_liki_half_1, length(take), 1, 1)
  take = 1:S_half
  log_liki_half_2 = log_liki_half[take]
  reshape(log_liki_half_2, length(take), 1, 1)
  r_eff_i1 = relative_eff(exp.(log_liki_half_1), cores = cores)
  r_eff_i2 = relative_eff(exp.(log_liki_half_2), cores = cores)
  r_eff_i = min(r_eff_i1,r_eff_i2)

  return Dict(
    "lwi" => lwi_half,
    "lwfi" => lwfi_half,
    "log_liki" => log_liki_half,
    "r_eff_i" => r_eff_i
  )
end
