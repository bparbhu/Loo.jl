using DataFrames
using LinearAlgebra
using Statistics
using Random
using Distributed
using StatsBase
using LinearAlgebra
using StatsFuns

function loo_subsample(x, data = nothing,draws = nothing,observations = 400,
    log_p = nothing,log_g = nothing,r_eff = nothing,save_psis = false,cores = Distributed.nprocs(),
    loo_approximation = "plpd",loo_approximation_draws = nothing,estimator = "diff_srs",llgrad = nothing,llhess = nothing)
    
    cores = min(cores, Distributed.nprocs())
    
    llfun = validate_llfun(x)
    
    if isnothing(data) || isnothing(draws)
        error("Data and draws cannot be nothing")
    end
    
    if size(data, 1) != length(draws)
        error("The number of rows in data and length of draws must be the same")
    end
    
    if isnothing(log_p) && isnothing(log_g)
        if isnothing(r_eff)
            throw_loo_r_eff_warning()
        else
            r_eff = prepare_psis_r_eff(r_eff, len = size(data, 1))
        end
    end
    
    if length(observations) == 1
        idxs = subsample_idxs(estimator, elpd_loo_approx, observations)
    else
        idxs = compute_idxs(observations)
    end
    
    data_subsample = data[idxs.idx, :]
    
    if length(r_eff) > 1
        r_eff = r_eff[idxs.idx]
    end
    
    if !isnothing(log_p) && !isnothing(log_g)
        loo_obj = loo_approximate_posterior_function(x = llfun,
                                                     data = data_subsample,
                                                     draws = draws,
                                                     log_p = log_p,
                                                     log_g = log_g,
                                                     save_psis = save_psis,
                                                     cores = cores)
    else
        loo_obj = loo_function(x = llfun,
                               data = data_subsample,
                               draws = draws,
                               r_eff = r_eff,
                               save_psis = save_psis,
                               cores = cores)
    end
    
    loo_ss = psis_loo_ss_object(x = loo_obj,
                                idxs = idxs,
                                elpd_loo_approx = elpd_loo_approx,
                                loo_approximation = loo_approximation,
                                loo_approximation_draws = loo_approximation_draws,
                                estimator = estimator,
                                llfun = llfun,
                                llgrad = llgrad,
                                llhess = llhess,
                                data_dim = size(data),
                                ndraws = length(draws))
    
    return loo_ss
end


function update_psis_loo_ss(object;
                            data = nothing,
                            draws = nothing,
                            observations = nothing,
                            r_eff = nothing,
                            cores = Sys.CPU_THREADS,
                            loo_approximation = nothing,
                            loo_approximation_draws = nothing,
                            llgrad = nothing,
                            llhess = nothing)

    # Fallback
    if isnothing(observations) &&
       isnothing(loo_approximation) &&
       isnothing(loo_approximation_draws) &&
       isnothing(llgrad) &&
       isnothing(llhess)
        return object
    end

    if !isnothing(data)
        @assert isa(data, DataFrame) || isa(data, Matrix)
        @assert all(size(data) .== object.loo_subsampling.data_dim)
    end
    cores = min(cores, Sys.CPU_THREADS)

    # Update elpd approximations
    if !isnothing(loo_approximation) || !isnothing(loo_approximation_draws)
        @assert isa(data, DataFrame) || isa(data, Matrix) && !isnothing(draws)
        if object.loo_subsampling.estimator in ["hh_pps"]
            # HH estimation uses elpd_loo approx to sample,
            # so updating it will lead to incorrect results
            error("Can not update loo_approximation when using PPS sampling.")
        end
        if isnothing(loo_approximation) loo_approximation = object.loo_subsampling.loo_approximation end
        if isnothing(loo_approximation_draws) loo_approximation_draws = object.loo_subsampling.loo_approximation_draws end
        if isnothing(llgrad) llgrad = object.loo_subsampling.llgrad else llgrad = validate_llfun(llgrad) end
        if isnothing(llhess) llhess = object.loo_subsampling.llhess else llhess = validate_llfun(llhess) end

        # Compute loo approximation
        elpd_loo_approx = elpd_loo_approximation(llfun = object.loo_subsampling.llfun,
                                                 data = data, draws = draws,
                                                 cores = cores,
                                                 loo_approximation = loo_approximation,
                                                 loo_approximation_draws = loo_approximation_draws,
                                                 llgrad = llgrad, llhess = llhess)
        # Update object
        object.loo_subsampling.elpd_loo_approx = elpd_loo_approx
        object.loo_subsampling.loo_approximation = loo_approximation
        object.loo_subsampling.loo_approximation_draws = loo_approximation_draws
        object.loo_subsampling.llgrad = llgrad
        object.loo_subsampling.llhess = llhess
        object.pointwise[:, "elpd_loo_approx"] = object.loo_subsampling.elpd_loo_approx[object.pointwise[:, "idx"]]
    end

    # Update observations
    if !isnothing(observations)
        observations = assert_observations(observations,
                                           N = object.loo_subsampling.data_dim[1],
                                           object.loo_subsampling.estimator)
        if length(observations) == 1
            @assert isa(data, DataFrame) || isa(data, Matrix) && !isnothing(draws)
       
        # Compute subsample indices
        if length(observations) > 1
            idxs = compute_idxs(observations)
        else
            current_obs = nobs(object)

            # If sampling with replacement
            if object.loo_subsampling.estimator in ["hh_pps"]
                idxs = subsample_idxs(estimator = object.loo_subsampling.estimator,
                                      elpd_loo_approximation = object.loo_subsampling.elpd_loo_approx,
                                      observations = observations - current_obs)
            # If sampling without replacement
            elseif object.loo_subsampling.estimator in ["diff_srs", "srs"]
                current_idxs = obs_idx(object, rep = false)
                new_idx = setdiff(1:length(object.loo_subsampling.elpd_loo_approx), current_idxs)
                idxs = subsample_idxs(estimator = object.loo_subsampling.estimator,
                                      elpd_loo_approximation = object.loo_subsampling.elpd_loo_approx[setdiff(1:end, current_idxs)],
                                      observations = observations - current_obs)
                idxs.idx = new_idx[idxs.idx]
            end
        end

        # Identify how to update object
        cidxs = compare_idxs(idxs, object)

        # Compute new observations
        if !isnothing(cidxs.new)
            @assert isa(data, DataFrame) || isa(data, Matrix) && !isnothing(draws)
            data_new_subsample = data[cidxs.new.idx, :, drop = false]
            if length(r_eff) > 1 r_eff = r_eff[cidxs.new.idx] end

            if !isnothing(object.approximate_posterior.log_p) && !isnothing(object.approximate_posterior.log_g)
                loo_obj = loo_approximate_posterior_function(x = object.loo_subsampling.llfun,
                                                             data = data_new_subsample,
                                                             draws = draws,
                                                             log_p = object.approximate_posterior.log_p,
                                                             log_g = object.approximate_posterior.log_g,
                                                             save_psis = !isnothing(object.psis_object),
                                                             cores = cores)
            else
                loo_obj = loo_function(x = object.loo_subsampling.llfun,
                                       data = data_new_subsample,
                                       draws = draws,
                                       r_eff = r_eff,
                                       save_psis = !isnothing(object.psis_object),
                                       cores = cores)
            end
            # Add stuff to pointwise
            loo_obj.pointwise = add_subsampling_vars_to_pointwise(loo_obj.pointwise,
                                                                  cidxs.new,
                                                                  object.loo_subsampling.elpd_loo_approx)
        else
            loo_obj = nothing
        end

        if length(observations) == 1
            # Add new samples pointwise and diagnostic
            object = rbind_psis_loo_ss(object, x = loo_obj)

            # Update m_i for current pointwise (diagnostic stay the same)
            object.pointwise = update_m_i_in_pointwise(object.pointwise, cidxs.add, type = "add")
        else
            # Add new samples pointwise and diagnostic
            object = rbind_psis_loo_ss(object, loo_obj)

            # Replace m_i current pointwise and diagnostics
            object.pointwise = update_m_i_in_pointwise(object.pointwise, cidxs.add, type = "replace")

            # Remove samples
            object = remove_idx_psis_loo_ss(object, idxs = cidxs.remove)

            @assert setequal(obs_idx(object), observations)

            # Order object as in observations
            object = order_psis_loo_ss(object, observations)
        end
    end

    # Compute estimates
    if object.loo_subsampling.estimator == "hh_pps"
        object = loo_subsample_estimation_hh(object)
    elseif object.loo_subsampling.estimator == "diff_srs"
        object = loo_subsample_estimation_diff_srs(object)
    elseif object.loo_subsampling.estimator == "srs"
        object = loo_subsample_estimation_srs(object)
    else
        error("No correct estimator used.")
    end
    assert_psis_loo_ss(object)
    return object
    end
end


function obs_idx(x, rep = true)
    @assert typeof(x) == PSIS_LOO_SS
    if rep
        idxs = repeat(x.pointwise[:, "idx"], x.pointwise[:, "m_i"])
    else
        idxs = x.pointwise[:, "idx"]
    end
    return convert(Array{Int64,1}, idxs)
end


function nobs_psis_loo_ss(object)
    return sum(object.pointwise[:, "m_i"])
end


function loo_approximation_choices(api = true)
    lac = ["plpd", "lpd", "waic", "waic_grad_marginal", "waic_grad", "waic_hess", "tis", "sis", "none"]
    if !api
        push!(lac, "psis")
    end
    return lac
end


function estimator_choices()
    return ["hh_pps", "diff_srs", "srs"]
end


function lpd_i(i, llfun, data, draws)
    ll_i = llfun(data_i = data[i, :], draws = draws)
    ll_i = vec(ll_i)
    lpd_i = logmeanexp(ll_i)
    return lpd_i
end


@everywhere begin
    using StatsFuns: logmeanexp
end

function compute_lpds(N, data, draws, llfun)
    lpds = @distributed (vcat) for i in 1:N
        lpd_i(i, llfun, data, draws)
    end
    return lpds
end


function elpd_loo_approximation(llfun, data, draws, cores, loo_approximation, loo_approximation_draws = nothing, llgrad = nothing, llhess = nothing)
    
    cores = min(cores, Distributed.nprocs())
    N = size(data, 1)

    if loo_approximation == "none"
        return ones(Int, N)
    end

    if loo_approximation in ["tis", "sis"]
        draws = thin_draws(draws, loo_approximation_draws)
        is_values = loo_function(llfun, data = data, draws = draws, is_method = loo_approximation)
        return is_values["pointwise"]["elpd_loo"]
    end

    if loo_approximation == "waic"
        draws = thin_draws(draws, loo_approximation_draws)
        waic_full_obj = waic_function(llfun, data = data, draws = draws)
        return waic_full_obj["pointwise"]["elpd_waic"]
    end

    if loo_approximation == "lpd"
        draws = thin_draws(draws, loo_approximation_draws)
        lpds = compute_lpds(N, data, draws, llfun, cores)
        return lpds
    end

    if loo_approximation in ["plpd", "waic_grad", "waic_grad_marginal", "waic_hess"]
        draws = thin_draws(draws, loo_approximation_draws)
        point_est = compute_point_estimate(draws)
        lpds = compute_lpds(N, data, point_est, llfun, cores)
        if loo_approximation == "plpd"
            return lpds
        end
    end

    if loo_approximation in ["waic_grad", "waic_grad_marginal", "waic_hess"]
        if isnothing(llgrad)
            error("llgrad cannot be nothing")
        end

        point_est = compute_point_estimate(draws)
        lpds = compute_lpds(N, data, point_est, llfun, cores)

        if loo_approximation in ["waic_grad", "waic_hess"]
            cov_est = cov(draws)
        end

        if loo_approximation == "waic_grad_marginal"
            marg_vars = var(draws, dims = 1)
        end

        p_eff_approx = zeros(N)
        if cores > 1
            println("Multicore is not implemented for waic_delta")
        end

        if loo_approximation == "waic_grad"
            for i in 1:N
                grad_i = llgrad(data[i, :], point_est)
                local_cov = cov_est[grad_i, grad_i]
                p_eff_approx[i] = grad_i' * local_cov * grad_i
            end
        elseif loo_approximation == "waic_grad_marginal"
            for i in 1:N
                grad_i = llgrad(data[i, :], point_est)
                p_eff_approx[i] = sum(grad_i .* marg_vars[grad_i] .* grad_i)
            end
        elseif loo_approximation == "waic_hess"
            if isnothing(llhess)
                error("llhess cannot be nothing")
            end
            for i in 1:N
                grad_i = llgrad(data[i, :], point_est)
                hess_i = llhess(data[i, :], point_est)
                local_cov = cov_est[grad_i, grad_i]
                p_eff_approx[i] = grad_i' * local_cov * grad_i +
                                  0.5 * sum(diag(local_cov * hess_i * local_cov * hess_i))
            end
        else
            error(loo_approximation, " is not implemented!")
        end
        return lpds - p_eff_approx
    end
end


# Define a function to compute point estimate
function compute_point_estimate(draws)
    if isa(draws, Matrix)
        return vec(mean(draws, dims=1))
    else
        error("compute_point_estimate() has not been implemented for objects of type ", typeof(draws))
    end
end

# Define a function to thin draws
function thin_draws(draws, loo_approximation_draws)
    if isa(draws, Matrix)
        if isnothing(loo_approximation_draws)
            return draws
        end
        S = size(draws, 1)
        idx = 1:loo_approximation_draws:(S * loo_approximation_draws ÷ loo_approximation_draws)
        return draws[idx, :]
    else
        error("thin_draws() has not been implemented for objects of type ", typeof(draws))
    end
end

# Define a function to get the number of draws
function ndraws(x)
    if isa(x, Matrix)
        return size(x, 1)
    else
        error("ndraws() has not been implemented for objects of type ", typeof(x))
    end
end


function subsample_idxs(estimator, elpd_loo_approximation, observations)
    if estimator == "hh_pps"
        pi_values = pps_elpd_loo_approximation_to_pis(elpd_loo_approximation)
        idxs_df = pps_sample(observations, pis = pi_values)
    elseif estimator in ["diff_srs", "srs"]
        if observations > length(elpd_loo_approximation)
            error("'observations' is larger than the total sample size in 'data'.")
        end
        idx = 1:length(elpd_loo_approximation)
        idx_m = sample(idx, observations, replace=false)
        sort!(idx_m)
        idxs_df = DataFrame(idx=idx_m, m_i=ones(Int, observations))
    else
        error("Invalid estimator choice")
    end
    return idxs_df
end


function pps_elpd_loo_approximation_to_pis(elpd_loo_approximation)
    pi_values = abs.(elpd_loo_approximation)
    pi_values ./= sum(pi_values) # normalize to sum to 1
    return pi_values
end


function pps_sample(observations, pis)
    idx = sample(1:length(pis), observations, replace=true, weights(pis))
    return DataFrame(idx=idx, m_i=ones(Int, observations))
end


function compute_idxs(observations)
    tab = counts(observations)
    idxs_df = DataFrame(idx=collect(keys(tab)), m_i=collect(values(tab)))
    return idxs_df
end


function compare_idxs(idxs, object)
    current_idx = compute_idxs(obs_idx(object))
    result = Dict()

    new_idx = .!(idxs[:idx] .∈ current_idx[:idx])
    remove_idx = .!(current_idx[:idx] .∈ idxs[:idx])

    result["new"] = idxs[new_idx, :]
    if nrow(result["new"]) == 0
        delete!(result, "new")
    end

    result["add"] = idxs[.!new_idx, :]
    if nrow(result["add"]) == 0
        delete!(result, "add")
    end

    result["remove"] = current_idx[remove_idx, :]
    if nrow(result["remove"]) == 0
        delete!(result, "remove")
    end

    return result
end


function pps_sample(m, pis)
    @assert isa(m, Int)
    @assert all(0 .<= pis .<= 1)
    idx = sample(1:length(pis), Weights(pis), m, replace=true)
    idx_counts = counts(idx, 1:length(pis))
    idxs_df = DataFrame(idx = findall(!iszero, idx_counts), m_i = idx_counts[.!iszero(idx_counts)])
    return idxs_df
end


function psis_loo_ss_object(x, idxs, elpd_loo_approx, loo_approximation, loo_approximation_draws, estimator, llfun, llgrad, llhess, data_dim, ndraws)
  # Assertions
  @assert isa(x, psis_loo)
  assert_subsample_idxs(idxs)
  @assert isa(elpd_loo_approx, Number)
  @assert loo_approximation in loo_approximation_choices()
  @assert isa(loo_approximation_draws, Int)
  @assert estimator in estimator_choices()
  @assert isa(llfun, Function)
  @assert isa(llgrad, Function)
  @assert isa(llhess, Function)
  @assert isa(data_dim, Int)
  @assert ndraws > 0

  # Construct object
  x = convert(psis_loo_ss, x)
  x.pointwise = add_subsampling_vars_to_pointwise(x.pointwise, idxs, elpd_loo_approx)
  x.estimates = vcat(x.estimates, zeros(nrow(x.estimates)))
  names(x.estimates)[end] = "subsampling SE"

  x.loo_subsampling = Dict()
  x.loo_subsampling["elpd_loo_approx"] = elpd_loo_approx
  x.loo_subsampling["loo_approximation"] = loo_approximation
  x.loo_subsampling["loo_approximation_draws"] = loo_approximation_draws
  x.loo_subsampling["estimator"] = estimator
  x.loo_subsampling["llfun"] = llfun
  x.loo_subsampling["llgrad"] = llgrad
  x.loo_subsampling["llhess"] = llhess
  x.loo_subsampling["data_dim"] = data_dim
  x.loo_subsampling["ndraws"] = ndraws

  # Compute estimates
  if estimator == "hh_pps"
    x = loo_subsample_estimation_hh(x)
  elseif estimator == "diff_srs"
    x = loo_subsample_estimation_diff_srs(x)
  elseif estimator == "srs"
    x = loo_subsample_estimation_srs(x)
  else
    error("No correct estimator used.")
  end
  assert_psis_loo_ss(x)
  return x
end


abstract type psis_loo end
abstract type psis_loo_ss end


function as_psis_loo_ss(x::psis_loo_ss)
  return x
end


function as_psis_loo_ss(x::psis_loo)
  x = convert(psis_loo_ss, x)
  x.estimates = hcat(x.estimates, zeros(nrow(x.estimates)))
  names(x.estimates)[end] = "subsampling SE"
  x.pointwise = hcat(x.pointwise, reshape(1:nrow(x.pointwise), :, 1), ones(nrow(x.pointwise)), x.pointwise[:, "elpd_loo"])
  ncp = ncol(x.pointwise)
  names(x.pointwise)[(ncp-2):ncp] = ["idx", "m_i", "elpd_loo_approx"]
  x.loo_subsampling = Dict("elpd_loo_approx" => x.pointwise[:, "elpd_loo"], "loo_approximation" => "psis", "loo_approximation_draws" => nothing, "estimator" => "diff_srs", "data_dim" => [nrow(x.pointwise), missing], "ndraws" => missing)
  assert_psis_loo_ss(x)
  return x
end


function as_psis_loo(x::psis_loo)
  return x
end


function as_psis_loo(x::psis_loo_ss)
  if x.loo_subsampling["data_dim"][1] == nrow(x.pointwise)
    x.estimates = x.estimates[:, 1:2]
    x.pointwise = x.pointwise[:, 1:5]
    x.loo_subsampling = nothing
    loo_obj = importance_sampling_loo_object(pointwise = x.pointwise[:, 1:5], diagnostics = x.diagnostics, dims = getproperty(x, :dims), is_method = "psis", is_object = x.psis_object)
    if isa(x, psis_loo_ap)
      loo_obj.approximate_posterior = Dict("log_p" => x.approximate_posterior["log_p"], "log_g" => x.approximate_posterior["log_g"])
      loo_obj = convert(psis_loo_ap, loo_obj)
      assert_psis_loo_ap(loo_obj)
    end
  else
    error("A subsampling loo object can only be coerced to a loo object if all observations in data have been subsampled.")
  end
  return loo_obj
end


function add_subsampling_vars_to_pointwise(pointwise, idxs, elpd_loo_approx)
  pw = hcat(DataFrame(pointwise), idxs)
  pw.elpd_loo_approx = elpd_loo_approx[idxs.idx]
  pw = convert(Matrix, pw)
  assert_subsampling_pointwise(pw)
  return pw
end


function rbind_psis_loo_ss(object::psis_loo_ss, x::psis_loo)
  if isnothing(x)
    return object
  end
  assert_subsampling_pointwise(object.pointwise)
  assert_subsampling_pointwise(x.pointwise)
  @assert isempty(intersect(object.pointwise[:, "idx"], x.pointwise[:, "idx"]))
  object.pointwise = vcat(object.pointwise, x.pointwise)
  object.diagnostics.pareto_k = vcat(object.diagnostics.pareto_k, x.diagnostics.pareto_k)
  object.diagnostics.n_eff = vcat(object.diagnostics.n_eff, x.diagnostics.n_eff)
  getproperty(object, :dims)[2] = nrow(object.pointwise)
  return object
end


function remove_idx_psis_loo_ss(object::psis_loo_ss, idxs)
  if isnothing(idxs)
    return object
  end
  assert_subsample_idxs(idxs)

  row_map = DataFrame(row_no = 1:nrow(object.pointwise), idx = object.pointwise[:, "idx"])
  row_map = join(row_map, idxs, on = :idx, kind = :outer)

  object.pointwise = object.pointwise[setdiff(1:end, row_map.row_no), :]
  object.diagnostics.pareto_k = object.diagnostics.pareto_k[setdiff(1:end, row_map.row_no)]
  object.diagnostics.n_eff = object.diagnostics.n_eff[setdiff(1:end, row_map.row_no)]
  getproperty(object, :dims)[2] = nrow(object.pointwise)
  return object
end


function order_psis_loo_ss(x::psis_loo_ss, observations)
  if observations == obs_idx(x)
    return x
  end
  @assert Set(obs_idx(x)) == Set(observations)

  row_map_x = DataFrame(row_no_x = 1:nrow(x.pointwise), idx = x.pointwise[:, "idx"])
  row_map_obs = DataFrame(row_no_obs = 1:length(observations), idx = observations)
  row_map = join(row_map_obs, row_map_x, on = :idx, kind = :inner)

  x.pointwise = x.pointwise[row_map.row_no_x, :]
  x.diagnostics.pareto_k = x.diagnostics.pareto_k[row_map.row_no_x]
  x.diagnostics.n_eff = x.diagnostics.n_eff[row_map.row_no_x]
  return x
end


function update_m_i_in_pointwise(pointwise, idxs, type = "replace")
  if isnothing(idxs)
    return pointwise
  end
  assert_subsample_idxs(idxs)

  row_map = DataFrame(row_no = 1:nrow(pointwise), idx = pointwise[:, "idx"])
  row_map = join(row_map, idxs, on = :idx, kind = :outer)

  if type == "replace"
    pointwise[row_map.row_no, "m_i"] = row_map.m_i
  elseif type == "add"
    pointwise[row_map.row_no, "m_i"] += row_map.m_i
  end
  return pointwise
end


function loo_subsample_estimation_hh(x::psis_loo_ss)
  @assert typeof(x) == psis_loo_ss
  N = length(x.loo_subsampling.elpd_loo_approx)
  pis = pps_elpd_loo_approximation_to_pis(x.loo_subsampling.elpd_loo_approx)
  pis_sample = pis[x.pointwise[:, "idx"]]

  hh_elpd_loo = whhest(z = pis_sample, m_i = x.pointwise[:, "m_i"], y = x.pointwise[:, "elpd_loo"], N)
  srs_elpd_loo = srs_est(y = x.pointwise[:, "elpd_loo"], y_approx = pis_sample)
  x.estimates["elpd_loo", "Estimate"]  = hh_elpd_loo.y_hat_ppz
  if hh_elpd_loo.hat_v_y_ppz > 0
    x.estimates["elpd_loo", "SE"]  = sqrt(hh_elpd_loo.hat_v_y_ppz)
  else
    @warn "Negative estimate of SE, more subsampling obs. needed."
    x.estimates["elpd_loo", "SE"]  = NaN
  end
  x.estimates["elpd_loo", "subsampling SE"] = sqrt(hh_elpd_loo.v_hat_y_ppz)

  hh_p_loo = whhest(z = pis_sample, m_i = x.pointwise[:, "m_i"], y = x.pointwise[:, "p_loo"], N)
  x.estimates["p_loo", "Estimate"] = hh_p_loo.y_hat_ppz
  if hh_p_loo.hat_v_y_ppz > 0
    x.estimates["p_loo", "SE"]  = sqrt(hh_p_loo.hat_v_y_ppz)
  else
    @warn "Negative estimate of SE, more subsampling obs. needed."
    x.estimates["elpd_loo", "SE"]  = NaN
  end
  x.estimates["p_loo", "subsampling SE"] = sqrt(hh_p_loo.v_hat_y_ppz)
  return update_psis_loo_ss_estimates(x)
end


function update_psis_loo_ss_estimates(x::psis_loo_ss)
  @assert typeof(x) == psis_loo_ss

  x.estimates["looic", "Estimate"] = (-2) * x.estimates["elpd_loo", "Estimate"]
  x.estimates["looic", "SE"] = 2 * x.estimates["elpd_loo", "SE"]
  x.estimates["looic", "subsampling SE"] = 2 * x.estimates["elpd_loo", "subsampling SE"]

  x.elpd_loo = x.estimates["elpd_loo", "Estimate"]
  x.p_loo = x.estimates["p_loo", "Estimate"]
  x.looic = x.estimates["looic", "Estimate"]
  x.se_elpd_loo = x.estimates["elpd_loo", "SE"]
  x.se_p_loo = x.estimates["p_loo", "SE"]
  x.se_looic = x.estimates["looic", "SE"]

  return x
end

function whhest(z, m_i, y, N)
  @assert all(z .>= 0) && all(z .<= 1)
  @assert length(y) == length(z)
  @assert length(m_i) == length(z)
  est_list = Dict{String, Any}()
  est_list["m"] = sum(m_i)
  est_list["y_hat_ppz"] = sum(m_i .* (y ./ z)) / est_list["m"]
  est_list["v_hat_y_ppz"] = (sum(m_i .* ((y ./ z .- est_list["y_hat_ppz"]) .^ 2)) / est_list["m"]) / (est_list["m"] - 1)

  # See unbiadness proof in supplementary material to the article
  est_list["hat_v_y_ppz"] =
    (sum(m_i .* (y .^ 2 ./ z)) / est_list["m"]) +
    est_list["v_hat_y_ppz"] / N - est_list["y_hat_ppz"] ^ 2 / N
  return est_list
end


function loo_subsample_estimation_diff_srs(x::psis_loo_ss)
  @assert typeof(x) == psis_loo_ss

  elpd_loo_est = srs_diff_est(y_approx = x.loo_subsampling.elpd_loo_approx, y = x.pointwise[:, "elpd_loo"], y_idx = x.pointwise[:, "idx"])
  x.estimates["elpd_loo", "Estimate"] = elpd_loo_est["y_hat"]
  x.estimates["elpd_loo", "SE"] = sqrt(elpd_loo_est["hat_v_y"])
  x.estimates["elpd_loo", "subsampling SE"] = sqrt(elpd_loo_est["v_y_hat"])

  p_loo_est = srs_est(y = x.pointwise[:, "p_loo"], y_approx = x.loo_subsampling.elpd_loo_approx)
  x.estimates["p_loo", "Estimate"] = p_loo_est["y_hat"]
  x.estimates["p_loo", "SE"] = sqrt(p_loo_est["hat_v_y"])
  x.estimates["p_loo", "subsampling SE"] = sqrt(p_loo_est["v_y_hat"])

  return update_psis_loo_ss_estimates(x)
end


function srs_diff_est(y_approx, y, y_idx)
  @assert length(y) <= length(y_approx)
  @assert length(y_idx) == length(y)

  N = length(y_approx)
  m = length(y)
  y_approx_m = y_approx[y_idx]

  e_i = y .- y_approx_m
  t_pi_tilde = sum(y_approx)
  t_pi2_tilde = sum(y_approx .^ 2)
  t_e = N * mean(e_i)
  t_hat_epsilon = N * mean(y .^ 2 .- y_approx_m .^ 2)

  est_list = Dict{String, Any}()
  est_list["m"] = length(y)
  est_list["N"] = N
  est_list["y_hat"] = t_pi_tilde + t_e
  est_list["v_y_hat"] = N^2 * (1 - m / N) * var(e_i) / m
  est_list["hat_v_y"] = (t_pi2_tilde + t_hat_epsilon) - # a (has been checked)
    (1/N) * (t_e^2 - est_list["v_y_hat"] + 2 * t_pi_tilde * est_list["y_hat"] - t_pi_tilde^2) # b
  return est_list
end


function loo_subsample_estimation_srs(x::psis_loo_ss)
  @assert typeof(x) == psis_loo_ss

  elpd_loo_est = srs_est(y = x.pointwise[:, "elpd_loo"], y_approx = x.loo_subsampling.elpd_loo_approx)
  x.estimates["elpd_loo", "Estimate"] = elpd_loo_est["y_hat"]
  x.estimates["elpd_loo", "SE"] = sqrt(elpd_loo_est["hat_v_y"])
  x.estimates["elpd_loo", "subsampling SE"] = sqrt(elpd_loo_est["v_y_hat"])

  p_loo_est = srs_est(y = x.pointwise[:, "p_loo"], y_approx = x.loo_subsampling.elpd_loo_approx)
  x.estimates["p_loo", "Estimate"] = p_loo_est["y_hat"]
  x.estimates["p_loo", "SE"] = sqrt(p_loo_est["hat_v_y"])
  x.estimates["p_loo", "subsampling SE"] = sqrt(p_loo_est["v_y_hat"])

  return update_psis_loo_ss_estimates(x)
end

function srs_est(y, y_approx)
  @assert length(y) <= length(y_approx)
  N = length(y_approx)
  m = length(y)
  est_list = Dict{String, Any}()
  est_list["m"] = m
  est_list["y_hat"] = N * mean(y)
  est_list["v_y_hat"] = N^2 * (1 - m / N) * var(y) / m
  est_list["hat_v_y"] = N * var(y)
  return est_list
end


function assert_observations(x, N::Int, estimator)
  @assert typeof(N) == Int
  @assert estimator in estimator_choices()
  if isnothing(x)
    return x
  end
  if typeof(x) == psis_loo_ss
    x = obs_idx(x)
    @assert all(x .>= 1) && all(x .<= N) && !any(ismissing.(x))
    return x
  end
  x = convert(Array{Int}, x)
  if length(x) > 1
    @assert all(x .>= 1) && all(x .<= N) && !any(ismissing.(x))
    if estimator in ["hh_pps"]
      println("Sampling proportional to elpd approximation and with replacement assumed.")
    end
    if estimator in ["diff_srs", "srs"]
      println("Simple random sampling with replacement assumed.")
    end
  else
    @assert all(x .>= 1) && !any(ismissing.(x))
  end
  return x
end


function assert_subsample_idxs(x::DataFrame)
  @assert typeof(x) == DataFrame
  @assert ncol(x) == 2
  @assert all(col -> typeof(col) == Array{Int}, eachcol(x))
  @assert !any(ismissing.(x))
  @assert nrow(x) >= 1
  @assert names(x) == ["idx", "m_i"]
  @assert all(x.idx .>= 1) && !any(ismissing.(x.idx)) && length(unique(x.idx)) == length(x.idx)
  @assert all(x.m_i .>= 1) && !any(ismissing.(x.m_i))
  return x
end


function assert_psis_loo_ss(x)
  @assert typeof(x) == psis_loo_ss
  @assert all(name -> name in fieldnames(x), ["estimates", "pointwise", "diagnostics", "psis_object", "loo_subsampling"])
  @assert all(name -> name in rownames(x.estimates), ["elpd_loo", "p_loo", "looic"])
  @assert all(name -> name in colnames(x.estimates), ["Estimate", "SE", "subsampling SE"])
  assert_subsampling_pointwise(x.pointwise)
  @assert all(name -> name in fieldnames(x.loo_subsampling), ["elpd_loo_approx", "loo_approximation", "loo_approximation_draws", "estimator", "data_dim", "ndraws"])
  @assert all(!ismissing.(x.loo_subsampling.elpd_loo_approx)) && length(x.loo_subsampling.elpd_loo_approx) == x.loo_subsampling.data_dim[1]
  @assert x.loo_subsampling.loo_approximation in loo_approximation_choices(api = false)
  @assert isnothing(x.loo_subsampling.loo_approximation_draws) || typeof(x.loo_subsampling.loo_approximation_draws) == Int
  @assert x.loo_subsampling.estimator in estimator_choices()
  @assert all(ismissing.(x.loo_subsampling.data_dim)) && length(x.loo_subsampling.data_dim) == 2
  @assert !ismissing(x.loo_subsampling.data_dim[1])
  @assert ismissing(x.loo_subsampling.ndraws) || length(x.loo_subsampling.ndraws) == 1
  return x
end


function assert_subsampling_pointwise(x)
  @assert typeof(x) == Matrix
  @assert !any(ismissing.(x))
  @assert size(x, 2) == 8
  @assert all(name -> name in colnames(x), ["elpd_loo", "mcse_elpd_loo", "p_loo", "looic", "influence_pareto_k", "idx", "m_i", "elpd_loo_approx"])
  return x
end
