using Distributions
using StatsBase
using StatsFuns
using SharedArrays
using Distributed


# Define the abstract type for ImportanceSampling
abstract type ImportanceSampling end

# Define the structure for ImportanceSampling
mutable struct IS <: ImportanceSampling
    log_weights::Array{Float64,2}
    pareto_k::Array{Float64,1}
    tail_len::Array{Int,1}
    r_eff::Array{Float64,1}
    method::String
    norm_const_log::Array{Float64,1}
    dims::Tuple{Int,Int}
end

# Define the function importance_sampling
function importance_sampling(log_ratios::Array{Float64,2}, method::String, r_eff::Array{Float64,1}, cores::Int)
    N = size(log_ratios, 2)
    S = size(log_ratios, 1)
    tail_len = n_pareto(r_eff, S) # n_pareto function needs to be defined

    if method == "psis"
        is_fun = do_psis_i # do_psis_i function needs to be defined
        throw_tail_length_warnings(tail_len) # throw_tail_length_warnings function needs to be defined
    elseif method == "tis"
        is_fun = do_tis_i # do_tis_i function needs to be defined
    elseif method == "sis"
        is_fun = do_sis_i # do_sis_i function needs to be defined
    else
        error("Incorrect IS method.")
    end

    lw_list = [is_fun(log_ratios[:, i], tail_len[i]) for i in 1:N]

    log_weights = hcat([lw["log_weights"] for lw in lw_list]...)
    pareto_k = [lw["pareto_k"] for lw in lw_list]

    throw_pareto_warnings(pareto_k) # throw_pareto_warnings function needs to be defined

    norm_const_log = logsumexp(log_weights, dims=1)

    return IS(log_weights, pareto_k, tail_len, r_eff, method, norm_const_log, (S, N))
end

# Define the function weights
function weights(object::IS, log::Bool=true, normalize::Bool=true)
    out = object.log_weights
    if normalize
        out = out .- object.norm_const_log
    end
    if !log
        out = exp.(out)
    end
    return out
end


# Define the function importance_sampling_object
function importance_sampling_object(unnormalized_log_weights::Array{Float64,2}, pareto_k::Array{Float64,1}, tail_len::Array{Int,1}, r_eff::Array{Float64,1}, method::String)
    methods = unique(method)
    if length(methods) == 1
        method = methods[1]
    end

    norm_const_log = logsumexp(unnormalized_log_weights, dims=1)

    # need normalized weights (not on log scale) for psis_n_eff
    w = weights(IS(unnormalized_log_weights, pareto_k, tail_len, r_eff, method, norm_const_log, size(unnormalized_log_weights)), normalize = true, log = false)
    n_eff = psis_n_eff(w, r_eff) # psis_n_eff function needs to be defined

    return IS(unnormalized_log_weights, pareto_k, tail_len, r_eff, method, norm_const_log, size(unnormalized_log_weights))
end

# Define the function do_importance_sampling
function do_importance_sampling(log_ratios::Array{Float64,2}, r_eff::Array{Float64,1}, cores::Int, method::String)
    N = size(log_ratios, 2)
    S = size(log_ratios, 1)
    tail_len = n_pareto(r_eff, S) # n_pareto function needs to be defined

    if method == "psis"
        is_fun = do_psis_i # do_psis_i function needs to be defined
        throw_tail_length_warnings(tail_len) # throw_tail_length_warnings function needs to be defined
    elseif method == "tis"
        is_fun = do_tis_i # do_tis_i function needs to be defined
    elseif method == "sis"
        is_fun = do_sis_i # do_sis_i function needs to be defined
    else
        error("Incorrect IS method.")
    end

    lw_list = [is_fun(log_ratios[:, i], tail_len[i]) for i in 1:N]

    log_weights = hcat([lw["log_weights"] for lw in lw_list]...)
    pareto_k = [lw["pareto_k"] for lw in lw_list]

    throw_pareto_warnings(pareto_k) # throw_pareto_warnings function needs to be defined

    return importance_sampling_object(log_weights, pareto_k, tail_len, r_eff, method)
end
