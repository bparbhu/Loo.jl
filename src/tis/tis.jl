using Distributions
using Distributed

abstract type Tis end

struct TisArray <: Tis
    log_ratios::Array
end

struct TisMatrix <: Tis
    log_ratios::Matrix
end

struct TisDefault <: Tis
    log_ratios::Vector
end

function tis(log_ratios::Array, r_eff = nothing, cores = 1)
    return importance_sampling(TisArray(log_ratios), r_eff = r_eff, cores = cores, method = "tis")
end

function tis(log_ratios::Matrix, r_eff = nothing, cores = 1)
    return importance_sampling(TisMatrix(log_ratios), r_eff = r_eff, cores = cores, method = "tis")
end

function tis(log_ratios::Vector, r_eff = nothing)
    return importance_sampling(TisDefault(log_ratios), r_eff = r_eff, method = "tis")
end

function is_tis(x::Tis)
    return isa(x, Tis)
end

function do_tis_i(log_ratios_i::Vector)
    S = length(log_ratios_i)
    log_Z = logsumexp(log_ratios_i) - log(S) # Normalization term, c-hat in Ionides (2008) appendix
    log_cutpoint = log_Z + 0.5 * log(S)
    lw_i = min.(log_ratios_i, log_cutpoint)
    return Dict("log_weights" => lw_i, "pareto_k" => 0)
end
