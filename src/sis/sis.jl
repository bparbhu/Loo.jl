using Distributions
using Distributed

abstract type Sis end

struct SisArray <: Sis
    log_ratios::Array
end

struct SisMatrix <: Sis
    log_ratios::Matrix
end

struct SisDefault <: Sis
    log_ratios::Vector
end

function sis(log_ratios::Array, r_eff = nothing, cores = 1)
    return importance_sampling(SisArray(log_ratios), r_eff = r_eff, cores = cores, method = "sis")
end

function sis(log_ratios::Matrix, r_eff = nothing, cores = 1)
    return importance_sampling(SisMatrix(log_ratios), r_eff = r_eff, cores = cores, method = "sis")
end

function sis(log_ratios::Vector, r_eff = nothing)
    return importance_sampling(SisDefault(log_ratios), r_eff = r_eff, method = "sis")
end

function is_sis(x::Sis)
    return isa(x, Sis)
end

function do_sis_i(log_ratios_i::Vector)
    S = length(log_ratios_i)
    return Dict("log_weights" => log_ratios_i, "pareto_k" => 0)
end
