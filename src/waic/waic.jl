using Distributions
using StatsBase
using StatsFuns
using Distributed

abstract type Waic end

struct WaicArray <: Waic
    x::Array
end

struct WaicMatrix <: Waic
    x::Matrix
end

struct WaicFunction <: Waic
    x::Function
    data::Union{DataFrame, Matrix}
    draws::Any
end

function waic(x::Array)
    return waic(WaicArray(x))
end

function waic(x::Matrix)
    ll = validate_ll(x)
    lldim = size(ll)
    lpd = logmeanexp(ll, dims=1) # colLogMeanExps
    p_waic = var(ll, dims=1)
    elpd_waic = lpd - p_waic
    waic = -2 * elpd_waic
    pointwise = hcat(elpd_waic, p_waic, waic)

    throw_pwaic_warnings(pointwise[:, 2], digits = 1)
    return waic_object(pointwise, dims = lldim)
end

function waic(x::Function, data = nothing, draws = nothing)
    @assert (isa(data, DataFrame) || isa(data, Matrix)) && !isnothing(draws)

    llfun = validate_llfun(x)
    N = size(data, 1)
    S = length(vec(llfun(data_i = data[1, :], draws = draws)))
    waic_list = [begin
            ll_i = llfun(data_i = data[i, :], draws = draws)
            ll_i = vec(ll_i)
            lpd_i = logmeanexp(ll_i)
            p_waic_i = var(ll_i)
            elpd_waic_i = lpd_i - p_waic_i
            [elpd_waic_i, p_waic_i]
        end for i in 1:N]
    pointwise = vcat(waic_list)
    pointwise = hcat(pointwise, waic = -2 * pointwise[:, 1])

    throw_pwaic_warnings(pointwise[:, 2], digits = 1)
    return waic_object(pointwise, dims = (S, N))
end

function dim_waic(x::Waic)
    return getfield(x, :dims)
end

function is_waic(x::Waic)
    return isa(x, Waic) && is_loo(x)
end

function waic_object(pointwise, dims)
    estimates = table_of_estimates(pointwise)
    out = Dict("estimates" => estimates, "pointwise" => pointwise)
    old_nms = ["elpd_waic", "p_waic", "waic", "se_elpd_waic", "se_p_waic", "se_waic"]
    out = merge(out, Dict(zip(old_nms, vec(estimates))))
    return Waic(out, dims)
end

function throw_pwaic_warnings(p, digits = 1, warn = true)
    badp = p .> 0.4
    if any(badp)
        count = sum(badp)
        prop = count / length(badp)
        msg = "\n$(count) (" * "$(round(100 * prop, digits = digits))%) p_waic estimates greater than 0.4. We recommend trying loo instead."
        if warn
            @warn msg
        else
            println(msg)
        end
    end
    return nothing
end
