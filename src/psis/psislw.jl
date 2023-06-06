using Statistics
using Distributions
using Distributed
using StatsBase
using StatsFuns

function psislw(lw, wcp = 0.2, wtrunc = 3/4)
    MIN_CUTOFF = -700
    MIN_TAIL_LENGTH = 5

    function psis(lw_i)
        x = lw_i .- maximum(lw_i)
        cutoff = lw_cutpoint(x, wcp, MIN_CUTOFF)
        above_cut = x .> cutoff
        x_body = x[.!above_cut]
        x_tail = x[above_cut]

        tail_len = length(x_tail)
        if tail_len < MIN_TAIL_LENGTH || all(x_tail .== x_tail[1])
            if all(x_tail .== x_tail[1])
                @warn "All tail values are the same. Weights are truncated but not smoothed."
            elseif tail_len < MIN_TAIL_LENGTH
                @warn "Too few tail samples to fit generalized Pareto distribution. Weights are truncated but not smoothed."
            end
            x_new = x
            k = Inf
        else
            tail_ord = sortperm(x_tail)
            exp_cutoff = exp(cutoff)
            fit = gpdfit(exp.(x_tail) .- exp_cutoff, wip=false, min_grid_pts = 80)
            k = fit.k
            sigma = fit.sigma
            prb = (collect(1:tail_len) .- 0.5) ./ tail_len
            qq = quantile(GeneralizedPareto(k, sigma), prb) .+ exp_cutoff
            smoothed_tail = zeros(tail_len)
            smoothed_tail[tail_ord] = log.(qq)
            x_new = x
            x_new[.!above_cut] = x_body
            x_new[above_cut] = smoothed_tail
        end
        lw_new = lw_normalize(lw_truncate(x_new, wtrunc))
        return Dict("lw_new" => lw_new, "k" => k)
    end

    N = size(lw, 2)
    addprocs(N)  # Add worker processes
    @distributed for i in 1:N
        lw_i = lw[:, i]
        psis = psis(lw_i)
        return psis
    end

    pareto_k = [out[i]["k"] for i in 1:N]
    psislw_warnings(pareto_k)

    lw_smooth = Array{Float64}(undef, funval, N)
    for i in 1:N
        lw_smooth[:, i] = out[i]["lw_new"]
    end
    out = Dict("lw_smooth" => lw_smooth, "pareto_k" => pareto_k)
    return out
end


function lw_cutpoint(y, wcp, min_cut)
    if min_cut < log(floatmin(Float64))
        min_cut = -700
    end
    cp = quantile(y, 1 - wcp)
    return max(cp, min_cut)
end


function lw_truncate(y, wtrunc)
    if wtrunc == 0
        return y
    end
    logS = log(length(y))
    lwtrunc = wtrunc * logS - logS + logsumexp(y)
    y[y .> lwtrunc] .= lwtrunc
    return y
end


function lw_normalize(y)
    return y .- logsumexp(y)
end


function psislw_warnings(k)
    if any(k .> 0.7)
        @warn "Some Pareto k diagnostic values are too high."
    elseif any(k .> 0.5)
        @warn "Some Pareto k diagnostic values are slightly high."
    end
end
