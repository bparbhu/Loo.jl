using LinearAlgebra, Statistics

function psis(log_ratios::Array, r_eff = nothing, cores = get(ENV, "mc.cores", 1))
    importance_sampling(log_ratios, r_eff = r_eff, cores = cores, method = "psis")
end


function psis(log_ratios::Matrix, r_eff = nothing, cores = get(ENV, "mc.cores", 1))
    importance_sampling(log_ratios, r_eff = r_eff, cores = cores, method = "psis")
end


function psis(log_ratios::Vector, r_eff = nothing)
    importance_sampling(log_ratios, r_eff = r_eff, method = "psis")
end


is_psis(x) = typeof(x) <: psis && typeof(x) <: Dict


function psis_object(unnormalized_log_weights, pareto_k, tail_len, r_eff)
    importance_sampling_object(unnormalized_log_weights = unnormalized_log_weights,
                               pareto_k = pareto_k,
                               tail_len = tail_len,
                               r_eff = r_eff,
                               method = "psis")
end


function do_psis(log_ratios, r_eff, cores, method)
    do_importance_sampling(log_ratios = log_ratios,
                           r_eff = r_eff,
                           cores = cores,
                           method = "psis")
end


function psis_apply(x::Vector, item, fun = ["[[", "attr"], fun_val = Float64[])
    if !isa(x, Vector)
        error("Internal error ('x' must be a Vector for psis_apply)")
    end
    vapply(x, FUN = fun, FUN.VALUE = fun_val, item)
end


function do_psis_i(log_ratios_i, tail_len_i)
    S = length(log_ratios_i)
    lw_i = log_ratios_i .- maximum(log_ratios_i)
    khat = Inf

    if enough_tail_samples(tail_len_i)
        ord = sortperm(lw_i)
        tail_ids = S - tail_len_i + 1 : S
        lw_tail = ord[tail_ids]
        if abs(maximum(lw_tail) - minimum(lw_tail)) < eps(Float64)/100
            @warn "Can't fit generalized Pareto distribution because all tail values are the same."
        else
            cutoff = ord[minimum(tail_ids) - 1]
            smoothed = psis_smooth_tail(lw_tail, cutoff)
            khat = smoothed["k"]
            lw_i[ord[tail_ids]] = smoothed["tail"]
        end
    end

    lw_i[lw_i .> 0] .= 0
    lw_i .+= maximum(log_ratios_i)

    return Dict("log_weights" => lw_i, "pareto_k" => khat)
end


function psis_smooth_tail(x, cutoff)
    len = length(x)
    exp_cutoff = exp(cutoff)

    fit = gpdfit(exp.(x) .- exp_cutoff, sort_x = false)
    k = fit["k"]
    sigma = fit["sigma"]
    if isfinite(k)
        p = (collect(1:len) .- 0.5) ./ len
        qq = qgpd(p, k, sigma) .+ exp_cutoff
        tail = log.(qq)
    else
        tail = x
    end
    return Dict("tail" => tail, "k" => k)
end


function n_pareto(r_eff, S)
    return ceil.(min.(0.2 .* S, 3 .* sqrt.(S ./ r_eff)))
end


function enough_tail_samples(tail_len, min_len = 5)
    return tail_len .>= min_len
end


function throw_pareto_warnings(k, high = 0.5, too_high = 0.7)
    if any(k .> too_high)
        @warn "Some Pareto k diagnostic values are too high. ", k_help()
    elseif any(k .> high)
        @warn "Some Pareto k diagnostic values are slightly high. ", k_help()
    end
end


function throw_tail_length_warnings(tail_lengths)
    tail_len_bad = .!map(enough_tail_samples, tail_lengths)
    if any(tail_len_bad)
        if length(tail_lengths) == 1
            @warn "Not enough tail samples to fit the generalized Pareto distribution."
        else
            bad = findall(tail_len_bad)
            Nbad = length(bad)
            @warn "Not enough tail samples to fit the generalized Pareto distribution in some or all columns of matrix of log importance ratios. Skipping the following columns: $(join(bad[1:min(Nbad, 10)], ", "))$(Nbad > 10 ? ", ... [$(Nbad - 10) more not printed]." : "")"
        end
    end
    return tail_lengths
end


function prepare_psis_r_eff(r_eff, len)
    if isnothing(r_eff) || all(ismissing, r_eff)
        if !called_from_loo() && isnothing(r_eff)
            throw_psis_r_eff_warning()
        end
        r_eff = ones(len)
    elseif length(r_eff) != len
        error("'r_eff' must have one value per observation.")
    elseif any(ismissing, r_eff)
        error("Can't mix missing and not missing values in 'r_eff'.")
    end
    return r_eff
end


function called_from_loo()
    calls = backtrace()
    txt = map(string, calls)
    patts = ["loo.array(", "loo.matrix(", "loo.function("]
    check = map(x -> any(occursin.(patts, x)), txt)
    return any(check)
end


function throw_psis_r_eff_warning()
    @warn "Relative effective sample sizes ('r_eff' argument) not specified. PSIS n_eff will not be adjusted based on MCMC n_eff."
end
