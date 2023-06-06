using DataFrames
using Statistics
using LinearAlgebra


function psis_approximate_posterior(log_p = nothing, log_g = nothing, log_liks = nothing,
                                    cores, save_psis; log_q = nothing)
    if !isnothing(log_q)
        @warn "psis_approximate_posterior() argument log_q has been changed to log_g"
        log_g = log_q
    end
    @assert !isnothing(log_p) && all(!ismissing, log_p) && length(log_p) == length(log_g)
    @assert !isnothing(log_g) && all(!ismissing, log_g) && length(log_g) == length(log_p)
    @assert isnothing(log_liks) || (size(log_liks, 1) == length(log_p))
    @assert isa(cores, Integer)
    @assert isa(save_psis, Bool)

    if isnothing(log_liks)
        approx_correction = log_p .- log_g
        approx_correction = approx_correction .- maximum(approx_correction)
        log_ratios = reshape(approx_correction, :, 1)
    else
        log_ratios = correct_log_ratios(log_ratios = -log_liks, log_p = log_p, log_g = log_g)
    end
    psis_out = psis_matrix(log_ratios, cores = cores, r_eff = ones(ncol(log_ratios)))

    if isnothing(log_liks)
        return psis_out
    end

    pointwise = pointwise_loo_calcs(log_liks, psis_out)
    importance_sampling_loo_object(
        pointwise = pointwise,
        diagnostics = psis_out["diagnostics"],
        dims = size(psis_out),
        is_method = "psis",
        is_object = save_psis ? psis_out : nothing
    )
end


function correct_log_ratios(log_ratios, log_p, log_g)
    approx_correction = log_p .- log_g
    log_ratios = log_ratios .+ approx_correction
    log_ratio_max = maximum(log_ratios, dims=2)
    log_ratios = log_ratios .- log_ratio_max
    return log_ratios
end

function ap_psis(log_ratios, log_p, log_g)
    # The function dispatch will be handled by Julia's multiple dispatch system
    # based on the type of `log_ratios`
end

function ap_psis(log_ratios::Array{Float64,3}, log_p, log_g)
    # cores = loo_cores(cores)  # Not sure what this does in your R code
    @assert ndims(log_ratios) == 3
    # log_ratios = validate_ll(log_ratios)  # Not sure what this does in your R code
    log_ratios = reshape(log_ratios, :, size(log_ratios, 3))
    r_eff = ones(size(log_ratios, 2))
    return ap_psis(log_ratios, log_p, log_g)
end

function ap_psis(log_ratios::Matrix{Float64}, log_p, log_g)
    @assert length(log_p) == size(log_ratios, 1)
    @assert length(log_g) == size(log_ratios, 1)
    cores = loo_cores(cores)  # Not sure what this does in your R code
    log_ratios = validate_ll(log_ratios)  # Not sure what this does in your R code
    log_ratios = correct_log_ratios(log_ratios, log_p, log_g)
    return do_psis(log_ratios, r_eff = ones(size(log_ratios, 2)), cores = cores)
end


function ap_psis(log_ratios::Vector{Float64}, log_p, log_g)
    log_ratios = reshape(log_ratios, length(log_ratios), 1)
    @warn "llfun values do not return a matrix, coerce to matrix"
    return ap_psis(log_ratios, log_p, log_g)
end
