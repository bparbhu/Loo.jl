using StatsBase
using Random


abstract type CRPS end

function crps(x::CRPS, args...)
    error("Not implemented")
end

function scrps(x::CRPS, args...)
    error("Not implemented")
end

function loo_crps(x::CRPS, args...)
    error("Not implemented")
end

function loo_scrps(x::CRPS, args...)
    error("Not implemented")
end

struct CRPSMatrix <: CRPS
    x::Matrix
    x2::Matrix
    y::Vector
    permutations::Int
end


function crps(x::CRPSMatrix)
    validate_crps_input(x.x, x.x2, x.y)
    repeats = [EXX_compute(x.x, x.x2) for _ in 1:x.permutations]
    EXX = sum(repeats) / x.permutations
    EXy = mean(abs.(x.x .- x.y))
    return crps_output(crps_fun(EXX, EXy))  # removed the dot operator
end


struct CRPSNumeric <: CRPS
    x::Vector
    x2::Vector
    y::Float64
    permutations::Int
end

function crps(x::CRPSNumeric)
    @assert length(x.x) == length(x.x2)
    @assert length(x.y) == 1
    return crps(CRPSMatrix(reshape(x.x, :, 1), reshape(x.x2, :, 1), [x.y], x.permutations))
end


function loo_crps(x::Matrix, x2::Matrix, y::Vector, log_lik::Vector, permutations::Int=1, r_eff::Vector=Vector{Float64}(), cores::Int=1)
    validate_crps_input(x, x2, y, log_lik)
    repeats = [EXX_loo_compute(x, x2, log_lik, r_eff = r_eff) for _ in 1:permutations]
    EXX = sum(repeats) / permutations
    psis_obj = psis(-log_lik, r_eff = r_eff, cores = cores)
    EXy = E_loo(abs.(x .- y), psis_obj, log_ratios = -log_lik).value
    return crps_output(crps_fun(EXX, EXy))
end

function scrps(x::Matrix, x2::Matrix, y::Vector, permutations::Int=1)
    validate_crps_input(x, x2, y)
    repeats = [EXX_compute(x, x2) for _ in 1:permutations]
    EXX = sum(repeats) / permutations
    EXy = mean(abs.(x .- y))
    return crps_output(crps_fun(EXX, EXy, scale = true))
end

function scrps(x::Vector, x2::Vector, y::Float64, permutations::Int=1)
    if length(x) != length(x2) || length(y) != 1
        throw(DimensionMismatch("Length of x and x2 must be equal and y must be a scalar"))
    end
    return scrps(reshape(x, :, 1), reshape(x2, :, 1), [y], permutations)
end

function loo_scrps(x::Matrix, x2::Matrix, y::Vector, log_lik::Vector, permutations::Int=1, r_eff::Vector=Vector{Float64}(), cores::Int=1)
    validate_crps_input(x, x2, y, log_lik)
    repeats = [EXX_loo_compute(x, x2, log_lik, r_eff = r_eff) for _ in 1:permutations]
    EXX = sum(repeats) / permutations
    psis_obj = psis(-log_lik, r_eff = r_eff, cores = cores)
    EXy = E_loo(abs.(x .- y), psis_obj, log_ratios = -log_lik).value
    return crps_output(crps_fun(EXX, EXy, scale = true))
end


function EXX_compute(x::Matrix, x2::Matrix)
    S = size(x, 1)
    return mean(abs.(x .- x2[randperm(S), :]), dims=1)
end

function EXX_loo_compute(x::Matrix, x2::Matrix, log_lik::Vector, r_eff::Vector=Vector{Float64}())
    S = size(x, 1)
    shuffle = randperm(S)
    x2 = x2[shuffle, :]
    log_lik2 = log_lik[shuffle]
    psis_obj_joint = psis(-log_lik .- log_lik2, r_eff = r_eff)
    return E_loo(abs.(x .- x2), psis_obj_joint, log_ratios = -log_lik .- log_lik2).value
end

function crps_fun(EXX::Vector, EXy::Vector, scale::Bool=false)
    if scale
        return -EXy ./ EXX .- 0.5 .* log.(EXX)
    end
    return 0.5 .* EXX .- EXy
end

function crps_output(crps_pw::Vector)
    n = length(crps_pw)
    out = Dict()
    out["estimates"] = [mean(crps_pw), std(crps_pw) / sqrt(n)]
    out["pointwise"] = crps_pw
    return out
end

function validate_crps_input(x::Matrix, x2::Matrix, y::Vector, log_lik::Vector=Vector{Float64}())
    if !(isa(x, Matrix) && isa(x2, Matrix) && isa(y, Vector) && size(x) == size(x2) && size(x, 2) == length(y) && (isempty(log_lik) || size(x) == size(log_lik)))
        throw(ArgumentError("Invalid input"))
    end
end
