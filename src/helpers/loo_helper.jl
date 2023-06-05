using Statistics
using LinearAlgebra
using LogExpFunctions
using DataFrames
using StatsBase

function logMeanExp(x)
    logS = log(length(x))
    return logsumexp(x) - logS
end

function colLogMeanExps(x)
    logS = log(size(x, 1))
    return logsumexp(x, dims=1) .- logS
end


function table_of_estimates(x)
    out = DataFrame(
        Estimate = vec(sum(x, dims=1)),
        SE = vec(sqrt(size(x, 1) .* var(x, dims=1)))
    )
    return out
end


function logsumexp(x; dims=:)
    max_x = maximum(x, dims=dims)
    return max_x .+ log.(sum(exp.(x .- max_x), dims=dims))
end


function validate_ll(x)
    if isa(x, AbstractArray) && eltype(x) <: AbstractArray
        error("List not allowed as input.")
    elseif any(ismissing, x)
        error("Missing values not allowed in input.")
    elseif !all(isfinite, x)
        error("All input values must be finite.")
    end
    return nothing
end


function llarray_to_matrix(x)
    if !(isa(x, AbstractArray) && ndims(x) == 3)
        error("Input must be a 3D array.")
    end
    xdim = size(x)
    return reshape(x, prod(xdim[1:2]), xdim[3])
end


function llmatrix_to_array(x, chain_id)
    if !(isa(x, AbstractMatrix))
        error("Input must be a matrix.")
    end

    if any(x -> !(x isa Integer), chain_id)
        error("All chain_id values must be integers.")
    end

    lldim = size(x)
    n_chain = length(unique(chain_id))
    chain_counts = counts(chain_id)

    if length(chain_id) != lldim[1]
        error("Number of rows in matrix not equal to length(chain_id).")
    elseif any(chain_counts .!= chain_counts[1])
        error("Not all chains have same number of iterations.")
    elseif maximum(chain_id) != n_chain
        error("max(chain_id) not equal to the number of chains.")
    end

    n_iter = lldim[1] / n_chain
    n_obs = lldim[2]
    a = Array{Float64}(undef, n_iter, n_chain, n_obs)
    for c in 1:n_chain
        a[:, c, :] = x[chain_id .== c, :]
    end
    return a
end


function validate_llfun(x)
    if !hasmethod(x, Tuple{Any, Any})
        error("Log-likelihood function must have at least the arguments 'data_i' and 'draws'")
    end
    return x
end


function nlist(args...; kwargs...)
    out = Dict{Symbol, Any}()
    for arg in args
        out[Symbol(arg)] = eval(Symbol(arg))
    end
    for (k, v) in kwargs
        out[k] = v
    end
    return out
end


function loo_cores(cores)
    loo_cores_op = get(ENV, "LOO_CORES", "NA")
    if loo_cores_op != "NA" && parse(Int, loo_cores_op) != cores
        cores = parse(Int, loo_cores_op)
        @warn "'LOO_CORES' is deprecated, please use 'MC_CORES' or pass 'cores' explicitly."
    end
    return cores
end


function release_questions()
    return [
        "Have you updated references?",
        "Have you updated inst/CITATION?",
        "Have you updated the vignettes?"
    ]
end
