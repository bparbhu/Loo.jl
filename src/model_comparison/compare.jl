using DataFrames
using Statistics
using StatsBase

function compare(; x::Vector=Vector{Any}(), args...)
    if length(args) > 0
        if length(x) > 0
            throw(ArgumentError("If 'x' is specified then '...' should not be specified."))
        end
        dots = args
        nms = [string(k) for k in keys(args)]
    else
        if !isa(x, Vector) || length(x) == 0
            throw(ArgumentError("'x' must be a list."))
        end
        dots = x
        nms = names(dots)
        if length(nms) == 0
            nms = ["model$(i)" for i in 1:length(dots)]
        end
    end

    if !all([isa(d, LOO) for d in dots])
        throw(ArgumentError("All inputs should have class 'LOO'."))
    end
    if length(dots) <= 1
        throw(ArgumentError("'compare' requires at least two models."))
    elseif length(dots) == 2
        loo1 = dots[1]
        loo2 = dots[2]
        comp = compare_two_models(loo1, loo2)
        return comp
    else
        Ns = [size(d.pointwise, 1) for d in dots]
        if !all(Ns .== Ns[1])
            throw(ArgumentError("Not all models have the same number of data points."))
        end

        x = DataFrame([d.estimates for d in dots])
        rename!(x, nms)
        comp = x
        ord = sortperm(comp[findall(occursin.("^elpd", names(comp))), :], rev=true)
        comp = comp[ord, :]
        patts = [r"elpd", r"p_", r"^waic$|^looic$", r"^se_waic$|^se_looic$"]
        col_ord = reduce(vcat, [findall(occursin.(p, names(comp))) for p in patts])
        comp = comp[:, col_ord]

        # compute elpd_diff and se_elpd_diff relative to best model
        diffs = [elpd_diffs(dots[ord[1]], d) for d in dots[ord]]
        elpd_diff = sum(diffs, dims=1)
        se_diff = se_elpd_diff.(diffs)
        comp = hcat(DataFrame(elpd_diff = elpd_diff, se_diff = se_diff), comp)
        return comp
    end
end


function compare_two_models(loo_a::Loo, loo_b::Loo, return_val = ["elpd_diff", "se"], check_dims = true)
    if check_dims
        if size(loo_a.pointwise, 1) != size(loo_b.pointwise, 1)
            error("Models don't have the same number of data points. Found N_1 = $(size(loo_a.pointwise, 1)) and N_2 = $(size(loo_b.pointwise, 1))")
        end
    end

    diffs = elpd_diffs(loo_a, loo_b)
    comp = Dict("elpd_diff" => sum(diffs), "se" => se_elpd_diff(diffs))
    return comp
end
