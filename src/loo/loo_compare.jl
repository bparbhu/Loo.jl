using DataFrames

function loo_compare_default(x::Any; kwargs...)
    if is_loo(x)
        loos = [x, kwargs...]
    else
        if !isa(x, AbstractArray) || length(x) == 0
            error("'x' must be a list if not a 'loo' object.")
        end
        if length(kwargs) > 0
            error("If 'x' is a list then '...' should not be specified.")
        end
        loos = x
    end

    # If subsampling is used
    if any(inherits_psis_loo_ss, loos)
        return loo_compare_psis_loo_ss_list(loos)
    end

    loo_compare_checks(loos)

    comp = loo_compare_matrix(loos)
    ord = loo_compare_order(loos)

    # compute elpd_diff and se_elpd_diff relative to best model
    rnms = rownames(comp)
    diffs = map(elpd_diffs, loos[ord[1]], loos[ord])
    elpd_diff = sum(diffs, dims=2)
    se_diff = se_elpd_diff(diffs, dims=2)
    comp = hcat(elpd_diff = elpd_diff, se_diff = se_diff, comp)
    rownames(comp) = rnms

    comp = setclass(comp, "compare.loo")
    return comp
end


function print_compare_loo(x::DataFrame; digits=1, simplify=true)
    xcopy = copy(x)
    if simplify && ncol(xcopy) >= 2
        patts = ["elpd_", "se_diff", "p_", "waic", "looic"]
        cols_to_keep = any(startswith.(string.(names(xcopy)), p) for p in patts)
        xcopy = xcopy[:, cols_to_keep]
    end
    println(round.(xcopy, digits=digits))
    return nothing
end


# Compute pointwise elpd differences
function elpd_diffs(loo_a::Dict, loo_b::Dict)
    pt_a = loo_a["pointwise"]
    pt_b = loo_b["pointwise"]
    elpd = findall(occursin.("^elpd", names(pt_a)))
    pt_b[:, elpd] .- pt_a[:, elpd]
end

# Compute standard error of the elpd difference
function se_elpd_diff(diffs::AbstractArray)
    N = length(diffs)
    # As `elpd_diff` is defined as the sum of N independent components,
    # we can compute the standard error by using the standard deviation
    # of the N components and multiplying by `sqrt(N)`.
    sqrt(N) * std(diffs)
end


function loo_compare_checks(loos::Array{Dict,1})
    ## errors
    if length(loos) <= 1
        error("'loo_compare' requires at least two models.")
    end
    if !all(map(is_loo, loos))
        error("All inputs should have class 'loo'.")
    end

    Ns = map(x -> size(x["pointwise"], 1), loos)
    if !all(Ns .== Ns[1])
        error("Not all models have the same number of data points.")
    end

    ## warnings

    yhash = map(x -> get(x, "yhash", nothing), loos)
    yhash_ok = map(x -> isequal(x, yhash[1]), yhash)
    if !all(yhash_ok)
        @warn "Not all models have the same y variable. ('yhash' attributes do not match)"
    end

    if all(map(is_kfold, loos))
        Ks = map(x -> get(x, "K", nothing), loos)
        if !all(Ks .== Ks[1])
            @warn "Not all kfold objects have the same K value. For a more accurate comparison use the same number of folds."
        end
    elseif any(map(is_kfold, loos)) && any(map(is_psis_loo, loos))
        @warn "Comparing LOO-CV to K-fold-CV. For a more accurate comparison use the same number of folds or loo for all models compared."
    end
end


function find_model_names(x::Array{Dict,1})
    out_names = fill("", length(x))

    names1 = [get(x[i], "name", "") for i in 1:length(x)]
    names2 = [get(get(x[i], "attributes", Dict()), "model_name", "") for i in 1:length(x)]
    names3 = [get(x[i], "model_name", "") for i in 1:length(x)]
    names4 = ["model"*string(i) for i in 1:length(x)]

    for j in 1:length(x)
        if !isempty(names1[j])
            out_names[j] = names1[j]
        elseif !isempty(names2[j])
            out_names[j] = names2[j]
        elseif !isempty(names3[j])
            out_names[j] = names3[j]
        else
            out_names[j] = names4[j]
        end
    end
    out_names
end


function loo_compare_matrix(loos::Vector{Loo})
    tmp = DataFrame()
    for loo in loos
        est = loo.estimates
        append!(est, Dict("se_"*k => v for (k, v) in est))
        tmp[loo.model_name] = est
    end

    patts = ["elpd", "p_", r"waic|looic", r"se_waic|se_looic"]
    col_ord = [col for col in names(tmp) if any(occursin.(patts, col))]

    comp = tmp[:, col_ord]
    comp = comp[sortperm(comp.elpd, rev=true), :]
    return comp
end



function loo_compare_order(loos::Array{Dict,1})
    tmp = hcat([vcat(get(get(x, "estimates", Dict()), "elpd_loo", 0),
                get(get(x, "estimates", Dict()), "p_loo", 0),
                get(get(x, "estimates", Dict()), "looic", 0),
                get(get(x, "estimates", Dict()), "se_elpd_loo", 0),
                get(get(x, "estimates", Dict()), "se_p_loo", 0),
                get(get(x, "estimates", Dict()), "se_looic", 0)) for x in loos]...)
    colnames = find_model_names(loos)
    rnms = ["elpd_loo", "p_loo", "looic", "se_elpd_loo", "se_p_loo", "se_looic"]
    ord = sortperm(tmp[findfirst(occursin("elpd", join(rnms, " "))), :], rev=true)
    return ord
end


function loo_compare_order(loos)
    tmp = DataFrame()
    for (i, loo) in enumerate(loos)
        est = loo.estimates
        tmp[!, i] = [est; [sqrt(val) for val in est]]
    end
    rename!(tmp, find_model_names(loos))
    rnms = names(tmp)
    elpd_indices = findall(x -> occursin(r"^elpd", x), rnms)
    ord = sortperm(tmp[elpd_indices, :], rev=true)
    return ord
end
