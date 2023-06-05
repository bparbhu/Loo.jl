using DataFrames


function loo_compare_psis_loo_ss_list(x::Vector{psis_loo_ss})
    for i in 1:length(x)
        if !isa(x[i], psis_loo_ss)
            x[i] = convert(psis_loo_ss, x[i])
        end
    end

    loo_compare_checks_psis_loo_ss_list(x)

    comp = loo_compare_matrix_psis_loo_ss_list(x)
    ord = loo_compare_order(x)
    rename!(x, comp[ord, :])

    rnms = names(comp)
    elpd_diff_mat = DataFrame(elpd_diff = zeros(length(rnms)), 
                              se_diff = zeros(length(rnms)), 
                              subsampling_se_diff = zeros(length(rnms)))

    for i in 2:length(ord)
        elpd_diff_mat[i, :] .= loo_compare_ss(ref_loo = x[ord[1]], compare_loo = x[ord[i]])
    end

    comp = hcat(elpd_diff_mat, comp)
    rename!(comp, rnms)

    return comp
end


function loo_compare_ss(ref_loo::psis_loo_ss, compare_loo::psis_loo_ss)
    ref_idx = obs_idx(ref_loo)
    compare_idx = obs_idx(compare_loo)
    intersect_idx = intersect(ref_idx, compare_idx)
    ref_subset_of_compare = intersect_idx == ref_idx
    compare_subset_of_ref = intersect_idx == compare_idx

    # Using HH estimation
    if ref_loo.loo_subsampling.estimator == "hh_pps" || compare_loo.loo_subsampling.estimator == "hh_pps"
        @warn "Hansen-Hurwitz estimator used. Naive diff SE is used."
        return loo_compare_ss_naive(ref_loo, compare_loo)
    end

    # Same observations in both
    if compare_subset_of_ref && ref_subset_of_compare
        return loo_compare_ss_diff(ref_loo, compare_loo)
    end

    # Use subset
    if compare_subset_of_ref || ref_subset_of_compare
        if compare_subset_of_ref
            ref_loo = update(ref_loo, observations = compare_loo)
        end
        if ref_subset_of_compare
            compare_loo = update(compare_loo, observations = ref_loo)
        end
        @info "Estimated elpd_diff using observations included in loo calculations for all models."
        return loo_compare_ss_diff(ref_loo, compare_loo)
    end

    # If different samples
    if !compare_subset_of_ref && !ref_subset_of_compare
        @warn "Different subsamples in '", ref_loo, "' and '", compare_loo, "'. Naive diff SE is used."
        return loo_compare_ss_naive(ref_loo, compare_loo)
    end
end


function loo_compare_ss_naive(ref_loo::psis_loo_ss, compare_loo::psis_loo_ss)
    elpd_loo_diff = ref_loo.estimates["elpd_loo","Estimate"] - compare_loo.estimates["elpd_loo","Estimate"]
    elpd_loo_diff_se = sqrt(
        (ref_loo.estimates["elpd_loo","SE"])^2 +
        (compare_loo.estimates["elpd_loo","SE"])^2)
    elpd_loo_diff_subsampling_se = sqrt(
        (ref_loo.estimates["elpd_loo","subsampling SE"])^2 +
        (compare_loo.estimates["elpd_loo","subsampling SE"])^2)

    return (elpd_loo_diff, elpd_loo_diff_se, elpd_loo_diff_subsampling_se)
end


function loo_compare_ss_diff(ref_loo::psis_loo_ss, compare_loo::psis_loo_ss)
    @assert obs_idx(ref_loo) == obs_idx(compare_loo)

    # Assert not none as loo approximation
    @assert ref_loo.loo_subsampling.loo_approximation != "none"
    @assert compare_loo.loo_subsampling.loo_approximation != "none"

    diff_approx = ref_loo.loo_subsampling.elpd_loo_approx - compare_loo.loo_subsampling.elpd_loo_approx
    diff_sample = ref_loo.pointwise["elpd_loo"] - compare_loo.pointwise["elpd_loo"]
    est = srs_diff_est(diff_approx, y = diff_sample, y_idx = ref_loo.pointwise["idx"])

    elpd_loo_diff = est.y_hat
    elpd_loo_diff_se = sqrt(est.hat_v_y)
    elpd_loo_diff_subsampling_se = sqrt(est.v_y_hat)

    return (elpd_loo_diff, elpd_loo_diff_se, elpd_loo_diff_subsampling_se)
end


function loo_compare_checks_psis_loo_ss_list(loos)
    if length(loos) <= 1
        error("'loo_compare' requires at least two models.")
    end
    if !all([isinstance(x, psis_loo_ss) for x in loos])
        error("All inputs should have class 'psis_loo_ss'.")
    end

    Ns = [x.loo_subsampling.data_dim[1] for x in loos]
    if !all(Ns .== Ns[1])
        error("Not all models have the same number of data points.")
    end

    # TODO: Add warnings based on your specific requirements
end


function print_compare_loo_ss(x, digits = 1, simplify = true)
    xcopy = copy(x)
    if simplify && size(xcopy, 2) >= 2
        patts = ["elpd", "se", "subsampling_se"]
        xcopy = xcopy[:, [any(occursin(p, c) for p in patts) for c in names(xcopy)]]
    end
    for col in names(xcopy)
        xcopy[!, col] = round.(xcopy[!, col], digits=digits)
    end
    println(xcopy)
end


function loo_compare_matrix_psis_loo_ss_list(loos)
    tmp = map(loos) do x
        est = x.estimates
        Dict([row => est[row, "Estimate"] for row in rownames(est)]...
            , ["se_"*row => est[row, "SE"] for row in rownames(est)]...
            , ["subsampling_se_"*row => est[row, "subsampling SE"] for row in rownames(est)])
    end
    setproperty!(tmp, :colnames, find_model_names(loos))
    rnms = rownames(tmp)
    comp = tmp
    ord = loo_compare_order(loos)
    comp = comp[ord, :]
    patts = ["elpd", "p_", "waic", "looic", "se_waic", "se_looic"]
    col_ord = [col for col in colnames(comp) if any([occursin(patt, col) for patt in patts])]
    comp = comp[:, col_ord]
    comp
end
