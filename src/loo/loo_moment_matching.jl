using Base.Threads
using DataFrames
using LinearAlgebra


function loo_moment_match(x, args...)
    error("Method not implemented: loo_moment_match")
end

function loo_moment_match(x, loo, post_draws, log_lik_i,
                          unconstrain_pars, log_prob_upars,
                          log_lik_i_upars; max_iters = 30,
                          k_threshold = 0.7, split = true,
                          cov = true, cores = get(ENV, "JULIA_NUM_THREADS", 1),
                          args...)

    # input checks
    @assert isa(loo, LOO)
    @assert isa(post_draws, Function)
    @assert isa(log_lik_i, Function)
    @assert isa(unconstrain_pars, Function)
    @assert isa(log_prob_upars, Function)
    @assert isa(log_lik_i_upars, Function)
    @assert isa(max_iters, Number)
    @assert isa(k_threshold, Number)
    @assert isa(split, Bool)
    @assert isa(cov, Bool)
    @assert isa(cores, Number)

    if isa(loo, PSIS_LOO)
        is_method = "psis"
    else
        error("loo_moment_match currently supports only the \"psis\" importance sampling class.")
    end

    S, N = size(loo)
    pars = post_draws(x, args...)
    # transform the model parameters to unconstrained space
    upars = unconstrain_pars(x, pars = pars, args...)
    # number of parameters in the **parameters** block only
    npars = size(upars)[2]
    # if more parameters than samples, do not do Cholesky transformation
    cov = cov && S >= 10 * npars
    # compute log-probabilities of the original parameter values
    orig_log_prob = log_prob_upars(x, upars = upars, args...)

    # loop over all observations whose Pareto k is high
    ks = loo.diagnostics.pareto_k
    kfs = zeros(N)
    I = findall(ks .> k_threshold)

    loo_moment_match_i_fun = function(i)
        loo_moment_match_i(i = i, x = x, log_lik_i = log_lik_i,
                           unconstrain_pars = unconstrain_pars,
                           log_prob_upars = log_prob_upars,
                           log_lik_i_upars = log_lik_i_upars,
                           max_iters = max_iters, k_threshold = k_threshold,
                           split = split, cov = cov, N = N, S = S, upars = upars,
                           orig_log_prob = orig_log_prob, k = ks[i],
                           is_method = is_method, npars = npars, args...)
    end

    if cores == 1
        mm_list = map(loo_moment_match_i_fun, I)
    else
        mm_list = @threads for i in I
            loo_moment_match_i_fun(i)
        end
    end

    # update results
    for ii in 1:length(I)
        i = mm_list[ii].i
        loo.pointwise[i, "elpd_loo"] = mm_list[ii].elpd_loo_i
        loo.pointwise[i, "p_loo"] = mm_list[ii].p_loo
        loo.pointwise[i, "mcse_elpd_loo"] = mm_list[ii].mcse_elpd_loo
        loo.pointwise[i, "looic"] = mm_list[ii].looic

        loo.diagnostics.pareto_k[i] = mm_list[ii].k
        loo.diagnostics.n_eff[i] = mm_list[ii].n_eff
        kfs[i] = mm_list[ii].kf

        if !isnothing(loo.psis_object)
            loo.psis_object.log_weights[:, i] = mm_list[ii].lwi
        end
    end
    if !isnothing(loo.psis_object)
        loo.psis_object.norm_const_log = colLogSumExps(loo.psis_object.log_weights)
        loo.psis_object.diagnostics = loo.diagnostics
    end

    # combined estimates
    cols_to_summarize = setdiff(names(loo.pointwise), ["mcse_elpd_loo", "influence_pareto_k"])
    loo.estimates = table_of_estimates(loo.pointwise[:, cols_to_summarize])

    # these will be deprecated at some point
    loo.elpd_loo = loo.estimates["elpd_loo","Estimate"]
    loo.p_loo = loo.estimates["p_loo","Estimate"]
    loo.looic = loo.estimates["looic","Estimate"]
    loo.se_elpd_loo = loo.estimates["elpd_loo","SE"]
    loo.se_p_loo = loo.estimates["p_loo","SE"]
    loo.se_looic = loo.estimates["looic","SE"]

    # Warn if some Pareto ks are still high
    psislw_warnings(loo.diagnostics.pareto_k)
    # if we don't split, accuracy may be compromised
    if !split
        throw_large_kf_warning(kfs)
    end

    return loo
end


function loo_moment_match_i(i,
                            x,
                            log_lik_i,
                            unconstrain_pars,
                            log_prob_upars,
                            log_lik_i_upars,
                            max_iters,
                            k_threshold,
                            split,
                            cov,
                            N,
                            S,
                            upars,
                            orig_log_prob,
                            k,
                            is_method,
                            npars,
                            args...)

    # initialize values for this LOO-fold
    uparsi = upars
    ki = k
    kfi = 0
    log_liki = log_lik_i(x, i, args...)
    S_per_chain = size(log_liki)[1]
    N_chains = size(log_liki)[2]
    reshape(log_liki, S_per_chain, N_chains, 1)
    r_eff_i = relative_eff(exp(log_liki), cores = 1)
    reshape(log_liki, :)

    is_obj = try
        importance_sampling_default(-log_liki, method = is_method, r_eff = r_eff_i, cores = 1)
    catch
        nothing
    end
    lwi = vec(weights(is_obj))
    lwfi = fill(-logsumexp(zeros(S)), S)

    # initialize objects that keep track of the total transformation
    total_shift = zeros(npars)
    total_scaling = ones(npars)
    total_mapping = Matrix{Float64}(I, npars, npars)

    # try several transformations one by one
    # if one does not work, do not apply it and try another one
    # to accept the transformation, Pareto k needs to improve
    # when transformation succeeds, start again from the first one
    iterind = 1
    while iterind <= max_iters && ki > k_threshold

        if iterind == max_iters
            throw_moment_match_max_iters_warning()
        end

        # 1. match means
        trans = shift(x, uparsi, lwi)
        # gather updated quantities
        quantities_i = update_quantities_i(x, trans.upars,  i = i,
                                           orig_log_prob = orig_log_prob,
                                           log_prob_upars = log_prob_upars,
                                           log_lik_i_upars = log_lik_i_upars,
                                           r_eff_i = r_eff_i,
                                           cores = 1,
                                           is_method = is_method,
                                           args...)
        if quantities_i.ki < ki
            uparsi = trans.upars
            total_shift += trans.shift

            lwi = quantities_i.lwi
            lwfi = quantities_i.lwfi
            ki = quantities_i.ki
            kfi = quantities_i.kfi
            log_liki = quantities_i.log_liki
            iterind += 1
            continue
        end

        # 2. match means and marginal variances
        trans = shift_and_scale(x, uparsi, lwi)
        # gather updated quantities
        quantities_i = update_quantities_i(x, trans.upars,  i = i,
                                           orig_log_prob = orig_log_prob,
                                           log_prob_upars = log_prob_upars,
                                           log_lik_i_upars = log_lik_i_upars,
                                           r_eff_i = r_eff_i,
                                           cores = 1,
                                           is_method = is_method,
                                           args...)
        if quantities_i.ki < ki
            uparsi = trans.upars
            total_shift += trans.shift
            total_scaling .*= trans.scaling

            lwi = quantities_i.lwi
            lwfi = quantities_i.lwfi
            ki = quantities_i.ki
            kfi = quantities_i.kfi
            log_liki = quantities_i.log_liki
            iterind += 1
            continue
        end

        # 3. match means and covariances
        if cov
            trans = shift_and_cov(x, uparsi, lwi)
            # gather updated quantities
            quantities_i = update_quantities_i(x, trans.upars,  i = i,
                                               orig_log_prob = orig_log_prob,
                                               log_prob_upars = log_prob_upars,
                                               log_lik_i_upars = log_lik_i_upars,
                                               r_eff_i = r_eff_i,
                                               cores = 1,
                                               is_method = is_method,
                                               args...)

            if quantities_i.ki < ki
                uparsi = trans.upars
                total_shift += trans.shift
                total_mapping = trans.mapping * total_mapping

                lwi = quantities_i.lwi
                lwfi = quantities_i.lwfi
                ki = quantities_i.ki
                kfi = quantities_i.kfi
                log_liki = quantities_i.log_liki
                iterind += 1
                continue
            end
        end
        # none of the transformations improved khat
        # so there is no need to try further
        break
    end

    # transformations are now done
    # if we don't do split transform, or
    # if no transformations were successful
    # stop and collect values
    if split && (iterind > 1)
        # compute split transformation
        split_obj = loo_moment_match_split(
            x, upars, cov, total_shift, total_scaling, total_mapping, i,
            log_prob_upars = log_prob_upars, log_lik_i_upars = log_lik_i_upars,
            cores = 1, r_eff_i = r_eff_i, is_method = is_method, args...
        )
        log_liki = split_obj.log_liki
        lwi = split_obj.lwi
        lwfi = split_obj.lwfi
        r_eff_i = split_obj.r_eff_i
    else
        reshape(log_liki, S_per_chain, N_chains, 1)
        r_eff_i = relative_eff(exp(log_liki), cores = 1)
        reshape(log_liki, :)
    end

    # pointwise estimates
    elpd_loo_i = logsumexp(log_liki + lwi)
    lpd = logsumexp(log_liki) - log(length(log_liki))
    mcse_elpd_loo = mcse_elpd(
        ll = reshape(log_liki, :, 1), lw = reshape(lwi, :, 1),
        E_elpd = exp(elpd_loo_i), r_eff = r_eff_i
    )

    return Dict("elpd_loo_i" => elpd_loo_i,
                "p_loo" => lpd - elpd_loo_i,
                "mcse_elpd_loo" => mcse_elpd_loo,
                "looic" => -2 * elpd_loo_i,
                "k" => ki,
                "kf" => kfi,
                "n_eff" => min(1.0 / sum(exp.(2 .* lwi)),
                               1.0 / sum(exp.(2 .* lwfi))) * r_eff_i,
                "lwi" => lwi,
                "i" => i)
end


function update_quantities_i(x, upars, i, orig_log_prob,
                             log_prob_upars, log_lik_i_upars,
                             r_eff_i, is_method, args...)

    log_prob_new = log_prob_upars(x, upars = upars, args...)
    log_liki_new = log_lik_i_upars(x, upars = upars, i = i, args...)
    # compute new log importance weights

    is_obj_new = try
        importance_sampling_default(-log_liki_new + log_prob_new - orig_log_prob,
                                    method = is_method,
                                    r_eff = r_eff_i,
                                    cores = 1)
    catch
        nothing
    end
    lwi_new = vec(weights(is_obj_new))
    ki_new = is_obj_new.diagnostics.pareto_k

    is_obj_f_new = try
        importance_sampling_default(log_prob_new - orig_log_prob,
                                    method = is_method,
                                    r_eff = r_eff_i,
                                    cores = 1)
    catch
        nothing
    end
    lwfi_new = vec(weights(is_obj_f_new))
    kfi_new = is_obj_f_new.diagnostics.pareto_k

    # gather results
    return Dict("lwi" => lwi_new,
                "lwfi" => lwfi_new,
                "ki" => ki_new,
                "kfi" => kfi_new,
                "log_liki" => log_liki_new)
end


function shift(x, upars, lwi)
    # compute moments using log weights
    mean_original = mean(upars, dims=1)
    mean_weighted = sum(exp.(lwi) .* upars, dims=1)
    shift = mean_weighted .- mean_original
    # transform posterior draws
    upars_new = upars .+ shift
    return Dict("upars" => upars_new, "shift" => shift)
end


function shift_and_scale(x, upars, lwi)
    # compute moments using log weights
    S = size(upars, 1)
    mean_original = mean(upars, dims=1)
    mean_weighted = sum(exp.(lwi) .* upars, dims=1)
    shift = mean_weighted .- mean_original
    mii = exp.(lwi) .* upars.^2
    mii = sum(mii, dims=1) .- mean_weighted.^2
    mii = mii .* S ./ (S - 1)
    scaling = sqrt.(mii ./ var(upars, dims=1))
    # transform posterior draws
    upars_new = (upars .- mean_original) .* scaling .+ mean_weighted
    return Dict("upars" => upars_new, "shift" => shift, "scaling" => scaling)
end



function shift_and_cov(x, upars, lwi, args...)
    # compute moments using log weights
    mean_original = mean(upars, dims=1)
    mean_weighted = sum(exp.(lwi) .* upars, dims=1)
    shift = mean_weighted .- mean_original
    covv = cov(upars)
    wcovv = cov(upars, weights(exp.(lwi)))
    try
        chol1 = cholesky(wcovv)
    catch
        chol1 = nothing
    end
    if chol1 === nothing
        mapping = Matrix{Float64}(I, length(mean_original), length(mean_original))
    else
        chol2 = cholesky(covv)
        mapping = chol1.U' * inv(chol2.U')
    end
    # transform posterior draws
    upars_new = (upars .- mean_original) * mapping .+ mean_weighted
    return Dict("upars" => upars_new, "shift" => shift, "mapping" => mapping)
end


function throw_moment_match_max_iters_warning()
    @warn "The maximum number of moment matching iterations ('max_iters' argument) was reached. Increasing the value may improve accuracy."
end


function throw_large_kf_warning(kf, k_threshold = 0.5)
    if any(kf .> k_threshold)
        @warn "The accuracy of self-normalized importance sampling may be bad. Setting the argument 'split' to 'TRUE' will likely improve accuracy."
    end
end


function psislw_warnings(k)
    if any(k .> 0.7)
        @warn "Some Pareto k diagnostic values are too high. ", k_help()
    elseif any(k .> 0.5)
        @warn "Some Pareto k diagnostic values are slightly high. ", k_help()
    end
end
