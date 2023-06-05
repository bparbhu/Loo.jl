using Optim
using Distributions
using StatsBase
using Printf

function loo_model_weights_default(x;
                                   method = "stacking",
                                   optim_method = BFGS(),
                                   optim_control = Optim.Options(),
                                   BB = true,
                                   BB_n = 1000,
                                   alpha = 1,
                                   r_eff_list = nothing,
                                   cores = Sys.CPU_THREADS)

    K = length(x) # number of models
    method = lowercase(method)

    if isa(x[1], Matrix)
        N = size(x[1], 2) # number of data points
        validate_log_lik_list(x)
        validate_r_eff_list(r_eff_list, K, N)
        lpd_point = Matrix{Float64}(undef, N, K)
        elpd_loo = Vector{Float64}(undef, K)
        for k in 1:K
            r_eff_k = getindex(r_eff_list, k, nothing) # possibly nothing
            log_likelihood = x[k]
            loo_object = loo(log_likelihood, r_eff = r_eff_k, cores = cores)
            lpd_point[:, k] .= loo_object.pointwise[:, "elpd_loo"]    #calculate log(p_k (y_i | y_-i))
            elpd_loo[k] = loo_object.estimates["elpd_loo", "Estimate"]
        end
    elseif isa(x[1], PSISLoo) # replace PSISLoo with the appropriate Julia type
        validate_psis_loo_list(x)
        lpd_point = hcat([obj.pointwise[:, "elpd_loo"] for obj in x]...)
        elpd_loo = [obj.estimates["elpd_loo", "Estimate"] for obj in x]
    else
        error("'x' must be a list of matrices or a list of 'psis_loo' objects.")
    end

    ## 1) stacking on log score
    if method == "stacking"
        wts = stacking_weights(
            lpd_point = lpd_point,
            optim_method = optim_method,
            optim_control = optim_control
        )
    else
        # method =="pseudobma"
        wts = pseudobma_weights(
            lpd_point = lpd_point,
            BB = BB,
            BB_n = BB_n,
            alpha = alpha
        )
    end

    if isa(x[1], Matrix)
        if !isnothing(names(x)) && all(!isempty(name) for name in names(x))
            wts = Dict(names(x) .=> wts)
        end
    else # list of loo objects
        wts = Dict(find_model_names(x) .=> wts)
    end
    return wts
end


function stacking_weights(lpd_point;
                          optim_method = BFGS(),
                          optim_control = Optim.Options())

    @assert isa(lpd_point, Matrix)
    N, K = size(lpd_point)
    if K < 2
        error("At least two models are required for stacking weights.")
    end

    exp_lpd_point = exp.(lpd_point)

    function negative_log_score_loo(w)
        # objective function: log score
        @assert length(w) == K - 1
        w_full = vcat(w, 1 - sum(w))
        return -sum(log.(exp_lpd_point * w_full))
    end

    function gradient(w)
        # gradient of the objective function
        @assert length(w) == K - 1
        w_full = vcat(w, 1 - sum(w))
        grad = zeros(K - 1)
        for k in 1:(K - 1)
            for i in 1:N
                grad[k] += (exp_lpd_point[i, k] - exp_lpd_point[i, K]) / (exp_lpd_point[i, :] ⋅ w_full)
            end
        end
        return -grad
    end

    ui = vcat(-ones(1, K - 1), Matrix{Float64}(I, K - 1, K - 1))  # K-1 simplex constraint matrix
    ci = vcat(-1, zeros(K - 1))
    w = Optim.minimizer(constrOptim(
        theta = fill(1 / K, K - 1),
        f = negative_log_score_loo,
        grad = gradient,
        ui = ui,
        ci = ci,
        method = optim_method,
        options = optim_control
    ))

    wts = Dict("model$(i)" => w[i] for i in 1:(K - 1))
    wts["model$K"] = 1 - sum(w)
    return wts
end


function pseudobma_weights(lpd_point;
                           BB = true,
                           BB_n = 1000,
                           alpha = 1)

    @assert isa(lpd_point, Matrix)
    N, K = size(lpd_point)
    if K < 2
        error("At least two models are required for pseudo-BMA weights.")
    end

    if !BB
        elpd = sum(lpd_point, dims=1)
        uwts = exp.(elpd .- maximum(elpd))
        wts = Dict("model$(i)" => uwts[i] for i in 1:K)
        return wts
    end

    temp = Matrix{Float64}(undef, BB_n, K)
    BB_weighting = rand(Dirichlet(alpha .* ones(N)), BB_n)
    for bb in 1:BB_n
        z_bb = BB_weighting[bb, :] ⋅ lpd_point * N
        uwts = exp.(z_bb .- maximum(z_bb))
        temp[bb, :] = uwts ./ sum(uwts)
    end
    wts = Dict("model$(i)" => mean(temp[:, i]) for i in 1:K)
    return wts
end



function dirichlet_rng(n, alpha)
    K = length(alpha)
    gamma_sim = reshape(rand(Gamma.(alpha), K * n), K, n)'
    gamma_sim ./ sum(gamma_sim, dims=2)
end


function print_weight_vector(x, digits=3)
    println("weight")
    for (k, v) in x
        @printf("%s %.*f\n", k, digits, v)
    end
    return nothing
end


function print_stacking_weights(x, digits=3)
    println("Method: stacking\n------")
    print_weight_vector(x, digits)
    return nothing
end


function print_pseudobma_weights(x, digits=3)
    println("Method: pseudo-BMA\n------")
    print_weight_vector(x, digits)
    return nothing
end


function print_pseudobma_bb_weights(x, digits=3)
    println("Method: pseudo-BMA+ with Bayesian bootstrap\n------")
    print_weight_vector(x, digits)
    return nothing
end


function validate_r_eff_list(r_eff_list, K, N)
    if r_eff_list == nothing
        return nothing
    end

    if length(r_eff_list) != K
        error("If r_eff_list is specified then it must contain one component for each model being compared.")
    end
    if any(length.(r_eff_list) .!= N)
        error("Each component of r_eff list must have the same length as the number of columns in the log-likelihood matrix.")
    end
    return nothing
end


function validate_log_lik_list(log_lik_list)
    if !isa(log_lik_list, Vector)
        error("log_lik_list must be a list.")
    end
    if length(log_lik_list) < 2
        error("At least two models are required.")
    end
    if length(unique(size.(log_lik_list, 2))) != 1 || length(unique(size.(log_lik_list, 1))) != 1
        error("Each log-likelihood matrix must have the same dimensions.")
    end
    return nothing
end


function validate_psis_loo_list(psis_loo_list)
    if !isa(psis_loo_list, Vector)
        error("psis_loo_list must be a list.")
    end
    if length(psis_loo_list) < 2
        error("At least two models are required.")
    end
    if !all(map(is_psis_loo, psis_loo_list))
        error("List elements must all be 'psis_loo' objects or log-likelihood matrices.")
    end

    dims = size.(psis_loo_list)
    if length(unique(first.(dims))) != 1 || length(unique(last.(dims))) != 1
        error("Each object in the list must have the same dimensions.")
    end
    return nothing
end
