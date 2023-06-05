using Statistics
using LinearAlgebra
using StatsFun
using Distributions
using Distributed


function match_arg(arg, choices)
    if !(arg in choices)
        error("'$arg' is not a valid choice. Choose from: ", join(choices, ", "))
    end
    return arg
end


function loo_array(x;r_eff = nothing,save_psis = false,cores = get(ENV, "MC_CORES", 1),is_method = "psis")
    if r_eff === nothing
        throw_loo_r_eff_warning()
    end
    is_method = match_arg(is_method, ["psis", "tis", "sis"])
    psis_out = importance_sampling_array(log_ratios = -x, r_eff = r_eff, cores = cores, method = is_method)
    ll = llarray_to_matrix(x)
    pointwise = pointwise_loo_calcs(ll, psis_out)
    return importance_sampling_loo_object(
                pointwise = pointwise,
                diagnostics = psis_out.diagnostics,
                dims = size(psis_out),
                is_method = is_method,
                is_object = save_psis ? psis_out : nothing
    )
end


function loo_matrix(x;
        r_eff = nothing,
        save_psis = false,
        cores = get(ENV, "MC_CORES", 1),
        is_method = "psis")
    is_method = match_arg(is_method, ["psis", "tis", "sis"])
    if r_eff === nothing
        throw_loo_r_eff_warning()
    end
    
    psis_out = importance_sampling_matrix(log_ratios = -x, r_eff = r_eff, cores = cores, method = is_method)
    pointwise = pointwise_loo_calcs(x, psis_out)
    return importance_sampling_loo_object(
        pointwise = pointwise,
        diagnostics = psis_out.diagnostics,
        dims = size(psis_out),
        is_method = is_method,
        is_object = save_psis ? psis_out : nothing
)
end


function loo_function(x;data = nothing,draws = nothing,r_eff = nothing,save_psis = false,cores = get(ENV, "MC_CORES", 1),is_method = "psis")
    is_method = match_arg(is_method, ["psis", "tis", "sis"])
    cores = loo_cores(cores)
    if !(isa(data, DataFrame) || isa(data, Matrix)) || draws === nothing
        error("Data must be a DataFrame or Matrix and draws must not be null.")
    end
    assert_importance_sampling_method_is_implemented(is_method)
    llfun = validate_llfun(x)
    N = size(data, 1)

    if r_eff === nothing
        throw_loo_r_eff_warning()
    else
        r_eff = prepare_psis_r_eff(r_eff, len = N)
    end

    psis_list = parallel_importance_sampling_list(
    N = N,
    llfun = llfun,
    data = data,
    draws = draws,
    r_eff = r_eff,
    save_psis = save_psis,
    cores = cores,
    method = is_method
    )

    pointwise = [psis["pointwise"] for psis in psis_list]
    if save_psis
        psis_object_list = [psis["psis_object"] for psis in psis_list]
        psis_out = list2importance_sampling(psis_object_list)
        diagnostics = psis_out.diagnostics
    else
        diagnostics_list = [psis["diagnostics"] for psis in psis_list]
        diagnostics = Dict(
        :pareto_k => psis_apply(diagnostics_list, "pareto_k"),
        :n_eff => psis_apply(diagnostics_list, "n_eff")
    )
    end

    return importance_sampling_loo_object(
                pointwise = vcat(pointwise...),
                diagnostics = diagnostics,
                dims = (getfield(psis_list[1], :S), N),
                is_method = is_method,
            is_object = save_psis ? psis_out : nothing
)
end

function loo(x::Array, args...; kwargs...)
    loo_array(x, args...; kwargs...)
end

function loo(x::Matrix, args...; kwargs...)
    loo_matrix(x, args...; kwargs...)
end

function loo(x::Function, args...; kwargs...)
    loo_function(x, args...; kwargs...)
end


function loo_i(i,llfun,data = nothing,draws = nothing,r_eff = nothing,is_method = "psis")
    if !(i isa Integer) || 
        !(isa(llfun, Function) || isa(llfun, AbstractString)) ||
        !(isa(data, DataFrame) || isa(data, Matrix)) ||
        i > size(data, 1) ||
        draws === nothing ||
        !(is_method in implemented_is_methods())
        error("Invalid arguments.")
    end
    return loo_i_internal(i = i,llfun = llfun,data = data,draws = draws,r_eff = r_eff[i],save_psis = false,is_method = is_method)
end


function loo_i_internal(i,llfun,data,draws,r_eff = nothing,save_psis = false,is_method)
    if r_eff !== nothing
        r_eff = r_eff[i]
    end
    d_i = data[i, :]
    ll_i = llfun(data_i = d_i, draws = draws)
    if !(ll_i isa Matrix)
        ll_i = Matrix(ll_i)
    end
    psis_out = importance_sampling_matrix(
                    log_ratios = -ll_i,
                    r_eff = r_eff,
                    cores = 1,
                    method = is_method
                )
    return (pointwise = pointwise_loo_calcs(ll_i, psis_out),
            diagnostics = psis_out.diagnostics,
            psis_object = save_psis ? psis_out : nothing,
            S = size(psis_out, 1),
            N = 1)
end

dim_loo(x) = getfield(x, :dims)

is_loo(x) = typeof(x) <: LOO

dim_psis_loo(x) = getfield(x, :dims)

is_psis_loo(x) = typeof(x) <: PSIS_LOO && is_loo(x)


function pointwise_loo_calcs(ll, psis_object)
    if !(ll isa Matrix)
        ll = Matrix(ll)
    end
    lw = weights(psis_object, normalize = true, log = true)
    elpd_loo = logsumexp(ll .+ lw, dims=1)
    lpd = logsumexp(ll, dims=1) .- log(size(ll, 1)) # colLogMeanExps
    p_loo = lpd .- elpd_loo
    mcse_elpd_loo = mcse_elpd(ll, lw, E_elpd = elpd_loo, r_eff = relative_eff(psis_object))
    looic = -2 .* elpd_loo
    influence_pareto_k = psis_object.diagnostics.pareto_k
    return hcat(elpd_loo, mcse_elpd_loo, p_loo, looic, influence_pareto_k)
end

function importance_sampling_loo_object(pointwise, diagnostics, dims,
                                        is_method, is_object = nothing)
    if !(pointwise isa Matrix)
        error("Internal error ('pointwise' must be a matrix)")
    end
    if !(diagnostics isa Dict)
        error("Internal error ('diagnostics' must be a Dict)")
    end
    assert_importance_sampling_method_is_implemented(is_method)

    cols_to_summarize = [!(colname in ["mcse_elpd_loo", "influence_pareto_k"]) for colname in names(pointwise)]
    estimates = table_of_estimates(pointwise[:, cols_to_summarize])

    out = Dict(:estimates => estimates, :pointwise => pointwise, :diagnostics => diagnostics)
    if is_object === nothing
        out[Symbol(is_method * "_object")] = nothing
    else
        out[Symbol(is_method * "_object")] = is_object
    end

    # maintain backwards compatibility
    old_nms = ["elpd_loo", "p_loo", "looic", "se_elpd_loo", "se_p_loo", "se_looic"]
    out = merge(out, Dict(Symbol(old_nm) => estimates for old_nm in old_nms))

    return (out, dims = dims, class = [is_method * "_loo", "importance_sampling_loo", "loo"])
end


function mcse_elpd(ll, lw, E_elpd, r_eff, n_samples = 1000)
    lik = exp.(ll)
    w2 = exp.(lw).^2
    E_epd = exp.(E_elpd)
    # zn is approximate ordered statistics of unit normal distribution with offset
    # recommended by Blom (1958)
    S = n_samples
    c = 3/8
    r = 1:n_samples
    zn = quantile(Normal(), (r .- c) ./ (S .- 2 .* c .+ 1))
    var_elpd = [var(log.(z[z .> 0])) for z in (E_epd[i] .+ sqrt(sum(w2[:, i] .* (lik[:, i] .- E_epd[i]).^2)) .* zn for i in 1:size(w2, 2))]
    return sqrt.(var_elpd ./ r_eff)
end

function throw_loo_r_eff_warning()
    @warn "Relative effective sample sizes ('r_eff' argument) not specified.\nFor models fit with MCMC, the reported PSIS effective sample sizes and \nMCSE estimates will be over-optimistic."
end


function list2importance_sampling(objects)
    log_weights = [getfield(obj, :log_weights) for obj in objects]
    diagnostics = [getfield(obj, :diagnostics) for obj in objects]

    method = unique([getfield(obj, :method) for obj in objects])
    if length(method) == 1
        classes = [method[1], "importance_sampling", "list"]
    else
        classes = ["importance_sampling", "list"]
    end

    return (
        log_weights = log_weights,
        diagnostics = Dict(
            :pareto_k => [getfield(d, :pareto_k) for d in diagnostics],
            :n_eff => [getfield(d, :n_eff) for d in diagnostics]
        ),
        norm_const_log = [getfield(obj, :norm_const_log) for obj in objects],
        tail_len = [getfield(obj, :tail_len) for obj in objects],
        r_eff = [getfield(obj, :r_eff) for obj in objects],
        dims = size(log_weights),
        method = method,
        class = classes
    )
end



function get_loo(x, i)
    flags = ["elpd_loo", "se_elpd_loo", "p_loo", "se_p_loo", "looic", "se_looic",
             "elpd_waic", "se_elpd_waic", "p_waic", "se_p_waic", "waic", "se_waic"]

    if i isa String
        needs_warning = findall(x -> x == i, flags)
        if length(needs_warning) > 0
            @warn "Accessing $(flags[needs_warning[1]]) using 'get_loo' is deprecated and will be removed in a future release. Please extract the $(flags[needs_warning[1]]) estimate from the 'estimates' component instead."
        end
    end
    return x[i]
end


function getfield_loo(x, name)
    flags = ["elpd_loo", "se_elpd_loo", "p_loo", "se_p_loo", "looic", "se_looic",
             "elpd_waic", "se_elpd_waic", "p_waic", "se_p_waic", "waic", "se_waic"]
    needs_warning = findall(x -> x == name, flags)
    if length(needs_warning) > 0
        @warn "Accessing $(flags[needs_warning[1]]) using 'getfield_loo' is deprecated and will be removed in a future release. Please extract the $(flags[needs_warning[1]]) estimate from the 'estimates' component instead."
    end
    return getfield(x, Symbol(name))
end


function parallel_psis_list(N, loo_i, llfun, data, draws, r_eff, save_psis, cores, method)
    if cores == 1
        psis_list = [loo_i(i, llfun, data, draws, r_eff, save_psis, method) for i in 1:N]
    else
        addprocs(cores - 1)  # add additional processes
        @distributed (vcat) for i in 1:N
            loo_i(i, llfun, data, draws, r_eff, save_psis, method)
        end
    end
end
