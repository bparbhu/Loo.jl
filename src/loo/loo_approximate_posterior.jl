# Define the loo_approximate_posterior function for different types
function loo_approximate_posterior(x::Array, log_p, log_g; save_psis=false, cores=1)
    # Check inputs
    @assert isa(save_psis, Bool)
    @assert isa(cores, Int)
    @assert size(log_p) == size(x)
    @assert size(log_g) == size(log_p)

    # Convert x to a matrix
    ll = reshape(x, :, size(x)[end])

    # Convert log_p and log_g to vectors
    log_p = vec(log_p)
    log_g = vec(log_g)

    # Call the loo_approximate_posterior function for Matrix type
    loo_approximate_posterior(ll, log_p, log_g, save_psis=save_psis, cores=cores)
end


function loo_approximate_posterior(x::Matrix, log_p, log_g; save_psis=false, cores=1)
    # Check inputs
    @assert isa(save_psis, Bool)
    @assert isa(cores, Int)
    @assert length(log_p) == size(x, 1)
    @assert ndims(log_p) == 1
    @assert length(log_g) == length(log_p)
    @assert ndims(log_g) == 1

    # Call the psis_approximate_posterior function
    ap_psis = psis_approximate_posterior(log_p=log_p, log_g=log_g, log_liks=x, cores=cores, save_psis=save_psis)

    # Add the approximate_posterior field
    ap_psis.approximate_posterior = (log_p = log_p, log_g = log_g)

    # Add the class
    ap_psis.class = ["psis_loo_ap", typeof(ap_psis)]

    # Check the ap_psis object
    assert_psis_loo_ap(ap_psis)

    return ap_psis
end


function loo_approximate_posterior(x::Function, data, draws, log_p, log_g; save_psis=false, cores=1, kwargs...)
    # Check inputs
    @assert isa(save_psis, Bool)
    @assert isa(cores, Int)
    @assert length(log_p) == length(log_g)
    @assert isa(data, Matrix) || isa(data, DataFrame)
    @assert !isnothing(draws)

    # Validate the log-likelihood function
    llfun = validate_llfun(x)
    N = size(data, 1)

    # Call the parallel_psis_list function
    psis_list = parallel_psis_list(N=N, loo_i=loo_ap_i, llfun=llfun, data=data, draws=draws, 
                                   r_eff=nothing, save_psis=save_psis, log_p=log_p, log_g=log_g, cores=cores, kwargs...)

    # Extract pointwise and diagnostics
    pointwise = [psis_list[i]["pointwise"] for i in 1:length(psis_list)]
    if save_psis
        psis_object_list = [psis_list[i]["psis_object"] for i in 1:length(psis_list)]
        psis_out = list2importance_sampling(psis_object_list)
        diagnostics = psis_out.diagnostics
    else
        diagnostics_list = [psis_list[i]["diagnostics"] for i in 1:length(psis_list)]
        diagnostics = Dict(
            "pareto_k" => psis_apply(diagnostics_list, "pareto_k"),
            "n_eff" => psis_apply(diagnostics_list, "n_eff")
        )
    end

    # Create the ap_psis object
    ap_psis = importance_sampling_loo_object(
        pointwise=vcat(pointwise...),
        diagnostics=diagnostics,
        dims=(size(psis_list[1], 1), N),
        is_method="psis",
        is_object=save_psis ? psis_out : nothing
    )

    # Add the approximate_posterior field
    ap_psis.approximate_posterior = (log_p = log_p, log_g = log_g)

    # Add the class
    ap_psis.class = ["psis_loo_ap", typeof(ap_psis)]

    # Check the ap_psis object
    assert_psis_loo_ap(ap_psis)

    return ap_psis
end


function loo_ap_i(i, llfun, data, draws, log_p, log_g; r_eff=nothing, save_psis=false, is_method="psis", kwargs...)
    # Check inputs
    @assert is_method == "psis"
    @assert isa(save_psis, Bool)
    if !isnothing(r_eff)
        @warn "r_eff not implemented for aploo."
    end

    # Extract the i-th row of data
    d_i = data[i, :]

    # Call the log-likelihood function
    ll_i = llfun(data_i=d_i, draws=draws, kwargs...)
    if !isa(ll_i, Matrix)
        ll_i = reshape(ll_i, :, 1)
    end

    # Call the ap_psis function
    psis_out = ap_psis(log_ratios=-ll_i, log_p=log_p, log_g=log_g, cores=1)

    # Create the output structure
    out = Dict(
        "pointwise" => pointwise_loo_calcs(ll_i, psis_out),
        "diagnostics" => psis_out.diagnostics,
        "psis_object" => save_psis ? psis_out : nothing
    )
    out["S"] = size(psis_out, 1)
    out["N"] = 1

    return out
end


function assert_psis_loo_ap(x)
    # Check class
    @assert "psis_loo_ap" in x.class

    # Check names
    required_names = ["estimates", "pointwise", "diagnostics", "psis_object", "approximate_posterior"]
    @assert all(name in keys(x) for name in required_names)

    # Check approximate_posterior names
    required_ap_names = ["log_p", "log_g"]
    @assert all(name in keys(x.approximate_posterior) for name in required_ap_names)

    # Check log_p and log_g
    @assert length(x.approximate_posterior.log_p) == length(x.approximate_posterior.log_g)
    @assert all(!isnan(val) for val in x.approximate_posterior.log_p)
    @assert all(!isnan(val) for val in x.approximate_posterior.log_g)
end
