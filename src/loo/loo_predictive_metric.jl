function loo_predictive_metric(x, args...)
    # This is a placeholder for a method dispatch system
    # In Julia, the method would be chosen based on the type of x
end

function loo_predictive_metric(x::Matrix, y, log_lik, metric = "mae", r_eff = nothing, cores = 1)
    @assert isnumeric(x)
    @assert isnumeric(y)
    @assert size(x, 2) == length(y)
    @assert size(x) == size(log_lik)
    metric = lowercase(metric)
    psis_object = psis(-log_lik, r_eff = r_eff, cores = cores)
    pred_loo = E_loo(x, psis_object = psis_object, log_ratios = -log_lik).value

    predictive_metric_fun = loo_predictive_metric_fun(metric)

    return predictive_metric_fun(y, pred_loo)
end


function loo_predictive_metric_fun(metric)
    if metric == "mae"
        return mae
    elseif metric == "rmse"
        return rmse
    elseif metric == "mse"
        return mse
    elseif metric == "acc"
        return accuracy
    elseif metric == "balanced_acc"
        return balanced_accuracy
    else
        error("Invalid metric")
    end
end


function mae(y, yhat)
    @assert length(y) == length(yhat)
    n = length(y)
    e = abs.(y .- yhat)
    return Dict("estimate" => mean(e), "se" => std(e) / sqrt(n))
end


function mse(y, yhat)
    @assert length(y) == length(yhat)
    n = length(y)
    e = (y .- yhat).^2
    return Dict("estimate" => mean(e), "se" => std(e) / sqrt(n))
end


function rmse(y, yhat)
    est = mse(y, yhat)
    mean_mse = est["estimate"]
    var_mse = est["se"]^2
    var_rmse = var_mse / mean_mse / 4 # Comes from the first order Taylor approx.
    return Dict("estimate" => sqrt(mean_mse), "se" => sqrt(var_rmse))
end


function accuracy(y, yhat)
    @assert length(y) == length(yhat)
    @assert all(y .<= 1 .& y .>= 0)
    @assert all(yhat .<= 1 .& yhat .>= 0)
    n = length(y)
    yhat = Int.(yhat .> 0.5)
    acc = Int.(yhat .== y)
    est = mean(acc)
    return Dict("estimate" => est, "se" => sqrt(est * (1-est) / n))
end


function balanced_accuracy(y, yhat)
    @assert length(y) == length(yhat)
    @assert all(y .<= 1 .& y .>= 0)
    @assert all(yhat .<= 1 .& yhat .>= 0)
    n = length(y)
    yhat = Int.(yhat .> 0.5)
    mask = y .== 0

    tn = mean(yhat[mask] .== y[mask]) # True negatives
    tp = mean(yhat[.!mask] .== y[.!mask]) # True positives

    bls_acc = (tp + tn) / 2
    # This approximation has quite large bias for small samples
    bls_acc_var = (tp * (1 - tp) + tn * (1 - tn)) / 4
    return Dict("estimate" => bls_acc, "se" => sqrt(bls_acc_var / n))
end
