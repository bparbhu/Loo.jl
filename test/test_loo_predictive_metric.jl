using Random
using Test
using JLD2
using Loo

LL = example_loglik_matrix()
chain_id = repeat(1:2, inner = size(LL, 1) ÷ 2)
r_eff = relative_eff(exp.(LL), chain_id)
psis_obj = psis(-LL, r_eff = r_eff, cores = 2)

Random.seed!(123)
x = randn(length(LL))
x = reshape(x, size(LL))
x_prob = 1 ./ (1 .+ exp.(-x))
y = randn(size(LL, 2))
y_binary = rand(Binomial(1, 0.5), size(LL, 2))

mae_mean = loo_predictive_metric(x, y, LL, metric = "mae", r_eff = r_eff)
mae_quant = loo_predictive_metric(x, y, LL, metric = "mae", r_eff = r_eff,
                                  type = "quantile", probs = 0.9)

rmse_mean = loo_predictive_metric(x, y, LL, metric = "rmse", r_eff = r_eff)
rmse_quant = loo_predictive_metric(x, y, LL, metric = "rmse", r_eff = r_eff,
                                   type = "quantile", probs = 0.9)

mse_mean = loo_predictive_metric(x, y, LL, metric = "mse", r_eff = r_eff)
mse_quant = loo_predictive_metric(x, y, LL, metric = "mse", r_eff = r_eff,
                                  type = "quantile", probs = 0.9)

acc_mean = loo_predictive_metric(x_prob, y_binary, LL, metric = "acc", r_eff = r_eff)
acc_quant = loo_predictive_metric(x_prob, y_binary, LL, metric = "acc", r_eff = r_eff,
                                  type = "quantile", probs = 0.9)

bacc_mean = loo_predictive_metric(x_prob, y_binary, LL, metric = "balanced_acc", r_eff = r_eff)
bacc_quant = loo_predictive_metric(x_prob, y_binary, LL, metric = "balanced_acc", r_eff = r_eff,
                                  type = "quantile", probs = 0.9)

@testset "loo_predictive_metric stops with incorrect inputs" begin
    @test_throws MethodError loo_predictive_metric(string.(x), y, LL, r_eff = r_eff)

    @test_throws ErrorException loo_predictive_metric(x, string.(y), LL, r_eff = r_eff)

    x_invalid = randn(9)
    x_invalid = reshape(x_invalid, (3, 3))
    @test_throws ErrorException loo_predictive_metric(x_invalid, y, LL, r_eff = r_eff)

    x_invalid = randn(64)
    x_invalid = reshape(x_invalid, (2, 32))
    @test_throws ErrorException loo_predictive_metric(x_invalid, y, LL, r_eff = r_eff)
end


@testset "loo_predictive_metric return types are correct" begin
    # MAE
    @test typeof(mae_mean) == Dict{String, Any}
    @test typeof(mae_quant) == Dict{String, Any}
    @test haskey(mae_mean, "estimate")
    @test haskey(mae_mean, "se")
    @test haskey(mae_quant, "estimate")
    @test haskey(mae_quant, "se")
    # RMSE
    @test typeof(rmse_mean) == Dict{String, Any}
    @test typeof(rmse_quant) == Dict{String, Any}
    @test haskey(rmse_mean, "estimate")
    @test haskey(rmse_mean, "se")
    @test haskey(rmse_quant, "estimate")
    @test haskey(rmse_quant, "se")
    # MSE
    @test typeof(mse_mean) == Dict{String, Any}
    @test typeof(mse_quant) == Dict{String, Any}
    @test haskey(mse_mean, "estimate")
    @test haskey(mse_mean, "se")
    @test haskey(mse_quant, "estimate")
    @test haskey(mse_quant, "se")
    # Accuracy
    @test typeof(acc_mean) == Dict{String, Any}
    @test typeof(acc_quant) == Dict{String, Any}
    @test haskey(acc_mean, "estimate")
    @test haskey(acc_mean, "se")
    @test haskey(acc_quant, "estimate")
    @test haskey(acc_quant, "se")
    # Balanced accuracy
    @test typeof(bacc_mean) == Dict{String, Any}
    @test typeof(bacc_quant) == Dict{String, Any}
    @test haskey(bacc_mean, "estimate")
    @test haskey(bacc_mean, "se")
    @test haskey(bacc_quant, "estimate")
    @test haskey(bacc_quant, "se")
end

using JLD2 # for loading .jld2 files

@testset "loo_predictive_metric is equal to reference" begin
    # Load reference results
    ref_mae_mean = load("reference-results/loo_predictive_metric_mae_mean.jld2")
    ref_mae_quant = load("reference-results/loo_predictive_metric_mae_quant.jld2")
    ref_rmse_mean = load("reference-results/loo_predictive_metric_rmse_mean.jld2")
    ref_rmse_quant = load("reference-results/loo_predictive_metric_rmse_quant.jld2")
    ref_mse_mean = load("reference-results/loo_predictive_metric_mse_mean.jld2")
    ref_mse_quant = load("reference-results/loo_predictive_metric_mse_quant.jld2")
    ref_acc_mean = load("reference-results/loo_predictive_metric_acc_mean.jld2")
    ref_acc_quant = load("reference-results/loo_predictive_metric_acc_quant.jld2")
    ref_bacc_mean = load("reference-results/loo_predictive_metric_bacc_mean.jld2")
    ref_bacc_quant = load("reference-results/loo_predictive_metric_bacc_quant.jld2")

    # Compare
    @test mae_mean == ref_mae_mean
    @test mae_quant == ref_mae_quant
    @test rmse_mean == ref_rmse_mean
    @test rmse_quant == ref_rmse_quant
    @test mse_mean == ref_mse_mean
    @test mse_quant == ref_mse_quant
    @test acc_mean == ref_acc_mean
    @test acc_quant == ref_acc_quant
    @test bacc_mean == ref_bacc_mean
    @test bacc_quant == ref_bacc_quant
end


@testset "MAE computation is correct" begin
    @test mae(repeat([0.5], 5), repeat([1], 5))["estimate"] == 0.5
    @test mae(repeat([0.5], 5), repeat([1], 5))["se"] == 0.0
    @test_throws DimensionMismatch mae(repeat([0.5], 5), repeat([1], 3))
end

@testset "MSE computation is correct" begin
    @test mse(repeat([0.5], 5), repeat([1], 5))["estimate"] == 0.25
    @test mse(repeat([0.5], 5), repeat([1], 5))["se"] == 0.0
    @test_throws DimensionMismatch mse(repeat([0.5], 5), repeat([1], 3))
end

@testset "RMSE computation is correct" begin
    @test rmse(repeat([0.5], 5), repeat([1], 5))["estimate"] == sqrt(0.25)
    @test mse(repeat([0.5], 5), repeat([1], 5))["se"] == 0.0
    @test_throws DimensionMismatch mse(repeat([0.5], 5), repeat([1], 3))
end

@testset "Accuracy computation is correct" begin
    @test accuracy([0, 0, 0, 1, 1, 1], [0.2, 0.2, 0.2, 0.7, 0.7, 0.7])["estimate"] == 1.0
    @test_throws DimensionMismatch accuracy([1, 0], [0.5])
    @test_throws ArgumentError accuracy([2, 1], [0.5, 0.5])
    @test_throws ArgumentError accuracy([1, 0], [1.1, 0.5])
end

@testset "Balanced accuracy computation is correct" begin
    @test balanced_accuracy([0, 0, 1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9, 0.9, 0.9])["estimate"] == 0.5
    @test_throws DimensionMismatch balanced_accuracy([1, 0], [0.5])
    @test_throws ArgumentError balanced_accuracy([2, 1], [0.5, 0.5])
    @test_throws ArgumentError balanced_accuracy([1, 0], [1.1, 0.5])
end
