using Test
using Random
using Loo

Random.seed!(123456789)
n = 10
S = 100
y = randn(n)
x1 = randn(S, n)
x2 = randn(S, n)
ll = randn(S, n) .* 0.1 .- 1

@testset "crps computation is correct" begin
    @test crps_fun(2.0, 1.0) == 0.0
    @test crps_fun(1.0, 2.0) == -1.5
    @test crps_fun(π, π^2) == 0.5 * π - π^2

    @test crps_fun(1.0, 0.0, scale = true) == 0.0
    @test crps_fun(1.0, 2.0, scale = true) == -2.0
    @test crps_fun(π, π^2, scale = true) == -π^2/π - 0.5 * log(π)
end

@testset "crps matches references" begin
    Random.seed!(1)
    # Replace the reference results with the actual expected values
    @test crps(x1, x2, y) == "reference-results/crps.rds"
    @test scrps(x1, x2, y) == "reference-results/crps.rds"
    @test loo_crps(x1, x2, y, ll) == "reference-results/crps.rds"
    @test loo_scrps(x1, x2, y, ll) == "reference-results/crps.rds"
end

@testset "input validation throws correct errors" begin
    @test_throws ErrorException validate_crps_input(string.(x1), x2, y)
    @test_throws ErrorException validate_crps_input(x1, string.(x2), y)
    @test_throws ErrorException validate_crps_input(x1, x2, ["a", "b"])
    @test_throws ErrorException validate_crps_input(x1, transpose(x2), y)
    @test_throws ErrorException validate_crps_input(x1, x2, [1, 2])
    @test_throws ErrorException validate_crps_input(x1, x2, y, transpose(ll))
end

@testset "methods for single data point don't error" begin
    @test_throws ErrorException crps(x1[:,1], x2[:,1], y[1])
    @test_throws ErrorException scrps(x1[:,1], x2[:,1], y[1])
end
