using Test
using Random
using StatsBase
using Loo

Random.seed!(123)

LLarr = example_loglik_array()
LLarr2 = randn(size(LLarr)) .+ LLarr .+ 0.5
LLarr3 = randn(size(LLarr)) .+ LLarr .+ 1
w1 = waic(LLarr)
w2 = waic(LLarr2)

@testset "loo_compare" begin
    @testset "loo_compare throws appropriate errors" begin
        w3 = waic(LLarr[:,:,1:end-1])
        w4 = waic(LLarr[:,:,1:end-2])

        @test_throws ErrorException loo_compare(2, 3)
        @test_throws ErrorException loo_compare(w1, w2, x = [w1, w2])
        @test_throws ErrorException loo_compare(w1, [1,2,3])
        @test_throws ErrorException loo_compare(w1)
        @test_throws ErrorException loo_compare(x = [w1])
        @test_throws ErrorException loo_compare(w1, w3)
        @test_throws ErrorException loo_compare(w1, w2, w3)
    end

    @testset "loo_compare throws appropriate warnings" begin
        w3 = w1; w4 = w2
        # TODO: Add code to change class and add attributes
        # TODO: Add code to check for warnings
    end

    @testset "loo_compare returns expected results (2 models)" begin
        comp1 = loo_compare(w1, w1)
        @test typeof(comp1) == Matrix
        @test size(comp1) == (2, 8)
        @test comp1[1,1] ≈ 0 atol=1e-8
        @test comp1[2,1] ≈ 0 atol=1e-8

        comp2 = loo_compare(w1, w2)
        @test typeof(comp2) == Matrix
        # TODO: Add code to check against reference results
    end

    @testset "loo_compare returns expected result (3 models)" begin
        w3 = waic(LLarr3)
        comp1 = loo_compare(w1, w2, w3)
        @test typeof(comp1) == Matrix
        @test size(comp1) == (3, 8)
        @test comp1[1,1] ≈ 0 atol=1e-8
        # TODO: Add code to check against reference results
    end
end
