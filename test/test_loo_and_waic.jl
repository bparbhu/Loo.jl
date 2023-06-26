using Test
using Random
using Loo

Random.seed!(123)

@testset "loo, waic and elpd" begin
    LLarr = example_loglik_array()
    LLmat = example_loglik_matrix()
    LLvec = LLmat[:, 1]
    chain_id = repeat(1:2, inner = size(LLarr, 1))
    r_eff_arr = relative_eff(exp.(LLarr))
    r_eff_mat = relative_eff(exp.(LLmat), chain_id = chain_id)

    loo1 = loo(LLarr, r_eff = r_eff_arr)
    waic1 = waic(LLarr)
    elpd1 = elpd(LLarr)

    @testset "using loo.cores is deprecated" begin
        ENV["mc.cores"] = nothing
        ENV["loo.cores"] = 1
        @test_throws Warning loo(LLarr, r_eff = r_eff_arr, cores = 2)
        ENV["loo.cores"] = nothing
        ENV["mc.cores"] = 1
    end

    @testset "loo, waic and elpd results haven't changed" begin
        @test loo1 == load("reference-results/loo.jld")
        @test waic1 == load("reference-results/waic.jld")
        @test elpd1 == load("reference-results/elpd.jld")
    end

    @testset "loo with cores=1 and cores=2 gives same results" begin
        loo2 = loo(LLarr, r_eff = r_eff_arr, cores = 2)
        @test loo1.estimates == loo2.estimates
    end

    @testset "waic returns object with correct structure" begin
        @test iswaic(waic1)
        @test isloo(waic1)
        @test !ispsis_loo(waic1)
        @test names(waic1) == ["estimates", "pointwise", "elpd_waic", "p_waic", "waic", "se_elpd_waic", "se_p_waic", "se_waic"]
        est_names = propertynames(waic1.estimates)
        @test est_names == [:elpd_waic, :p_waic, :waic]
        @test propertynames(waic1.pointwise) == est_names
        @test size(waic1) == size(LLmat)
    end

    @testset "two pareto k values are equal" begin
        @test loo1.pointwise.influence_pareto_k == loo1.diagnostics.pareto_k
    end
    
    @testset "loo.array and loo.matrix give same result" begin
        l2 = loo(LLmat, r_eff = r_eff_mat)
        @test loo1.estimates == l2.estimates
        @test loo1.diagnostics == l2.diagnostics
        @test loo1.pointwise[:, Not(:mcse_elpd_loo)] == l2.pointwise[:, Not(:mcse_elpd_loo)]
        @test isapprox(loo1.pointwise.mcse_elpd_loo, l2.pointwise.mcse_elpd_loo, atol = 0.005)
    end
    
    @testset "loo.array runs with multiple cores" begin
        loo_with_arr1 = loo(LLarr, cores = 1, r_eff = nothing)
        loo_with_arr2 = loo(LLarr, cores = 2, r_eff = nothing)
        @test loo_with_arr1.estimates == loo_with_arr2.estimates
    end
    
    @testset "waic.array and waic.matrix give same result" begin
        waic2 = waic(LLmat)
        @test waic1 == waic2
    end

    @testset "elpd.array and elpd.matrix give same result" begin
        elpd2 = elpd(LLmat)
        @test elpd1 == elpd2
    end
    
    @testset "loo, waic, and elpd error with vector input" begin
        @test_throws MethodError loo(LLvec)
        @test_throws MethodError waic(LLvec)
        @test_throws MethodError elpd(LLvec)
    end
end


@testset "testing function methods" begin
    include("data-for-tests/function_method_stuff.jl")

    waic_with_fn = waic(llfun, data = data, draws = draws)
    waic_with_mat = waic(llmat_from_fn)

    loo_with_fn = loo(llfun, data = data, draws = draws, r_eff = ones(size(data, 1)))
    loo_with_mat = loo(llmat_from_fn, r_eff = ones(size(llmat_from_fn, 2)), save_psis = true)

    @testset "loo.cores deprecation warning works with function method" begin
        ENV["loo.cores"] = "1"
        @test_throws Warning loo(llfun, cores = 2, data = data, draws = draws, r_eff = ones(size(data, 1)))
        ENV["loo.cores"] = nothing
    end

    @testset "loo_i results match loo results for ith data point" begin
        @test_throws Warning loo_i_val = loo_i(i = 2, llfun = llfun, data = data, draws = draws)
        @test loo_i_val.pointwise[:, "elpd_loo"] == loo_with_fn.pointwise[2, "elpd_loo"]
        @test loo_i_val.pointwise[:, "p_loo"] == loo_with_fn.pointwise[2, "p_loo"]
        @test loo_i_val.diagnostics.pareto_k == loo_with_fn.diagnostics.pareto_k[2]
        @test loo_i_val.diagnostics.n_eff == loo_with_fn.diagnostics.n_eff[2]
    end

    @testset "function and matrix methods return same result" begin
        @test waic_with_mat == waic_with_fn
        @test loo_with_mat.estimates == loo_with_fn.estimates
        @test loo_with_mat.diagnostics == loo_with_fn.diagnostics
        @test size(loo_with_mat) == size(loo_with_fn)
    end

    @testset "loo.function runs with multiple cores" begin
        loo_with_fn1 = loo(llfun, data = data, draws = draws, r_eff = ones(size(data, 1)), cores = 1)
        loo_with_fn2 = loo(llfun, data = data, draws = draws, r_eff = ones(size(data, 1)), cores = 2)
        @test loo_with_fn2.estimates == loo_with_fn1.estimates
    end

    @testset "save_psis option to loo.function makes correct psis object" begin
        loo_with_fn2 = loo(llfun, data = data, draws = draws, r_eff = ones(size(data, 1)), save_psis = true)
        @test loo_with_fn2.psis_object == loo_with_mat.psis_object
    end

    @testset "loo throws r_eff warnings" begin
        @test_throws Warning loo(-LLarr)
        @test_throws Warning loo(-LLmat)
        @test_throws Warning loo(llfun, data = data, draws = draws)
    end
end
