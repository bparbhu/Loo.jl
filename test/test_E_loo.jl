using Test

LLarr = example_loglik_array()
LLmat = example_loglik_matrix()
LLvec = LLmat[:, 1]
chain_id = repeat(1:2, inner = size(LLarr)[1])
r_eff_mat = relative_eff(exp.(LLmat), chain_id)
r_eff_vec = relative_eff(exp.(LLvec), chain_id = chain_id)
psis_mat = psis(-LLmat, r_eff = r_eff_mat, cores = 2)
psis_vec = psis(-LLvec, r_eff = r_eff_vec)

Random.seed!(123)
x = randn(length(LLmat))
log_rats = -LLmat

# matrix method
E_test_mean = E_loo(x, psis_mat, type = "mean", log_ratios = log_rats)
E_test_var = E_loo(x, psis_mat, type = "var", log_ratios = log_rats)
E_test_quant = E_loo(x, psis_mat, type = "quantile", probs = 0.5, log_ratios = log_rats)
E_test_quant2 = E_loo(x, psis_mat, type = "quantile", probs = [0.1, 0.9], log_ratios = log_rats)

# vector method
E_test_mean_vec = E_loo(x[:, 1], psis_vec, type = "mean", log_ratios = log_rats[:,1])
E_test_var_vec = E_loo(x[:, 1], psis_vec, type = "var", log_ratios = log_rats[:,1])
E_test_quant_vec = E_loo(x[:, 1], psis_vec, type = "quant", probs = 0.5, log_ratios = log_rats[:,1])
E_test_quant_vec2 = E_loo(x[:, 1], psis_vec, type = "quant", probs = [0.1, 0.5, 0.9], log_ratios = log_rats[:,1])

# E_loo_khat
khat = E_loo_khat(x, psis_mat, log_rats)

@testset "E_loo" begin
    @testset "E_loo return types correct for matrix method" begin
        @test typeof(E_test_mean) == Dict{String, Any}
        @test length(E_test_mean) == 2
        @test length(E_test_mean["value"]) == size(x)[2]
        @test length(E_test_mean["pareto_k"]) == size(x)[2]

        @test typeof(E_test_var) == Dict{String, Any}
        @test length(E_test_var) == 2
        @test length(E_test_var["value"]) == size(x)[2]
        @test length(E_test_var["pareto_k"]) == size(x)[2]

        @test typeof(E_test_quant) == Dict{String, Any}
        @test length(E_test_quant) == 2
        @test length(E_test_quant["value"]) == size(x)[2]
        @test length(E_test_quant["pareto_k"]) == size(x)[2]

        @test typeof(E_test_quant2) == Dict{String, Any}
        @test length(E_test_quant2) == 2
        @test size(E_test_quant2["value"]) == (2, size(x)[2])
        @test length(E_test_quant2["pareto_k"]) == size(x)[2]
    end

    @testset "E_loo return types correct for default/vector method" begin
        @test typeof(E_test_mean_vec) == Dict{String, Any}
        @test length(E_test_mean_vec) == 2
        @test length(E_test_mean_vec["value"]) == 1
        @test length(E_test_mean_vec["pareto_k"]) == 1

        @test typeof(E_test_var_vec) == Dict{String, Any}
        @test length(E_test_var_vec) == 2
        @test length(E_test_var_vec["value"]) == 1
        @test length(E_test_var_vec["pareto_k"]) == 1

        @test typeof(E_test_quant_vec) == Dict{String, Any}
        @test length(E_test_quant_vec) == 2
        @test length(E_test_quant_vec["value"]) == 1
        @test length(E_test_quant_vec["pareto_k"]) == 1

        @test typeof(E_test_quant_vec2) == Dict{String, Any}
        @test length(E_test_quant_vec2) == 2
        @test length(E_test_quant_vec2["value"]) == 3
        @test length(E_test_quant_vec2["pareto_k"]) == 1
    end

    @testset "E_loo throws correct errors and warnings" begin
        @test_throws ErrorException E_loo(x, 1)
        @test_throws ErrorException E_loo(x, psis_mat, type = "quantile", probs = 2)
        @test_throws ErrorException E_loo(repeat(["a"], length(x)), psis_vec)
        @test_throws ErrorException E_loo(1:10, psis_vec)
        @test_throws ErrorException E_loo(hcat(1:10, 1:10), psis_mat)
    end

    @testset "weighted quantiles work" begin
        wquant_rapprox(x, w, probs) = begin
            @test all(probs .> 0 .& probs .< 1)
            ord = sortperm(x)
            d = x[ord]
            ww = w[ord]
            p = cumsum(ww) / sum(ww)
            return StatsBase.approx(p, d, probs, rule = 2).y
        end
        wquant_sim(x, w, probs, n_sims) = begin
            xx = sample(x, n_sims, replace = true, prob = w / sum(w))
            return quantile(xx, probs, names = false)
        end

        Random.seed!(123)
        pr = 0.025:0.025:0.975

        x1 = randn(100)
        w1 = exp.(randn(100))
        @test wquant(x1, w1, pr) ≈ wquant_rapprox(x1, w1, pr)

        x1 = randn(10000)
        w1 = exp.(randn(10000))
        @test wquant(x1, rep(1, length(x1)), pr) ≈ quantile(x1, probs = pr, names = false)
    end
end