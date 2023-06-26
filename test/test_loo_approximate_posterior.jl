using Test
using Random
using Distributions
using Optim
using DataFrames
using Loo


@testset "testing function methods" begin
    Random.seed!(123)
    N = 50
    K = 10
    S = 1000
    a0 = 1
    b0 = 1
    p = 0.5
    y = rand(Binomial(K, p), N)
    fake_data = DataFrame(y=y, K=fill(K, N))

    # The log posterior
    log_post = function(p, y, a0, b0, K)
        log_lik = sum(logpdf.(Binomial(K, p), y))  # the log likelihood
        log_post = log_lik + logpdf(Beta(a0, b0), p)  # the log prior
        log_post
    end


    it = optimize(p -> -log_post(p, y, a0, b0, K), 0.01, 0.99, GoldenSection())
    lap_params = Dict("mu" => it.minimizer, "sd" => sqrt(inv(-it.minimum)))

    a = a0 + sum(y)
    b = b0 + N * K - sum(y)
    fake_true_posterior = rand(Beta(a, b), S)
    fake_laplace_posterior = rand(Normal(lap_params["mu"], lap_params["sd"]), S)

    p_draws = vec(fake_laplace_posterior)
    log_p = [log_post(p_draws[s], y = y, a0 = a0, b0 = b0, K = K) for s in 1:S]
    log_g = logpdf.(Normal(lap_params["mu"], lap_params["sd"]), vec(fake_laplace_posterior))

    function llfun(data_i, draws)
        return logpdf(Binomial(data_i[:K]), data_i[:y]) .+ log.(draws)
    end
    
    ll = [llfun(fake_data[j, :], fake_laplace_posterior) for j in 1:N]

    @testset "loo_approximate_posterior.array works as loo_approximate_posterior.matrix" begin
        # Create array with two "chains"
        log_p_mat = reshape(log_p, (S ÷ 2, 2))
        log_g_mat = reshape(log_g, (S ÷ 2, 2))
        ll_array = zeros((S ÷ 2, 2, size(ll, 2)))
        ll_array[:, 1, :] = ll[1:(S ÷ 2), :]
        ll_array[:, 2, :] = ll[(S ÷ 2 + 1):S, :]
    
        # Assert that they are ok
        @test ll_array[1:2, 1, 1:2] == ll[1:2, 1:2]
        @test ll_array[1:2, 2, 1:2] == ll[(S ÷ 2 + 1):((S ÷ 2) + 2), 1:2]
    
        # Compute aploo
        aploo1 = loo_approximate_posterior(ll, log_p, log_g)
        aploo2 = loo_approximate_posterior(ll_array, log_p_mat, log_g_mat)
        aploo1b = loo(ll, r_eff = ones(N))
    
        # Check equivalence
        @test aploo1.estimates == aploo2.estimates
        @test typeof(aploo1) == typeof(aploo2)
        @test_throws ErrorException aploo1b.estimates == aploo2.estimates
        @test_throws ErrorException typeof(aploo1) == typeof(aploo1b)
    
        # Should fail with matrix
        @test_throws ErrorException loo_approximate_posterior(ll, reshape(log_p, :, 1), log_g)
        @test_throws ErrorException loo_approximate_posterior(ll, reshape(log_p, :, 1), reshape(log_g, :, 1))
    
        # Expect log_p and log_g be stored in the approximate_posterior in the same way
        @test length(aploo1.approximate_posterior.log_p) == size(ll, 1)
        @test length(aploo1.approximate_posterior.log_g) == size(ll, 1)
        @test aploo1.approximate_posterior.log_p == aploo2.approximate_posterior.log_p
        @test aploo1.approximate_posterior.log_g == aploo2.approximate_posterior.log_g
    end
    
    @testset "loo_approximate_posterior.function works as loo_approximate_posterior.matrix" begin
        # Compute aploo
        aploo1 = loo_approximate_posterior(ll, log_p, log_g)
        aploo1b = loo(ll, r_eff = ones(N))
        aploo2 = loo_approximate_posterior(llfun, log_p, log_g, fake_data, fake_laplace_posterior)
    
        # Check equivalence
        @test aploo1.estimates == aploo2.estimates
        @test typeof(aploo1) == typeof(aploo2)
        @test_throws ErrorException aploo1b.estimates == aploo2.estimates
    
        # Check equivalence
        # Expect log_p and log_g be stored in the approximate_posterior in the same way
        @test length(aploo2.approximate_posterior.log_p) == size(fake_laplace_posterior, 1)
        @test length(aploo2.approximate_posterior.log_g) == size(fake_laplace_posterior, 1)
        @test aploo1.approximate_posterior.log_p == aploo2.approximate_posterior.log_p
        @test aploo1.approximate_posterior.log_g == aploo2.approximate_posterior.log_g
    end
end