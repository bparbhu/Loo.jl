using Distributions, Random, DataFrames
using Loo
using Test, StatsFuns, MatrixStats


Random.seed!(123)
S = 4000

# helper functions for sampling from the posterior distribution
function rinvchisq(n, df, scale = 1/df)
    if ((length(scale) != 1) & (length(scale) != n))
        error("scale should be a scalar or a vector of the same length as x")
    end
    if (df <= 0)
        error("df must be greater than zero")
    end
    if (any(scale .<= 0))
        error("scale must be greater than zero")
    end
    return((df*scale)./rand(Chisq(df), n))
end

function dinvchisq(x, df, scale=1/df, log = false)
    if (df <= 0)
        error("df must be greater than zero")
    end
    if (scale <= 0)
        error("scale must be greater than zero")
    end
    nu = df/2
    if (log)
        return(ifelse(x .> 0, nu*log(nu) - log(gamma(nu)) + nu*log(scale) -
                    (nu .+ 1).*log.(x) - (nu*scale./x), NaN))
    else
        return(ifelse(x .> 0,
                  (((nu)^(nu))/gamma(nu)) * (scale^nu) *
                    (x.^(-(nu .+ 1))) .* exp.(-nu*scale./x), NaN))
    end
end


# generate toy data
# normally distributed data with known variance
data_sd = 1.1
data_mean = 1.3
n = 30
y = rand(Normal(data_mean, data_sd), n)
y_tilde = 11
y[1] = y_tilde

ymean = mean(y)
s2 = sum((y .- ymean).^2)/(n - 1)

# draws from the posterior distribution when including all observations
draws_full_posterior_sigma2 = rinvchisq(S, n - 1, s2)
draws_full_posterior_mu = rand(Normal(ymean, sqrt.(draws_full_posterior_sigma2/n)), S)


# create a dummy model object
x = Dict()
x["data"] = Dict()
x["data"]["y"] = y
x["data"]["n"] = n
x["data"]["ymean"] = ymean
x["data"]["s2"] = s2

x["draws"] = DataFrame(
  mu = draws_full_posterior_mu,
  sigma = sqrt.(draws_full_posterior_sigma2)
)

# implement functions for moment matching loo

# extract original posterior draws
function post_draws_test(x)
    convert(Matrix, x["draws"])
end

# extract original log lik draws
function log_lik_i_test(x, i)
    -0.5*log(2*π) - log.(x["draws"][:sigma]) - 1.0./(2*x["draws"][:sigma].^2).*(x["data"]["y"][i] - x["draws"][:mu]).^2
end

loglik = zeros(S,n)
for j in 1:n
    loglik[:,j] = log_lik_i_test(x, j)
end

# mu, log(sigma)
function unconstrain_pars_test(x, pars)
    upars = convert(Matrix, pars)
    upars[:,2] = log.(upars[:,2])
    upars
end

function log_prob_upars_test(x, upars)
    dinvchisq(exp.(upars[:,2]).^2, x["data"]["n"] - 1, x["data"]["s2"], log = true) +
    logpdf(Normal(x["data"]["ymean"], exp.(upars[:,2])/sqrt(x["data"]["n"])), upars[:,1])
end

# compute log_lik_i values based on the unconstrained parameters
function log_lik_i_upars_test(x, upars, i)
    -0.5*log(2*π) - upars[:,2] - 1.0./(2*exp.(upars[:,2]).^2).*(x["data"]["y"][i] - upars[:,1]).^2
end

upars = unconstrain_pars_test(x, x["draws"])
lwi_1 = -loglik[:,1]
lwi_1 = lwi_1 .- logsumexp(lwi_1)

@testset "log_prob_upars_test works" begin
    upars = unconstrain_pars_test(x, x["draws"])
    xloo = Dict()
    xloo["data"] = Dict()
    xloo["data"]["y"] = y[2:end]
    xloo["data"]["n"] = n - 1
    xloo["data"]["ymean"] = mean(y[2:end])
    xloo["data"]["s2"] = sum((y[2:end] .- mean(y[2:end])).^2)/(n - 2)

    post1 = log_prob_upars_test(x,upars)
    post1 = post1 .- logsumexp(post1)
    post2 = log_prob_upars_test(xloo,upars) .+ loglik[:,1]
    post2 = post2 .- logsumexp(post2)
    @test post1 ≈ post2
end


@testset "loo_moment_match.default warnings work" begin
    # loo object
    loo_manual = @suppress loo(loglik)
    loo_manual_tis = @suppress loo(loglik, is_method = "tis")

    @test_throws ErrorException loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                              unconstrain_pars_test, log_prob_upars_test,
                              log_lik_i_upars_test, max_iters = 30,
                              k_thres = 100, split = false,
                              cov = true, cores = 1)

    @test_throws ErrorException loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                              unconstrain_pars_test, log_prob_upars_test,
                              log_lik_i_upars_test, max_iters = 30,
                              k_thres = 0.5, split = false,
                              cov = true, cores = 1)

    @test_throws ErrorException loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                              unconstrain_pars_test, log_prob_upars_test,
                              log_lik_i_upars_test, max_iters = 1,
                              k_thres = 0.5, split = true,
                              cov = true, cores = 1)

    @test_throws ErrorException loo_moment_match(x, loo_manual_tis, post_draws_test, log_lik_i_test,
                       unconstrain_pars_test, log_prob_upars_test,
                       log_lik_i_upars_test, max_iters = 30,
                       k_thres = 0.5, split = true,
                       cov = true, cores = 1)
end


@testset "loo_moment_match.default works" begin
    # loo object
    loo_manual = @suppress loo(loglik)

    loo_moment_match_object = @suppress loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                                                unconstrain_pars_test, log_prob_upars_test,
                                                log_lik_i_upars_test, max_iters = 30,
                                                k_thres = 0.8, split = false,
                                                cov = true, cores = 1)

    # diagnostic pareto k decreases but influence pareto k stays the same
    @test loo_moment_match_object["diagnostics"]["pareto_k"][1] < loo_moment_match_object["pointwise"][1]["influence_pareto_k"]
    @test loo_moment_match_object["pointwise"]["influence_pareto_k"] == loo_manual["pointwise"]["influence_pareto_k"]
    @test loo_moment_match_object["pointwise"]["influence_pareto_k"] == loo_manual["diagnostics"]["pareto_k"]    

    # Here you would compare loo_moment_match_object with the reference result
    # using a function similar to `expect_equal_to_reference` in R

    # Here you would compare loo_moment_match_object2 with the reference result

    loo_moment_match_object3 = @suppress loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                                                unconstrain_pars_test, log_prob_upars_test,
                                                log_lik_i_upars_test, max_iters = 30,
                                                k_thres = 0.5, split = true,
                                                cov = true, cores = 1)

    # Here you would compare loo_moment_match_object3 with the reference result

    loo_moment_match_object4 = @suppress loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                                                unconstrain_pars_test, log_prob_upars_test,
                                                log_lik_i_upars_test, max_iters = 30,
                                                k_thres = 100, split = false,
                                                cov = true, cores = 1)

    @test loo_manual == loo_moment_match_object4

    loo_manual_with_psis = @suppress loo(loglik, save_psis = true)
    loo_moment_match_object5 = @suppress loo_moment_match(x, loo_manual_with_psis, post_draws_test, log_lik_i_test,
                                          unconstrain_pars_test, log_prob_upars_test,
                                          log_lik_i_upars_test, max_iters = 30,
                                          k_thres = 0.8, split = false,
                                          cov = true, cores = 1)

    @test loo_moment_match_object5["diagnostics"] == loo_moment_match_object5["psis_object"]["diagnostics"]
end


using Test

@testset "variance and covariance transformations work" begin
    Random.seed!(8493874)
    draws_full_posterior_sigma2 = rinvchisq(S, n - 1, s2)
    draws_full_posterior_mu = randn(S) .* sqrt.(draws_full_posterior_sigma2 ./ n) .+ ymean

    x["draws"] = DataFrame(mu = draws_full_posterior_mu, sigma = sqrt.(draws_full_posterior_sigma2))

    loglik = zeros(S, n)
    for j in 1:n
        loglik[:, j] = log_lik_i_test(x, j)
    end

    upars = unconstrain_pars_test(x, x["draws"])
    lwi_1 = -loglik[:, 1]
    lwi_1 = lwi_1 .- logsumexp(lwi_1)

    loo_manual = suppress_warnings(loo(loglik))

    loo_moment_match_object = suppress_warnings(loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                                         unconstrain_pars_test, log_prob_upars_test,
                                         log_lik_i_upars_test, max_iters = 30,
                                         k_thres = 0.0, split = false,
                                         cov = true, cores = 1))

    # You will need to implement the `expect_equal_to_reference` function in Julia
    expect_equal_to_reference(loo_moment_match_object, "reference-results/moment_match_var_and_cov.rds")
end

@testset "loo_moment_match.default works with multiple cores" begin
    loo_manual = suppress_warnings(loo(loglik))

    loo_moment_match_manual3 = suppress_warnings(loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                                                 unconstrain_pars_test, log_prob_upars_test,
                                                 log_lik_i_upars_test, max_iters = 30,
                                                 k_thres = 0.5, split = false,
                                                 cov = true, cores = 1))

    loo_moment_match_manual4 = suppress_warnings(loo_moment_match(x, loo_manual, post_draws_test, log_lik_i_test,
                                                 unconstrain_pars_test, log_prob_upars_test,
                                                 log_lik_i_upars_test, max_iters = 30,
                                                 k_thres = 0.5, split = false,
                                                 cov = true, cores = 2))

    @test loo_moment_match_manual3["diagnostics"]["pareto_k"] == loo_moment_match_manual4["diagnostics"]["pareto_k"]
    @test loo_moment_match_manual3["diagnostics"]["n_eff"] == loo_moment_match_manual4["diagnostics"]["n_eff"]

    @test loo_moment_match_manual3["estimates"] == loo_moment_match_manual4["estimates"]

    @test loo_moment_match_manual3["pointwise"] ≈ loo_moment_match_manual4["pointwise"] atol=5e-4
end

@testset "loo_moment_match_split works" begin
    is_obj_1 = suppress_warnings(importance_sampling_default(lwi_1, method = "psis", r_eff = 1, cores = 1))
    lwi_1_ps = weights(is_obj_1)


    split = loo_moment_match_split(x, upars, cov = false, total_shift = [0,0], total_scaling = [1,1], total_mapping = diagm([1,1]), i = 1,
                                   log_prob_upars = log_prob_upars_test, log_lik_i_upars = log_lik_i_upars_test,
                                   cores = 1, r_eff_i = 1, is_method = "psis")

    @testset "named split" begin
        @test haskey(split, "lwi")
        @test haskey(split, "lwfi")
        @test haskey(split, "log_liki")
        @test haskey(split, "r_eff_i")
    end

    @test lwi_1_ps == split["lwi"]

    split2 = loo_moment_match_split(x, upars, cov = false, total_shift = [-0.1,-0.2], total_scaling = [0.7,0.7],
                                    total_mapping = [1 0.1; 0.1 1], i = 1,
                                    log_prob_upars = log_prob_upars_test, log_lik_i_upars = log_lik_i_upars_test,
                                    cores = 1, r_eff_i = 1, is_method = "psis")

    # You will need to implement the `expect_equal_to_reference` function in Julia
    expect_equal_to_reference(split2, "reference-results/moment_match_split.rds")
end


@testset "passing arguments works" begin
    function log_lik_i_upars_test_additional_argument(x, upars, i; passed_arg = false)
        if !passed_arg
            @warn "passed_arg was not passed here"
        end
        -0.5*log(2*pi) - upars[:,2] - 1.0 ./ (2 .* exp.(upars[:,2]).^2) .* (x["data"]["y"][i] - upars[:,1]).^2
    end

    function unconstrain_pars_test_additional_argument(x, pars; passed_arg = false)
        if !passed_arg
            @warn "passed_arg was not passed here"
        end
        upars = convert(Matrix, pars)
        upars[:,2] = log.(upars[:,2])
        upars
    end

    function log_prob_upars_test_additional_argument(x, upars; passed_arg = false)
        if !passed_arg
            @warn "passed_arg was not passed here"
        end
        dinvchisq(exp.(upars[:,2]).^2, x["data"]["n"] - 1, x["data"]["s2"], log = true) +
        dnorm(upars[:,1], x["data"]["ymean"], exp.(upars[:,2]) ./ sqrt(x["data"]["n"]), log = true)
    end

    function post_draws_test_additional_argument(x; passed_arg = false)
        if !passed_arg
            @warn "passed_arg was not passed here"
        end
        convert(Matrix, x["draws"])
    end

    function log_lik_i_test_additional_argument(x, i; passed_arg = false)
        if !passed_arg
            @warn "passed_arg was not passed here"
        end
        -0.5*log(2*pi) - log.(x["draws"]["sigma"]) - 1.0 ./ (2 .* x["draws"]["sigma"].^2) .* (x["data"]["y"][i] - x["draws"]["mu"]).^2
    end

    loo_manual = suppress_warnings(loo(loglik))
    @test_throws ErrorException loo_moment_match(x, loo_manual, post_draws_test_additional_argument, log_lik_i_test_additional_argument,
                                 unconstrain_pars_test_additional_argument, log_prob_upars_test_additional_argument,
                                 log_lik_i_upars_test_additional_argument, max_iters = 30,
                                 k_thres = 0.5, split = true,
                                 cov = true, cores = 1, passed_arg = true)
end
