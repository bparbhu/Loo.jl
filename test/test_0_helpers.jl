using Test

LLarr = example_loglik_array()
LLmat = example_loglik_matrix()

@testset "helper functions and example data" begin
    @testset "example_loglik_array and example_loglik_matrix dimensions ok" begin
        dim_arr = size(LLarr)
        dim_mat = size(LLmat)
        @test dim_mat[1] == dim_arr[1] * dim_arr[2]
        @test dim_mat[2] == dim_arr[3]
    end

    @testset "example_loglik_array and example_loglik_matrix contain same values" begin
        @test LLmat[1:500, :] == LLarr[:, 1, :]
        @test LLmat[501:1000, :] == LLarr[:, 2, :]
    end

    @testset "reshaping functions result in correct dimensions" begin
        LLmat2 = llarray_to_matrix(LLarr)
        @test LLmat2 == LLmat

        LLarr2 = llmatrix_to_array(LLmat2, chain_id = repeat(1:2, inner = 500))
        @test LLarr2 == LLarr
    end

    @testset "reshaping functions throw correct errors" begin
        @test_throws ErrorException llmatrix_to_array(LLmat, chain_id = repeat(1:2, inner = [400, 600]))
        @test_throws ErrorException llmatrix_to_array(LLmat, chain_id = repeat(1:2, inner = 400))
        @test_throws ErrorException llmatrix_to_array(LLmat, chain_id = repeat(2:3, inner = 500))
        @test_throws ErrorException llmatrix_to_array(LLmat, chain_id = randn(1000))
    end

    @testset "colLogMeanExps(x) = log(colMeans(exp(x)))" begin
        @test colLogMeanExps(LLmat) == log(mean(exp.(LLmat), dims = 1))
    end

    @testset "validating log-lik objects and functions works" begin
        f_ok(data_i, draws) = nothing
        f_bad1(data_i) = nothing
        f_bad2(data, draws) = nothing
        @test validate_llfun(f_ok) == f_ok

        @test_throws ErrorException validate_llfun(f_bad1)
        @test_throws ErrorException validate_llfun(f_bad2)
    end

    @testset "nlist works" begin
        a = 1; b = 2; c = 3;
        nlist_val = [nlist(a, b, c), nlist(a, b, c = "tornado")]
        nlist_ans = [Dict("a" => 1, "b" => 2, "c" => 3), Dict("a" => 1, "b" => 2, "c" => "tornado")]
        @test nlist_val == nlist_ans
        @test nlist(a = 1, b = 2, c = 3) == Dict("a" => 1, "b" => 2, "c" => 3)
    end
    
    @testset "loo_cores works" begin
        @test loo_cores(10) == 10
        ENV["mc.cores"] = 2
        @test loo_cores(get(ENV, "mc.cores", 1)) == 2
        ENV["mc.cores"] = 1

        ENV["loo.cores"] = 2
        @test loo_cores(10) == 2
        delete!(ENV, "loo.cores")
    end
end
