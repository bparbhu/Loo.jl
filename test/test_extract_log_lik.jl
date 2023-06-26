using Test
using Loo

@testset "extract_log_lik throws appropriate errors" begin
    x1 = randn(100)
    @test_throws ErrorException extract_log_lik(x1)
    x2 = x1 
    @test_throws ErrorException extract_log_lik(x2)
end
