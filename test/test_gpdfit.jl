using Test
using Random

@testset "gpdfit returns correct result" begin
    Random.seed!(123)
    x = randexp(100)
    gpdfit_val_old = gpdfit(x, wip=false, min_grid_pts = 80)
    @test gpdfit_val_old ≈ load("reference-results/gpdfit_old.jld")["data"]

    gpdfit_val_wip = gpdfit(x, wip=true, min_grid_pts = 80)
    @test gpdfit_val_wip ≈ load("reference-results/gpdfit.jld")["data"]

    gpdfit_val_wip_default_grid = gpdfit(x, wip=true)
    @test gpdfit_val_wip_default_grid ≈ load("reference-results/gpdfit_default_grid.jld")["data"]
end

@testset "qgpd returns the correct result" begin
    probs = range(0, stop=1, length=5)
    q1 = qgpd(probs, k = 1, sigma = 1)
    @test q1 ≈ [0, 1/3, 1, 3, Inf]

    q2 = qgpd(probs, k = 1, sigma = 0)
    @test all(isnan.(q2))
end
