using Test
using Random
using StatsBase: countmap
using Loo

@testset "kfold_split_random works" begin
    fold_rand = kfold_split_random(10, 100)
    @test length(fold_rand) == 100
    @test sort(unique(fold_rand)) == collect(1:10)
    @test sum(fold_rand .== 2) == sum(fold_rand .== 9)
end

@testset "kfold_split_stratified works" begin
    y = repeat([0, 1], inner = [10, 190])
    fold_strat = kfold_split_stratified(5, y)
    @test all(values(countmap(fold_strat)) .== 40)

    y = repeat([1, 2, 3], inner = [15, 33, 42])
    fold_strat = kfold_split_stratified(7, y)
    @test extrema(values(countmap(fold_strat))) == (12, 13)

    y = mtcars.cyl
    fold_strat = kfold_split_stratified(10, y)
    @test extrema(values(countmap(fold_strat))) == (3, 4)
end

@testset "kfold_split_grouped works" begin
    grp = repeat(state.name, inner = 15)[1:750]
    fold_group = kfold_split_grouped(grp)
    @test all(values(countmap(fold_group)) .== 75)
    @test sum(values(countmap(fold_group))) == length(grp)

    fold_group = kfold_split_grouped(9, grp)
    @test !all(values(countmap(fold_group)) .== 75)
    @test sum(values(countmap(fold_group))) == length(grp)

    grp = repeat(state.name, inner = 4)[1:200]
    grp[grp .== "Montana"] .= "Utah"
    fold_group = kfold_split_grouped(10, grp)
    @test sum(values(countmap(fold_group))) == length(grp) - 4

    grp = repeat(["A","B"], inner = 20)
    fold_group = kfold_split_grouped(2, grp)
    @test fold_group == map(x -> x == "A" ? 1 : 2, grp)
end


@testset "print_dims.kfold works" begin
    xx = Kfold(17)
    @test sprint(show, xx) == "Based on 17-fold cross-validation"

    xx.K = nothing
    @test sprint(show, xx) == ""
end
