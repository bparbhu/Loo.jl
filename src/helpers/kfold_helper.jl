using Random
using StatsBase

# kfold_split_random function
function kfold_split_random(K::Int, N::Int)
    @assert K > 1 && K <= N
    perm = randperm(N)
    idx = ceil.(range(1, stop=N, length=K+1))
    bins = cut(perm, breaks=idx, extend=false)
    return bins
end

# kfold_split_stratified function
function kfold_split_stratified(K::Int, x::Array)
    @assert K > 1 && K <= length(x)
    x = labelmap(x)
    Nlev = length(unique(x))
    N = length(x)
    xids = Int[]
    for l in 1:Nlev
        append!(xids, shuffle(findall(==(l), x)))
    end
    bins = fill(NaN, N)
    bins[xids] = repeat(1:K, inner=ceil(Int, N/K))[1:N]
    return bins
end

# kfold_split_grouped function
function kfold_split_grouped(K::Int, x::Array)
    @assert K > 1 && K <= length(x)
    Nlev = length(unique(x))
    @assert Nlev >= K
    x = labelmap(x)
    if Nlev == K
        return x
    end
    S1 = ceil(Int, Nlev / K)
    N_S2 = S1 * K - Nlev
    N_S1 = K - N_S2
    perm = randperm(Nlev)
    brks = range(S1 + 0.5, step=S1, length=N_S1)
    if N_S2 > 0
        brks2 = range(brks[N_S1] + S1 - 1, step=S1 - 1, length=N_S2 - 1)
        brks = vcat(brks, brks2)
    end
    grps = searchsortedfirst(brks, perm) .+ 1
    bins = fill(NaN, length(x))
    for j in perm
        bins[x .== j] .= grps[j]
    end
    return bins
end
