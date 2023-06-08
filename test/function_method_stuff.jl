using Distributions
using DataFrames

N = 50; K = 10; S = 100; a0 = 3; b0 = 2
p = rand(Beta(a0, b0))
y = rand(Binomial(K, p), N)
a = a0 + sum(y); b = b0 + N * K - sum(y)
draws = rand(Beta(a, b), S)

data = DataFrame(y = y, K = fill(K, N))

llfun(data_i, draws) = logpdf(Binomial(data_i.K, draws), data_i.y)
llmat_from_fn = [llfun(data[i, :], draws) for i in 1:N]
