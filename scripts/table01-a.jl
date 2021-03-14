# Code for the main results, listed in table 1, two-type DL model
# Author: Arthur Zwaenepoel
using TwoTypeDLModel
using TwoTypeDLModel: BetaGeometric
using DataFrames, CSV, NewickTree, Distributions
using Serialization, StatsBase, ThreadTools

prefix = "primates-GO:0002376"  # drosophila-8taxa | GO:0002376
suffix = string(rand())[3:6]  # terrible unique suffix
tree   = readnw(readline("data/$prefix.nw"))
rdata  = CSV.read("data/$prefix.csv", DataFrame)
pth    = mkpath("output/table01-a/$prefix/$suffix")
root   = Dict("ygob"                => (0.98, 3.06), 
              "drosophila"          => (0.96, 3.01),
              "primates-GO:0002376" => (0.92, 2.23))

meanμ  = Dict("ygob" => 1.61, "drosophila" => 3.94, "primates-GO:0002376" => 3.46)

# Two-type DL model
# =================
# set root prior 
ETA, ZETA = root[prefix]
settings = PSettings(n=16, N=32, abstol=1e-7, reltol=1e-7)
rprior = TwoTypeDLModel.BBGPrior(ETA, ZETA, 0.5, 1:(settings.n-1)*2)

# settings, data, model, priors
data  = TwoTypeDLModel.CountDAG(rdata, tree, settings.n)
model  = TwoTypeTree(tree, TwoTypeDL(rand(4)...), rprior);
priors = (Beta(), Beta(), Beta(), truncated(Exponential(meanμ[prefix]), 0, 10), Beta())

# set up the chain and sample
n, b = 11000, 1000  # n iterations, burnin
chn = Chain(model, priors, data, settings)
initialize!(chn, 5)
spl = sample(chn, n)
p, m = TwoTypeDLModel.post(chn, spl)

CSV.write(joinpath(pth, "chain.csv"), p)

# Extended dataframe
# ------------------
p[:,:px] = p[:,:ν] ./ (p[:,:ν] .+ p[:,:μ₂])
p[:,:t12]  = 100 .* (log(2) ./ p[:,:μ₂])
p = select(p, Not(:support))

# extinction probabilities
nnodes = length(postwalk(tree))
extpfun(x) = TwoTypeDLModel.ϕlogeny!(x.params, zeros(nnodes, 4), tree)[1,1:2]
extp = tmap(extpfun, m) |> x->permutedims(hcat(x...))
p = hcat(p, DataFrame(extp, [:extp1, :extp2]))

# condition probabilities (P one gene in each clade stemming from root)
poibfun(x) = TwoTypeDLModel.p_nonextinct_bothclades(x, settings)
p[:,:oib] = exp.(tmap(poibfun, m))

# extended chain
CSV.write(joinpath(pth, "chain.csv"), p)

# posterior prediction
N = nrow(rdata)
ppdf = ppsim(m[b:10:end], N)
CSV.write(joinpath(pth, "ppsim.csv"), ppdf)

