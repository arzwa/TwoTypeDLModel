# Code for the main results, listed in table 1
# Arthur Zwaenepoel
using TwoTypeDLModel
using TwoTypeDLModel: BetaGeometric
using DataFrames, CSV, NewickTree, Distributions
using Serialization, StatsBase, ThreadTools
using Turing

prefix = "ygob"  # drosophila-8taxa | GO:0002376
tree   = readnw(readline("data/$prefix.nw"))
rdata  = CSV.read("data/$prefix.csv", DataFrame)
pth    = mkpath("output/table01")
root   = Dict("ygob"                => (0.98, 3.06), 
              "drosophila"          => (0.96, 3.01),
              "primates-GO:0002376" => (0.92, 2.23))

# Two-type DL model
# =================
# set root prior 
ETA, ZETA = root[prefix]
settings = PSettings(n=12, N=16, abstol=1e-6, reltol=1e-6)
rprior = TwoTypeDLModel.BBGPrior(ETA, ZETA, 0.5, 1:(settings.n-1)*2)

# settings, data, model, priors
data  = TwoTypeDLModel.CountDAG(rdata, tree, settings.n)
model  = TwoTypeTree(tree, TwoTypeDL(rand(4)...), rprior);
priors = (Beta(), Beta(), Beta(), Exponential(10.), Beta())

# set up the chain and sample
n, b = 11000, 1000  # n iterations, burnin
chn = Chain(model, priors, data, settings)
spl = sample(chn, n)
p, m = TwoTypeDLModel.post(chn, spl)

CSV.write(joinpath(pth, "$prefix-chain.csv"), p)

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
CSV.write(joinpath(pth, "$prefix-chain-ext.csv"), p)

# posterior prediction
ppdf = ppsim(m[b:10:end], N)
CSV.write(joinpath(pth, "$prefix-ppsims.csv"), ppdf)


# Single-type models 
# ==================
using DeadBird
data, bound = DeadBird.CountDAG(rdata, tree)

# function for posterior prediction
function simfun(m, N)
    X = DeadBird.simulate(m, N)[:,1:end-2]
    y = mapreduce(x->proportions(x, 0:9), hcat, eachcol(X))
    z = proportions(Matrix(X), 0:9)
    DataFrame(hcat(0:9, z, y), vcat(:k, :all, Symbol.(names(X))))
end

# 1. Fixed ShiftedBetaGeometric root prior single rate
# ----------------------------------------------------
@model shiftedbg(data, bound) = begin
    λ ~ Turing.FlatPos(0.)
    η = ETA
    ζ = ZETA 
    κ = zero(eltype(λ))
    rootp = ShiftedBetaGeometric(η, ζ)
    rates = ConstantDLG(λ=λ, μ=λ, κ=κ)
    model = PhyloBDP(rates, rootp, tree, bound)
    data ~ model
end

chain1 = sample(shiftedbg(data, bound), NUTS(), 1000) 
pdf1 = DataFrame(chain1)
CSV.write(joinpath(pth, "$prefix-st-chain1.csv"), pdf1)

function getmodels(pdf, tree, bound, η, ζ)
    map(eachrow(pdf)) do row
        rootp = ShiftedBetaGeometric(η, ζ)
        rates = ConstantDLG(λ=row[:λ], μ=row[:λ], κ=0.)
        PhyloBDP(rates, rootp, tree, bound)
    end
end

N = nrow(rdata)
models = getmodels(pdf1, tree, bound, ETA, ZETA)
ys = tmap(m->simfun(m, N), models)
CSV.write(joinpath(pth, "$prefix-st-ppsims1.csv"), vcat(ys...))


# 2. Fixed ShiftedBetaGeometric root prior
# ----------------------------------------
@model shiftedbg(data, bound) = begin
    λ ~ Turing.FlatPos(0.)
    μ ~ Turing.FlatPos(0.)
    η = ETA
    ζ = ZETA 
    κ = zero(eltype(λ))
    rootp = ShiftedBetaGeometric(η, ζ)
    rates = ConstantDLG(λ=λ, μ=μ, κ=κ)
    model = PhyloBDP(rates, rootp, tree, bound)
    data ~ model
end

chain2 = sample(shiftedbg(data, bound), NUTS(), 1000) 
pdf2 = DataFrame(chain2)
CSV.write(joinpath(pth, "$prefix-st-chain2.csv"), pdf2)

function getmodels(pdf, tree, bound, η, ζ)
    map(eachrow(pdf)) do row
        rootp = ShiftedBetaGeometric(η, ζ)
        rates = ConstantDLG(λ=row[:λ], μ=row[:μ], κ=0.)
        PhyloBDP(rates, rootp, tree, bound)
    end
end

N = nrow(rdata)
models = getmodels(pdf2, tree, bound, ETA, ZETA)
ys = tmap(m->simfun(m, N), models)
CSV.write(joinpath(pth, "$prefix-st-ppsims2.csv"), vcat(ys...))

# 3. Core gene families Beta-Geometric
# ------------------------------------
core = filter(x->all(Array(x) .> 0), rdata) .- 1
data_core, bound_core = DeadBird.CountDAG(core, tree)

@model bg(data, bound) = begin
    μ ~ Turing.FlatPos(0.)
    α ~ Beta()
    η = ETA 
    ζ = ZETA 
    λ = (1 - α) * μ 
    rootp = DeadBird.BetaGeometric(η, ζ)
    rates = ConstantDLG(λ=λ, μ=μ, κ=λ)
    model = PhyloBDP(rates, rootp, tree, bound, cond=:none)
    data ~ model
end

chain3 = sample(bg(data_core, bound_core), NUTS(), 1000) 

pdf = DataFrame(chain3)
pdf[:, :λ] = (1 .- pdf[:,:α]) .* pdf[:,:μ]
CSV.write(joinpath(pth, "$prefix-st-chain3.csv"), pdf)

function getmodels(pdf, tree, bound, η, ζ)
    map(eachrow(pdf)) do row
        rootp = BetaGeometric(η, ζ)
        rates = ConstantDLG(λ=row[:λ], μ=row[:μ], κ=row[:λ])
        PhyloBDP(rates, rootp, tree, bound, cond=:none)
    end
end

N = nrow(core)
models = getmodels(pdf, tree, bound, ETA, ZETA)
ys = tmap(m->simfun(m, N), models)
CSV.write(joinpath(pth, "$prefix-st-ppsims3.csv"), vcat(ys...))


