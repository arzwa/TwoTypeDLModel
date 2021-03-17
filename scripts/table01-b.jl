# Code for the main results, listed in table 1, single-type models
# Author: Arthur Zwaenepoel
using DataFrames, CSV, NewickTree, Distributions
using Serialization, StatsBase, ThreadTools
using DeadBird, Turing

prefix = "ygob"  # drosophila-8taxa | GO:0002376
suffix = string(rand())[3:6]  # terrible unique suffix
tree   = readnw(readline("data/$prefix.nw"))
rdata  = CSV.read("data/$prefix.csv", DataFrame)
pth    = mkpath("output/table01-b/$prefix")
root   = Dict("ygob"                => (0.98, 3.06), 
              "drosophila"          => (0.96, 3.01),
              "primates-GO:0002376" => (0.92, 2.23))
#              prefix               => (η   , ζ-1 )

# Single-type models 
# ==================
ETA, ZETA = root[prefix]
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
CSV.write(joinpath(pth, "chain1.csv"), pdf1)

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
CSV.write(joinpath(pth, "ppsim1.csv"), vcat(ys...))


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
CSV.write(joinpath(pth, "chain2.csv"), pdf2)

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
CSV.write(joinpath(pth, "ppsim2.csv"), vcat(ys...))

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
CSV.write(joinpath(pth, "chain3.csv"), pdf)

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
CSV.write(joinpath(pth, "ppsim3.csv"), vcat(ys...))


