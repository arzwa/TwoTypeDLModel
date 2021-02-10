# Code for conducting simulations as shown in figure 3 
using TwoTypeDLModel
using DataFrames, CSV, NewickTree
using Serialization, Printf

# Simulation and inference for two-type model
# ===========================================
const ETA = 0.95
const ZETA = 4.0 - 1  # recall we offset by 1 here
tree = readnw(readline("data/drosophila-8taxa.nw"))
priors = (Beta(), Beta(), Beta(), Exponential(5.), Beta(), Exponential(3.))
settings = PSettings(n=12, N=16, abstol=1e-6, reltol=1e-6)

function dosim(tree, eta, N)
    μ₂ = 5.
    μ₁ = 0.1
    λ  = 0.2
    ν  = 0.2
    ζ  = ZETA   
    η  = ETA
    r  = 0.5
    rootprior = TwoTypeDLModel.BBGPrior(η, ζ, r, 1:9)
    rates = TwoTypeDL(μ₂=μ₂, μ₁=μ₁, λ=λ, ν=ν)
    model = TwoTypeTree(tree, rates, rootprior)
    @info "" model
    c = TwoTypeDLModel.default_condition(tree)
    condition = TwoTypeDLModel.default_condition(tree, 10)
    X, Y = TwoTypeDLModel.simulate(model, N, condition)
    (X=X, Y=Y, model=model)
end

# N, n = 10_000, 11_000  # number of families, number of samples
N, n = 100, 110
sim = dosim(tree, eta, N)
@info maximum(Matrix(sim.X))

pth = mkpath("output/fig03")
serialize(joinpath(pth, "sim.jls"), sim)

# inference from incompletely observed data
chn = Chain(sim.model, priors, CountDAG(sim.X, tree, settings.n), settings) 
spl = sample(chn, n)
p, m = TwoTypeDLModel.post(chn, spl);
CSV.write(joinpath(pth, "sim-X.chain.csv"), p)

# inference from completely observed data
chn = Chain(sim.model, priors, CountDAG(sim.Y, tree, settings.n), settings) 
spl = sample(chn, n)
p, m = TwoTypeDLModel.post(chn, spl);
CSV.write(joinpath(pth, "sim-Y.chain.csv"), p)


# Inference for single-type models
# ================================
using DeadBird
data, bound = DeadBird.CountDAG(sim.X, tree)

# (1) Fixed ShiftedBetaGeometric root prior
# -----------------------------------------
@model shiftedbg(data, bound) = begin
    λ ~ Turing.FlatPos(0.)
    μ ~ Turing.FlatPos(0.)
    ζ ~ Exponential(ZETA) 
    η = zero(eltype(ζ)) + ETA 
    κ = zero(eltype(λ))
    rootp = ShiftedBetaGeometric(η, ζ)
    rates = ConstantDLG(λ=λ, μ=μ, κ=κ)
    model = PhyloBDP(rates, rootp, tree, bound)
    data ~ model
end

chain1 = sample(shiftedbg(data, bound), NUTS(), 1000) 

# (2) With Fixed ζ
# ----------------
@model shiftedbgfζ(data, bound) = begin
    λ ~ Turing.FlatPos(0.)
    μ ~ Turing.FlatPos(0.)
    ζ = ZETA 
    η = zero(eltype(ζ)) + ETA
    κ = zero(eltype(λ))
    rootp = ShiftedBetaGeometric(η, ζ)
    rates = ConstantDLG(λ=λ, μ=μ, κ=κ)
    model = PhyloBDP(rates, rootp, tree, bound)
    data ~ model
end

chain2 = sample(shiftedbgfζ(data, bound), NUTS(), 1000) 

# (3) Core (non-extinct) gene families Beta-Geometric
# ---------------------------------------------------
core = filter(x->all(Array(x) .> 0), rdata) .- 1
data_core, bound_core = DeadBird.CountDAG(core, tree)

@model bg(data, bound) = begin
    μ ~ Turing.FlatPos(0.)
    α ~ Beta()
    ζ ~ Exponential(ZETA) 
    η = zero(eltype(ζ)) + ETA
    λ = (1 - α) * μ 
    rootp = DeadBird.BetaGeometric(η, ζ)
    rates = ConstantDLG(λ=λ, μ=μ, κ=λ)
    model = PhyloBDP(rates, rootp, tree, bound, cond=:none)
    data ~ model
end

chain3 = sample(bg(data_core, bound_core), NUTS(), 1000) 

pdf3 = DataFrame(chain3)
pdf3[:,:λ] = (1. .- pdf3[:,:α]) .* pdf3[:,:μ]
chain3 = Chains(Matrix(pdf3[:,[:λ,:μ,:α,:ζ]]), names(pdf3[:,[:λ,:μ,:α,:ζ]]))

# (4) Core gene families Beta-Geometric, fix ζ
# --------------------------------------------
@model bgfζ(data, bound) = begin
    μ ~ Turing.FlatPos(0.)
    α ~ Beta()
    ζ = ZETA
    η = zero(eltype(ζ)) + ETA
    λ = (1 - α) * μ 
    rootp = DeadBird.BetaGeometric(η, ζ)
    rates = ConstantDLG(λ=λ, μ=μ, κ=λ)
    model = PhyloBDP(rates, rootp, tree, bound, cond=:none)
    data ~ model
end

chain4 = sample(bgfζ(data_core, bound_core), NUTS(), 1000) 

pdf4 = DataFrame(chain4)
pdf4[:,:λ] = (1. .- pdf4[:,:α]) .* pdf4[:,:μ]
chain4 = Chains(Matrix(pdf4[:,[:λ,:μ,:α]]), names(pdf4[:,[:λ,:μ,:α]]))

serialize(joinpath(pth, "st-chains"), (chain1, chain2, chain3, chain4))
