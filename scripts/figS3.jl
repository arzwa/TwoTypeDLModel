# Code for simulations shown in figure S3
# This involves simulating from the DLF model, and performing inference
# using the two-type model
using TwoTypeDLModel
using TwoTypeDLModel: IdealTwoTypeDL, IdealModelPrior, BBGPrior
using DataFrames, CSV, NewickTree, Distributions
using Serialization, Printf

const ETA = 0.95
const ZETA = 4.0 - 1
tree = readnw(readline("data/drosophila.nw"))
priors = (Beta(), Beta(), Beta(), Exponential(5.), Beta(), Exponential(3.))
settings = PSettings(n=12, N=16, abstol=1e-6, reltol=1e-6)

function dosim(tree, eta, N)
    μ₂ = 3.
    μ₁ = 0.1
    λ  = 0.2
    ν  = 0.2
    η  = ETA
    ζ  = ZETA   # recall we offset by 1 here
    r  = 0.5
    rootprior = IdealModelPrior(BBGPrior(ETA, ZETA, r, 1:9))
    rates = IdealTwoTypeDL(μ₂=μ₂, μ₁=μ₁, λ=λ, ν=ν)
    simmodel = TwoTypeTree(tree, rates, rootprior)
    c = TwoTypeDLModel.default_condition(tree)
    condition = TwoTypeDLModel.default_condition(tree, 10)
    X, Y = TwoTypeDLModel.simulate(simmodel, N, condition)
    infmodel = TwoTypeTree(tree, TwoTypeDL(μ₂=μ₂, μ₁=μ₁, λ=λ, ν=ν), rootprior.p)   
    (X=X, Y=Y, simmodel=simmodel, infmodel=infmodel)
end

#N, n, b = 10_000, 11_000, 1000  # n families, n samples, burnin
N, n, b = 100, 110, 10  # n families, n samples, burnin
sim = dosim(tree, eta, N)
@info maximum(Matrix(sim.X))

pth = mkpath("output/figS3")
serialize(joinpath(pth, "sim.jls"), sim)

# inference for two-type model
chn = Chain(sim.infmodel, priors, CountDAG(sim.X, tree, settings.n), settings) 
spl = sample(chn, n)
p, m = TwoTypeDLModel.post(chn, spl);
CSV.write(joinpath(pth, "sim-X.chain.csv"), p)

# posterior prediction
ypp = ppsim(m[b:10:end], N)
CSV.write(joinpath(pth, "ppsims.csv"), ypp)

# inference for single-type model
using DeadBird
data, bound = DeadBird.CountDAG(sim.X, tree)

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
CSV.write(joinpath(pth, "sim-X.st.chain.csv"), DataFrame(chain1))
