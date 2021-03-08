# Code for conducting simulations as shown in figure 4 
# Author: Arthur Zwaenepoel
using Pkg; Pkg.activate("..")
using TwoTypeDLModel
import TwoTypeDLModel: IdealTwoTypeDL, IdealModelPrior, BBGPrior
Pkg.activate(".")
using DataFrames, CSV, NewickTree, Distributions
using Serialization, Printf

# Simulation for DLF model
# ========================
tree = readnw(readline("data/drosophila.nw"))
priors = (Beta(), Beta(), Beta(), Exponential(5.), Beta(), Exponential(3.))
settings = PSettings(n=12, N=16, abstol=1e-6, reltol=1e-6)

function dosim(tree, N)
    ζ  = rand(LogNormal(log(3.), 0.2)) 
    η  = rand(Beta(9,1))
    μ₂ = rand(Uniform(1,5))
    a  = rand(Beta(1,9), 3)
    λ, μ₁, ν = a .* μ₂ 
    r  = rand(Beta(1,4))
    rootprior = IdealModelPrior(BBGPrior(η, ζ, r, 1:9))
    rates = IdealTwoTypeDL(μ₂=μ₂, μ₁=μ₁, λ=λ, ν=ν)
    simmodel = TwoTypeTree(tree, rates, rootprior)
    c = TwoTypeDLModel.default_condition(tree)
    condition = TwoTypeDLModel.default_condition(tree, 10)
    X, Y = TwoTypeDLModel.simulate(simmodel, N, condition)
    infmodel = TwoTypeTree(tree, TwoTypeDL(μ₂=μ₂, μ₁=μ₁, λ=λ, ν=ν), rootprior.p)   
    (X=X, Y=Y, simmodel=simmodel, model=infmodel, 
     θ=(ζ=ζ, η=η, μ₂=μ₂, λ=λ, μ₁=μ₁, ν=ν, r=r))
end

function addtodf(df, θ)
    for (k, v) in pairs(θ) 
        kk = Symbol("$(string(k))_true")
        df[:,kk] .= v
    end
    return df
end

# number of families, number of samples
N, n = 1000, 11_000
sim = dosim(tree, N)
@info maximum(Matrix(sim.X))

pre = string(rand())[3:6]
pth = mkpath("output/figS3b/$N-$pre")
@info pth
serialize(joinpath(pth, "sim.jls"), sim)

# inference under Two-Type DL model from incompletely observed data
chn = Chain(sim.model, priors, CountDAG(sim.X, tree, settings.n), settings) 
TwoTypeDLModel.initialize!(chn)
spl = sample(chn, n)
p, m = TwoTypeDLModel.post(chn, spl);
p = addtodf(p, sim.θ)
CSV.write(joinpath(pth, "sim-X.chain.csv"), p)

