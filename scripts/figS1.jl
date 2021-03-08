# Code for conducting simulations as shown in figure S1 
# Author: Arthur Zwaenepoel
using Pkg; Pkg.activate("..")
using TwoTypeDLModel
Pkg.activate(".")
using DataFrames, CSV, NewickTree, Distributions
using Serialization, Printf

# Simulation and inference for two-type model
# ===========================================
tree = readnw(readline("data/drosophila.nw"))
priors = (Beta(), Beta(), Beta(), Exponential(5.), Beta(), Exponential(3.))
settings = PSettings(n=12, N=16, abstol=1e-6, reltol=1e-6)

function dosim(tree, N)
    ζ  = rand(LogNormal(log(3.), 0.2)) 
    η  = rand(Beta(9,1))
    μ₂ = rand(Uniform(1,10))
    a  = rand(Beta(1,9), 3)
    λ, μ₁, ν = a .* μ₂ 
    r  = rand(Beta(1,4))
    rootprior = TwoTypeDLModel.BBGPrior(η, ζ, r, 1:9)
    rates = TwoTypeDL(μ₂=μ₂, μ₁=μ₁, λ=λ, ν=ν)
    model = TwoTypeTree(tree, rates, rootprior)
    @info "" model
    condition = TwoTypeDLModel.default_condition(tree, 10)
    X, Y = TwoTypeDLModel.simulate(model, N, condition)
    (X=X, Y=Y, model=model, θ=(ζ=ζ, η=η, μ₂=μ₂, λ=λ, μ₁=μ₁, ν=ν, r=r))
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
pth = mkpath("output/fig03c/$N-$pre")
@info pth
serialize(joinpath(pth, "sim.jls"), sim)

# inference from incompletely observed data
chn = Chain(sim.model, priors, CountDAG(sim.X, tree, settings.n), settings) 
TwoTypeDLModel.initialize!(chn)
spl = sample(chn, n)
p, m = TwoTypeDLModel.post(chn, spl);
p = addtodf(p, sim.θ)
CSV.write(joinpath(pth, "sim-X.chain.csv"), p)

# inference from completely observed data
chn = Chain(sim.model, priors, CountDAG(sim.Y, tree, settings.n), settings) 
TwoTypeDLModel.initialize!(chn)
spl = sample(chn, n)
p, m = TwoTypeDLModel.post(chn, spl);
p = addtodf(p, sim.θ)
CSV.write(joinpath(pth, "sim-Y.chain.csv"), p)


