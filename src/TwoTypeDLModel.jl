# Arthur Zwaenepoel 2020 <arzwa@psb.vib-ugent.be>
"""
    TwoTypeDLModel

Inference for a two-type branching process model of gene family evolution by
duplication and loss. The main aim is more realistic estimation of gene
duplication and loss rates from gene family data. Inspired by the work of Xu
et al. on the BDS process.

The model is a two-type continuous-time branching process with following 
stochastic dynamics

```
1 → 12   with rate λ
1 →  ∅   with rate μ₁
2 → 22   with rate λ
2 →  ∅   with rate μ₂ > μ₁ and μ₂ > λ + ν
2 →  1   with rate ν
```

A special case of this model that may be of interest is the `μ₁ = 0.` case.

Here a type 1 gene denotes a non-redundant gene in a type 2 gene a redundant 
duplicated gene. The most important aspect of the model type 2 genes get lost 
at a higher rate than type 1 genes, and that type 2 genes can get established
(capturing the processes of sub- and neofunctionalization) and become stably
incorporated type 1 genes.

Throughout we assume a Geometric prior is specified for the number of genes
at the root of the species tree for each gene family, and we assume at least
one 'stable' (type 1) gene is among these.
"""
module TwoTypeDLModel

using FFTW, DifferentialEquations, Random, SpecialFunctions, LightGraphs
using Parameters, DataFrames, NewickTree, StatsBase, StatsFuns, Distributions
using ThreadTools

export TwoTypeDL, GeometricPrior, BetaGeometricPrior, TwoTypeTree
export CountDAG, Profiles, simulate, PSettings, ppsim, Chain, sample
export initialize!

abstract type RootPrior end

"""
    TwoTypeDL{T}

Two-type branching process model for gene family evolution by duplication and
loss.

To reparameterize an existing TwoTypeDL struct, one can use the struct as a 
function.

# Example
```
julia> θ1 = TwoTypeDL(rand(5)...);

julia> θ2 = θ1(η=0.8, λ=0.2)
TwoTypeDL{Float64}
  λ: Float64 0.2
  μ₁: Float64 0.675312373529684
  ν: Float64 0.6090749991263547
  μ₂: Float64 0.40013553628911924
```
"""
@with_kw struct TwoTypeDL{T}
    λ ::T
    μ₁::T
    ν ::T
    μ₂::T
end

# Two useful functions
Base.NamedTuple(θ::TwoTypeDL) = (λ=θ.λ, μ₁=θ.μ₁, ν=θ.ν, μ₂=θ.μ₂)
(θ::TwoTypeDL)(; kwargs...) = TwoTypeDL(merge(NamedTuple(θ), (; kwargs...))...)

"""
    IdealTwoTypeDL{T}

An ideal two-type duplication loss model, where loss rates are dependent on the
number of *redundant* genes (and not the number of *excess* genes). Inference
is not defined for the this model, but we can simulate from it.
"""
@with_kw struct IdealTwoTypeDL{T}
    λ ::T
    μ₁::T
    ν ::T
    μ₂::T
end

"""
    TwoTypeTree(tree, θ, prior)

This bundles a parameterization, tree and prior into a single model object that
can be used to compute the marginal likelihood.
"""
struct TwoTypeTree{T1,T2,T3}
    tree  ::T1
    params::T2
    prior ::T3
end

(m::TwoTypeTree)(θ::TwoTypeDL) = TwoTypeTree(m.tree, θ, m.prior)
(m::TwoTypeTree)(θ::RootPrior) = TwoTypeTree(m.tree, m.params, θ)
(m::TwoTypeTree)(a, b) = TwoTypeTree(m.tree, a, b)

# includes
include("probabilities.jl")
include("rootprior.jl")
include("profile.jl")
include("dag.jl")
include("simulation.jl")
include("mcmc.jl")

end # module

