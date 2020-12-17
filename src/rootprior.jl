"""
    GeometricPrior

A simple prior distribution for the number of genes of each type at the root of
the gene tree, assuming a geometric distribution for the total number Z, and 
assuming X₂ ~ Binomial(Z-1, 1-r), X₂ = Z - X₁. Using this law we get a marginal
geometric distribution at the root and at least one type 1 gene.
"""
struct GeometricPrior{T} <: RootPrior{T}
    η::T
    r::T
end

Base.NamedTuple(θ::GeometricPrior) = (η=θ.η, r=θ.r)
(θ::GeometricPrior)(; kwargs...) = GeometricPrior(merge(NamedTuple(θ), (; kwargs...))...)

function Base.rand(d::GeometricPrior) 
    Z = rand(Geometric(d.η)) + 1
    X2 = rand(Binomial(Z-1, 1. - d.r))
    X1 = Z - X2  # at least 1 stable gene assumed
    (X1, X2)
end

function Distributions.logpdf(d::GeometricPrior, X1, X2)
    X1 == 0 && return -Inf
    Z = X1 + X2
    return logpdf(Geometric(d.η), Z-1) + logpdf(Binomial(Z-1, d.r), X1-1)
end

"""
    BetaGeometricPrior
"""
struct BetaGeometricPrior{T} <: RootPrior{T}
    m::T  # mean
    n::T  # 'sample size'
    r::T  
end

Base.NamedTuple(θ::BetaGeometricPrior) = (m=θ.m, n=θ.n, r=θ.r)
(θ::BetaGeometricPrior)(; kwargs...) = BetaGeometricPrior(merge(NamedTuple(θ), (; kwargs...))...)
getαβ(d::BetaGeometricPrior) = d.m * (1 + d.n), (1 - d.m) * (1 + d.n)

function Base.rand(d::BetaGeometricPrior)
    α, β = getαβ(d)
    p = rand(Beta(α, β))
    p = p <= zero(p) ? 1e-7 : p >= one(p) ? 1-1e-7 : p
    Z = rand(Geometric(p))
    X2 = rand(Binomial(Z, 1. - d.r))
    X1 = Z - X2 + 1
    (X1, X2)
end

function Distributions.logpdf(d::BetaGeometricPrior, X1, X2)
    X1 == 0 && return -Inf
    Z = X1 + X2
    α, β = getαβ(d)
    lp = logbeta(α + 1, β + Z - 1) - logbeta(α, β) 
    return lp + logpdf(Binomial(Z-1, d.r), X1-1)
end
