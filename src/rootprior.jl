"""
    integrate_prior(L, d::Distribution)

Integrate a prior distribution along the matrix of likelihood values for the
two type process.
"""
function integrate_prior(L::Array{T,3}, d) where T
    ℓ = Vector{T}(undef, size(L, 3))
    for i=1:length(ℓ)
        @inbounds ℓ[i] = integrate_prior(L[:,:,i], d)
    end
    return ℓ
end

function integrate_prior(L::Matrix, d)
    ℓ = -Inf
    for i=1:size(L, 1), j=1:size(L, 2)
        ℓ = logaddexp(ℓ, L[i,j] + logpdf(d, i-1, j-1))
    end
    return ℓ
end

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
    Z = rand(Geometric(d.η))
    X2 = rand(Binomial(Z, 1. - d.r))
    X1 = Z - X2 + 1  # at least 1 stable gene assumed
    (X1, X2)
end

function Distributions.logpdf(d::GeometricPrior, X1, X2)
    X1 == 0 && return -Inf
    Z = X1 + X2
    return logpdf(Geometric(d.η), Z-1) + logpdf(Binomial(Z-1, d.r), X1-1)
end

"""
    BetaGeometricPrior

A Beta-Geometric prior distribution for the two-type process. The total number
of genes at the root Z ~ BetaGeometric(α, β), and X₂ ~ Binomial(Z-1, 1-r).
The Beta distribution is no parameterized by α and β however, but by the mean
`m = α / (α + β)` and 'sample size' `n = α + β`. However, we assume n is offset
by one to ensure the distribution is proper (?). So the `n` in this struct 
should be `α + β - 1`.
"""
struct BetaGeometricPrior{T} <: RootPrior{T}
    m::T  # mean
    n::T  # 'sample size' - 1
    r::T  
end

Base.NamedTuple(θ::BetaGeometricPrior) = (m=θ.m, n=θ.n, r=θ.r)
(θ::BetaGeometricPrior)(; kwargs...) = BetaGeometricPrior(merge(NamedTuple(θ), (; kwargs...))...)
getαβ(d::BetaGeometricPrior) = d.m * (1 + d.n), (1 - d.m) * (1 + d.n)

function Base.rand(d::BetaGeometricPrior)
    α, β = getαβ(d)
    η = rand(Beta(α, β))
    η = η <= zero(η) ? 1e-7 : η >= one(η) ? 1-1e-7 : η
    Z = rand(Geometric(η))
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

