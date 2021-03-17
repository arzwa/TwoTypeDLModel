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
struct GeometricPrior{T} <: RootPrior
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
The Beta distribution is not parameterized by α and β however, but by the mean
`η = α / (α + β)` and 'sample size' `ζ = α + β - 1`.

!!! warn
    We assume `ζ` is offset by one to ensure the distribution behaves reasonably.
    Please provide the value of `α + β - 1`.
"""
struct BetaGeometricPrior{T} <: RootPrior
    η::T  # mean
    ζ::T  # 'sample size' - 1
    r::T  
end

Base.NamedTuple(θ::BetaGeometricPrior) = (η=θ.η, ζ=θ.ζ, r=θ.r)
(θ::BetaGeometricPrior)(; kwargs...) = BetaGeometricPrior(merge(NamedTuple(θ), (; kwargs...))...)
getαβ(d::BetaGeometricPrior) = d.η * (1 + d.ζ), (1 - d.η) * (1 + d.ζ)

function Base.rand(d::BetaGeometricPrior)
    α, β = getαβ(d)
    η = rand(Beta(α, β))
    η = η <= zero(η) ? 1e-16 : η >= one(η) ? 1-1e-16 : η
    Zm1 = rand(Geometric(η))  # this is Z - 1
    X2 = rand(Binomial(Zm1, 1. - d.r))
    X1 = Zm1 + 1 - X2
    (X1, X2)
end

function Distributions.logpdf(d::BetaGeometricPrior, X1, X2)
    X1 == 0 && return -Inf
    Z = X1 + X2
    α, β = getαβ(d)
    lp = logbeta(α + 1, β + Z - 1) - logbeta(α, β) 
    return lp + logpdf(Binomial(Z-1, d.r), X1-1)
end

"""
    BoundedBetaGeometricPrior

Often this is preferred over the BetaGeometricPrior I think, since often we 
restrict the domain, and the unrestricted Beta-Geometric distribution can
give real large samples.

!!! warn
    We assume `ζ` is offset by one to ensure the distribution behaves reasonably.
    Please provide the value of `α + β - 1`.
"""
struct BoundedBetaGeometricPrior{T,D} <: RootPrior
    η::T
    ζ::T
    r::T
    d::D
end

function BoundedBetaGeometricPrior(η::T, ζ::T, r::T, support::D) where {T,D<:UnitRange}
    α = η * (1 + ζ)
    β = (1 - η) * (1 + ζ)
    p = exp.([logbeta(α + 1, β + Z - 1) - logbeta(α, β) for Z=support])
    d = DiscreteNonParametric(support, p ./ sum(p))
    BoundedBetaGeometricPrior(η, ζ, r, d)
end

const BBGPrior{T,D} = BoundedBetaGeometricPrior{T,D} where {T,D}

Base.NamedTuple(θ::BoundedBetaGeometricPrior) = (η=θ.η, ζ=θ.ζ, r=θ.r, support=θ.d.support)
(θ::BoundedBetaGeometricPrior)(; kwargs...) = BBGPrior(merge(NamedTuple(θ), (; kwargs...))...)

function Base.rand(d::BoundedBetaGeometricPrior)
    Z  = rand(d.d)
    X2 = rand(Binomial(Z - 1, 1. - d.r))
    X1 = Z - X2 
    (X1, X2)
end

function Distributions.logpdf(d::BoundedBetaGeometricPrior, X1, X2)
    X1 == 0 && return -Inf
    Z = X1 + X2
    return logpdf(d.d, Z) + logpdf(Binomial(Z-1, d.r), X1-1)
end

# wrapper for the IdealTwoTypeDL model
struct IdealModelPrior{P} <: RootPrior
    p::P
end

function Base.rand(d::IdealModelPrior)
    X1, X2 = rand(d.p)
    x = ones(Int, X1)
    for i=1:X2
        x[rand(1:X1)] += 1
    end
    return x
end

"""
    BetaGeometric(η, ζ)

A beta-geometric distribution on the domain (1,2,3,...).  Note that `ζ` is not
offset by one.
"""
struct BetaGeometric{T} <: DiscreteUnivariateDistribution 
    η::T
    ζ::T
    α::T
    β::T
    BetaGeometric(η::T, ζ::T) where T = new{T}(η, ζ, getαβ(η, ζ)...)
end

getαβ(η, ζ) = η * ζ, (1 - η) * ζ
BetaGeometric(η, ζ) = BetaGeometric(promote(η, ζ)...)

function Base.rand(rng::AbstractRNG, d::BetaGeometric)
    p = rand(Beta(d.α, d.β))
    p = p <= zero(p) ? 1e-16 : p >= one(p) ? 1-1e-16 : p
    return rand(Geometric(p))
end

Base.rand(rng::AbstractRNG, d::BetaGeometric, n::Int) = map(rand(rng, d), 1:n)
Distributions.logpdf(d::BetaGeometric, k::Int) = 
    logbeta(d.α + 1, d.β + k - 1) - logbeta(d.α, d.β)

"""
   loglikelihood(d::BetaGeometric, ks::Vector{Int})

Loglikelihod for a vector of counts `ks`, i.e. `[x1, x2, x3, ...]` where `x1`
is the number of times k=1 is observed in the data, `x2` the number of times
k=2 is observed, etc.
"""
function Distributions.loglikelihood(d::BetaGeometric, ks::Vector{Int})
    logp = 0.
    for (k, count) in enumerate(ks)
        logp += count * logpdf(d, k)
    end
    return logp
end
