# This is for the non-DAG data structure...
# A simple custom dedicated MCMC algorithm for the two-type DL model
# This assumes λ, μ₁, ν are expressed as fractions of μ₂.
# If we 'settle' on a model we might further 'dedicate' this, currently it's a
# bit messy... Or we should write a more generic AMWG algorithm (but its harder
# to get tricks in there lik eonly reocmputing the root vector etc.)
using AdvancedMH, Bijectors, Printf

"""
    Transition

Simple Transition type.
"""
struct Transition{T,V,W}
    θ::T
    ℓ::V
    p::V
    L::W
end

"""
    Chain(model, priors, data, settings)

A chain object bundles a loglikelihood function, some priors, proposals and a
state. `hyper` is a vector of indices storing the indices for those parameters
that do not affect the likelihood when modified while other parameters are held
fixed.
"""
mutable struct Chain{T,V,Π,P}
    state::Transition{T,V}
    priors::Π
    proposals::P
    lhood::Function
end

function Chain(model, priors, data=nothing, settings=PSettings())
    initL = Array{Float64,3}(undef, 0, 0, 0)
    t = Transition(randn(length(priors)), -Inf, -Inf, initL)
    p = [AdaptiveProposal(Normal(0., 0.5)) for i=1:length(priors)]
    ℓ = !isnothing(data) ? lhood_fun(model, priors, data, settings) : mock_lhood
    Chain(t, priors, p, ℓ)
end

Base.show(io::IO, c::Chain) = write(io, 
    (@sprintf " %8.4f" logdensity(c)), 
    join([@sprintf("%8.4f ", x) for x in c.state.θ]))

"""
    init!(chain)

Initialize a chain by doing some draws from the prior.
"""
function init!(chain::Chain, n=10)
    for i=1:n
        x = vcat(link.(chain.priors, rand.(chain.priors))...)
        ld, ℓ, p, L = logdensity(chain, x)
        ld > logdensity(chain) && (chain.state = Transition(x, ℓ, p, L))
    end
    prop = MvNormal(chain.state.θ, 0.1)
    for i=1:n
        x = rand(prop)
        ld, ℓ, p, L = logdensity(chain, x)
        ld > logdensity(chain) && (chain.state = Transition(x, ℓ, p, L))
    end
end

mock_lhood(x, L=nothing) = (0., Array{Float64,3}(undef, 0, 0, 0))

"""
    lhood_fun

Obtain the loglikelihood function as a closure over data and algorithm settings
"""
function lhood_fun(model, priors, data, settings)
    function lhood(x, L=nothing)
        m = getmodel(model, priors, x)
        return isnothing(L) ? loglhood(m, data, settings) : loglhood(L, m, data, settings)
    end
end

""" 
    getmodel(model::TwoTypeTree, priors, x)

Get a parameterized model from a real vector. 
"""
function getmodel(model, priors, x)
    θ  = invlink.(priors, x)  # ℝ -> constrained
    μ₂ = θ[4]
    λ  = θ[1]*μ₂
    μ₁ = θ[2]*μ₂
    ν  = θ[3]*μ₂
    rootprior = length(θ) > 4 ? getrootprior(model.prior, θ) : model.prior
    m  = model(TwoTypeDL(λ=λ, μ₁=μ₁, ν=ν, μ₂=μ₂), rootprior)
end 

getrootprior(p::GeometricPrior, θ) = p(r=θ[end])
getrootprior(p::BetaGeometricPrior, θ) = p(r=θ[5], n=θ[6]) 

# logdensity, loglhood, logprior
logdensity(chain) = chain.state.ℓ + chain.state.p

function logdensity(chain, x)
    ℓ, L = loglhood(chain, x)
    p = logprior(chain, x)
    ℓ + p, ℓ, p, L
end

transform(chain, x::Vector) = invlink.(chain.priors, x)
transform(chain, X::Matrix) = mapslices(x->transform(chain, x), X, dims=2)

logprior(chain, x) = sum(logpdf_with_trans.(chain.priors, transform(chain, x), true))
loglhood(chain, x) = chain.lhood(x)
loglhood(chain, x, L) = chain.lhood(x, L)

"""
    mwg_sweep!(chain)

Perform a single iteration for a Metropolis-within-Gibbs algorithm.
"""
function mwg_sweep!(chain)
    for (i, p) in enumerate(chain.proposals)
        x = copy(chain.state.θ)
        x[i] += rand(p)
        if i >= 5  # root prior paramaters
            ℓ_, L = loglhood(chain, x, chain.state.L)
        else 
            ℓ_, L = loglhood(chain, x) 
        end
        π_ = logprior(chain, x) 
        if log(rand()) < ℓ_ + π_ - logdensity(chain)
            chain.state = Transition(x, ℓ_, π_, L)
            AdvancedMH.accepted!(p)
        end
        AdvancedMH.consider_adaptation!(p)
    end
    return chain.state
end
