# A simple custom dedicated MCMC algorithm for the two-type DL model
using AdvancedMH, Bijectors

"""
    Transition

Simple Transition type.
"""
struct Transition{T,V,W}
    θ::T
    ℓ::V
    p::V
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
    hyper::Vector{Int}
end

function Chain(model, priors, data=nothing, settings=PSettings())
    t = Transition(randn(length(priors)), -Inf, -Inf)
    p = [AdaptiveProposal(Normal(0., 0.1)) for i=1:length(priors)]
    ℓ = !isnothing(data) ? lhood_fun(model, priors, data, settings) : x->0.
    Chain(t, priors, p, ℓ, length(priors) == 5 ? [5] : Int[])
end

"""
    lhood_fun

Obtain the loglikelihood function as a closure over data and algorithm settings
"""
function lhood_fun(model, priors, data, settings)
    function lhood(x)
        θ  = invlink.(priors, x)  # ℝ -> constrained
        μ₂ = θ[4]
        λ  = θ[1]*μ₂
        μ₁ = θ[2]*μ₂
        ν  = θ[3]*μ₂
        p  = length(priors) == 5 ? θ[end] : model.prior.p 
        m  = model(TwoTypeDL(λ=λ, μ₁=μ₁, ν=ν, μ₂=μ₂), model.prior(p=p))
        return loglikelihood(m, data, settings)
    end
end

# logdensity, loglhood, logprior
logdensity(chain) = chain.state.ℓ + chain.state.p
logdensity(chain, x) = loglhood(chain, x) + logprior(chain, x)
transform( chain, x) = invlink.(chain.priors, x)
loglhood(  chain, x) = chain.lhood(x)
logprior(  chain, x) = sum(logpdf_with_trans.(chain.priors, transform(chain, x), true))

"""
    mwg_sweep!(chain)

Perform a single iteration for a Metropolis-within-Gibbs algorithm.
"""
function mwg_sweep!(chain)
    for (i, p) in enumerate(chain.proposals)
        x = copy(chain.state.θ)
        x[i] += rand(p)
        #ℓ_ = i ∈ chain.hyper ? chain.state.ℓ : 
        ℓ_ = loglhood(chain, x) 
        π_ = logprior(chain, x) 
        if log(rand()) < ℓ_ + π_ - logdensity(chain)
            chain.state = Transition(x, ℓ_, π_)
            AdvancedMH.accepted!(p)
        end
        AdvancedMH.consider_adaptation!(p)
    end
    return chain.state
end

