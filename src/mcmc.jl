# For the DAG
using AdvancedMH, Bijectors, Printf

# Note: we still need to keep the root state matrices in the transition if we
# want to skip recomputing the whole PGM upon changing η/n, this is because the
# DAG is modified each iteration of the chain, but not necessarily accepted.
"""
    Transition

Simple Transition type.
"""
struct Transition{T}
    θ::Vector{T}
    ℓ::T
    p::T
    L::Array{T,3}
end

"""
    Chain(model, priors, data, settings)

A chain object bundles a loglikelihood function, some priors, proposals and a
state. `hyper` is a vector of indices storing the indices for those parameters
that do not affect the likelihood when modified while other parameters are held
fixed.
"""
mutable struct Chain{T,Π,P}
    priors::Π
    proposals::P
    state::Transition{T}
    mfun::Function
    ℓfun::Function
    ℓroot::Function
end

function Chain(model, priors, data=nothing, settings=PSettings())
    t = Transition(randn(length(priors)), -Inf, -Inf, zeros(1, 1, 1))
    p = [AdaptiveProposal(Normal(0., 0.5)) for i=1:length(priors)]
    mfun  = x->getmodel(model, priors, x)
    ℓfun  = isnothing(data) ? x->(0.,t.L) : x->loglhood(mfun(x), data, settings)
    ℓroot = isnothing(data) ? (x,L)->(0.,L) : (x,L)->loglhoodroot(mfun(x), data, L, settings)
    Chain(priors, p, t, mfun, ℓfun, ℓroot)
end

"""
    initialize!(chain, ntry=10)

Initialize the chain in a reasonable way (informed by the model).  This will
sample `μ₂` from its prior, and set `μ₁, λ, ν` each to `(1 - η)*μ₂`. We can do
this several times and start from the position with highest log-density.

!!! note
    While I prefer completely random draws from the prior to initialize, I
    think this is a sensible way to proceed, since we fix the root prior to a
    certain value of `η` which *shoul* approximately correspond to `1 - λ/μ₂`.
    Using this initialization procedure, we get random draws for μ₂, but
    restrict the other values to conform to the expectation under `η`.
"""
function initialize!(chain::Chain, ntry=10)
    best = chain.state
    for i=1:ntry
        θ = vcat(rand.(chain.priors)...)  # draw from prior
        η = chain.mfun.model.prior.η  # the η value is fixed for the root
        θ[1:3] .= (1 - η)  # ≈ λ/μ₂
        y = TwoTypeDLModel.link.(chain.priors, θ)  # to ℝ
        y[1:3] .+= randn(3)  # some variation on the ratios
        ℓ, L = chain.ℓfun(y)
        p = TwoTypeDLModel.logprior(chain, y)
        @info "" (ℓ + p) θ
        ℓ + p > (best.ℓ + best.p) && (best = TwoTypeDLModel.Transition(y, ℓ, p, L))
    end
    chain.state = best
end

function Base.show(io::IO, c::Chain) 
    s = @sprintf(" %8.4f", logdensity(c))
    s *= join([@sprintf("%8.4f ", x) for x in c.state.θ])
    write(io, s, "\n")
end

function loglhood(model::TwoTypeTree, dag::CountDAG, settings)
    ℓ = loglikelihood(model, dag, settings) 
    L = dag.parts[dag.nodes[1],:,:,1]
    return ℓ, L
end

function loglhoodroot(model, dag, L, settings)
    ℓ = _loglhoodroot(model, dag, L, settings)
    return ℓ, L
end

function getmodel(model, priors, x)
    θ  = invlink.(priors, x)  # ℝ -> constrained
    μ₂ = θ[4]
    λ  = θ[1]*μ₂
    μ₁ = θ[2]*μ₂
    ν  = θ[3]*μ₂
    rootprior = length(θ) > 4 ? getrootprior(model.prior, θ[5:end]) : model.prior
    return model(TwoTypeDL(λ=λ, μ₁=μ₁, ν=ν, μ₂=μ₂), rootprior)
end 

getrootprior(p::GeometricPrior, θ) = p(r=θ[1])
getrootprior(p::Union{BetaGeometricPrior,BBGPrior}, θ) = 
    length(θ) > 1 ? p(r=θ[1], ζ=θ[2]) : p(r=θ[1])

# logdensity, loglhood, logprior
logdensity(chain) = chain.state.ℓ + chain.state.p
logprior(c, x) = sum(logpdf_with_trans.(c.priors, transform(c, x), true))
transform(c, x::Vector) = invlink.(c.priors, x)

"""
    mwg_sweep!(chain)

Perform a single iteration for a Metropolis-within-Gibbs algorithm.
"""
function mwg_sweep!(chain)
    for (i, p) in enumerate(chain.proposals)
        x = deepcopy(chain.state.θ)
        x[i] += rand(p)
        if i < 5  # process parameters
            ℓ_, L_ = chain.ℓfun(x) 
        else      # root prior parameters
            ℓ_, L_ = chain.ℓroot(x, chain.state.L)
        end
        π_ = logprior(chain, x) 
        if log(rand()) < ℓ_ + π_ - logdensity(chain)
            chain.state = Transition(x, ℓ_, π_, L_)
            AdvancedMH.accepted!(p)
        end
        AdvancedMH.consider_adaptation!(p)
    end
    return chain.state
end

"""
    sample(chain, n)

Take `n` samples from the chain, returns intermediate results when interrupted.
"""
function StatsBase.sample(chain, n; progress=true)
    i = 0
    samples = typeof(chain.state)[]
    while i <= n
        i += 1
        try
            progress && (@printf "%10d " i)
            x = mwg_sweep!(chain)
            push!(samples, x)
            progress && (println(join([@sprintf("%7.4f", xi) for xi in x.θ], " ")))
        catch e
            @info "Sampler interrupted" e
            return samples
        end
    end
    return samples
end

"""
    post(chain, samples)

Get posterior distribution for parameters of interest and the model objects for
posterior prediction.
"""
function post(chain, samples)
    θ = getfield.(samples, :θ)
    models = chain.mfun.(θ)
    pdf = map(zip(models, samples)) do (m, s)
        merge(NamedTuple(m.params), NamedTuple(m.prior), (ℓ=s.ℓ, p=s.p))
    end
    (posterior=DataFrame(pdf), models=models)
end

function models_from_post(pdf, tree, rprior)
    map(eachrow(pdf)) do row
        r = TwoTypeDL(λ=row[:λ], μ₁=row[:μ₁], ν=row[:ν], μ₂=row[:μ₂])
        p = get_rprior(rprior, row)
        TwoTypeTree(tree, r, p)
    end
end

get_rprior(rprior::BBGPrior, row) = BBGPrior(row[:η], row[:ζ], row[:r], rprior.d.support)
