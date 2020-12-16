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

using Parameters, FFTW, DifferentialEquations
using NewickTree, StatsFuns, Distributions
using DataFrames, Random, StatsBase

export TwoTypeDL, RootPrior, TwoTypeTree
export Profiles, simulate, PSettings

"""
    Profiles(data::DataFrame)

Reduce a dataframe to the unique rows it contains and a count for each row.
"""
struct Profiles{T}
    data::DataFrame
    counts::Vector{T}
    n::Int
end

function Profiles(data)
    rows = sort(collect(countmap(eachrow(data))), by=x->x[2], rev=true)
    df = DataFrame(first.(rows))
    xs = vcat(last.(rows))
    Profiles(df, xs, nrow(data))
end

Base.show(io::IO, p::Profiles) = show(io, hcat(p.counts, p.data))

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
    RootPrior

A simple prior distribution for the number of genes of each type at the root of
the gene tree, assuming a geometric distribution for the total number Z, and 
assuming X₂ ~ Binomial(Z-1, 1-p), X₂ = Z - X₁. Using this law we get a marginal
geometric distribution at the root and at least one type 1 gene.
"""
struct RootPrior{T}
    η::T
    p::T
end

Base.NamedTuple(θ::RootPrior) = (η=θ.η, p=θ.p)
(θ::RootPrior)(; kwargs...) = RootPrior(merge(NamedTuple(θ), (; kwargs...))...)

function Base.rand(d::RootPrior) 
    Z = rand(Geometric(d.η)) + 1
    X2 = rand(Binomial(Z-1, 1. - d.p))
    X1 = Z - X2  # at least 1 stable gene assumed
    (X1, X2)
end

function Distributions.logpdf(d::RootPrior, X1, X2)
    X1 == 0 && return -Inf
    Z = X1 + X2
    return logpdf(Geometric(d.η), Z-1) + logpdf(Binomial(Z-1, d.p), X1-1)
end

"""
    TwoTypeTree

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

"""
    loglikelihood(m::TwoTypeTree, data, settings)

Compute the loglikelihood P(data|θ,tree) for model θ, data and a tree,
marginalized over the root prior distribution.

# Example

```julia-repl
julia> θ = TwoTypeDL(rand(4)...);

julia> tree = readnw("((A:1.2,B:1.2):0.8,C:2.0);");

julia> data = DataFrame(:A=>rand(1:5, 5), :B=>rand(1:5, 5), :C=>rand(1:5, 5));

julia> data = Profiles(data)
5×4 DataFrame
 Row │ x1     A      B      C
     │ Int64  Int64  Int64  Int64
─────┼────────────────────────────
   1 │     1      3      1      5
   2 │     1      2      3      1
   3 │     1      4      2      1
   4 │     1      4      4      5
   5 │     1      1      2      3

julia> model = TwoTypeTree(tree, θ, RootPrior(0.8, 0.5));

julia> loglikelihood(model, data)
-34.21028952032416
```
"""
Distributions.loglikelihood(m::TwoTypeTree, X, settings=PSettings()) = 
    loglhood(m, X, settings)[1]

function loglhood(m::TwoTypeTree, X, settings)
    L = prune(m.params, m.tree, X.data, settings)
    loglhood(L, m, X, settings)
end

function loglhood(L::Array{T,3}, m::TwoTypeTree, X, settings) where T
    ℓ = integrate_prior(L, m.prior)
    p = p_nonextinct_bothclades(m, settings)
    sum(ℓ .* X.counts) - X.n * p, L
end

"""
    integrate_prior(L, d::Distribution)

Integrate a prior distribution on a discrete likelihood vector/matrix. 
"""
function integrate_prior(L::Array{T,3}, d) where T
    ℓ = Vector{T}(undef, size(L, 3))
    for i=1:length(ℓ)
        @inbounds ℓ[i] = integrate_prior(L[:,:,i], d)
    end
    return ℓ
end

function integrate_prior(L::Matrix, d::RootPrior)
    ℓ = -Inf
    for i=1:size(L, 1), j=1:size(L, 2)
        ℓ = logaddexp(ℓ, L[i,j] + logpdf(d, i-1, j-1))
    end
    return ℓ
end

"""
    prune(θ::TwoTypeDL, tree, data, settings)
"""
function prune(θ::TwoTypeDL{T}, tree, data, settings=PSettings()) where T
    ndata = size(data, 1)
    function prunewalk(node)
        isleaf(node) && return data[:,name(node)] 
        L = map(children(node)) do child
            Lchild = prunewalk(child)
            Ledge = prune_edge(Lchild, θ, distance(child), settings, ndata)
        end
        return L[1] .+ L[2]
    end
    # couldn't get it type stable, although prune_edge is...
    return prunewalk(getroot(tree))::Array{T,3}
end

"""
    prune_edge(Lvs, θ, t, N, n)

Do the pruning step along an edge [u → v].
"""
function prune_edge(Lvs, θ, t, settings, ndata)
    @unpack n = settings
    Lus = fill(-Inf, n, n, ndata)
    ϕ1, ϕ2 = ϕ_fft_grid(θ, t, settings)
    Threads.@threads for j=0:n-1
        #@info "thread $(Threads.threadid())"
        for k=0:n-1  
            P = transitionp_fft(ϕ1, ϕ2, j, k, θ.μ₁ == 0.)
            _prune_edge!(Lus, Lvs, P, j, k, n) 
        end
    end
    return Lus
end
# TiledIteration? Better equipped array types?

# inner loop for pruning along internal edge
function _prune_edge!(Lus, Lvs::Array{T,3}, P, j, k, n) where T
    Pjk = (@view P[1:n, 1:n]) .+ Lvs  # (j, k) -> (l, m) matrix
    Lus[j+1, k+1, :] .= vec(mapslices(logsumexp, Pjk, dims=[1,2]))
    # lots of allocations in this last step...
end

# inner loop for edge leading to leaf, where observations are Z
function _prune_edge!(Lus, Lvs::Vector{Int}, P, j, k, n) 
    for (i,x) in enumerate(Lvs)
        @assert n > x "Observation $x exceeds bound $n, do increase `n`."
        for l=0:x
            Lus[j+1, k+1, i] = logaddexp(Lus[j+1, k+1, i], P[l+1, x-l+1])
        end
    end
end

# inner loop for edge leading to leaf, where observations are (X₁, X₂)
function _prune_edge!(Lus, Lvs::Vector{Tuple{Int,Int}}, P, j, k, n) 
    for (i,x) in enumerate(Lvs)
        @assert n > sum(x) "Observation $x exceeds bound $n, do increase `n`."
        Lus[j+1, k+1, i] = logaddexp(Lus[j+1, k+1, i], P[x[1]+1, x[2]+1])
    end
end

include("probabilities.jl")
include("simulation.jl")
include("mcmc.jl")

end # module

# There are several issues, as illustrated by a proof of principle
# implementation I did, that make the whole deal tricky:
# 
# (1) Numerical issues: to obtain transition probabilities, we need to solve a
# system of two ODEs to obtain the generating functions, evaluate them for
# complex arguments so that they become a Fourier series and so that we can
# obtain the transition probabilities using a FFT. For the FFT we need to
# truncate the state space, and the numerical accuracy is dependent on this
# truncation. 
# 
# (2) Computational burden: 
# Better data structures/loop organization in pruning algorithm! (less
# allocation!), currently most of the time is spent in doing sums in the 
# algorithm (logaddexp in prune_edge)...
#
# (3) Identifiability: given that we observe only the total number of genes
# (i.e. X₁ + X₂, not (X₁, X₂)), we run in some identifiability issues. We
# essentially have an overdetermined model, and our only hope to obtain
# reasonable paramter estimates is to use (strongly) informative priors...
# Note that with μ₁ = 0, we get a special case of the model involving only
# three parameters that is still of interest.

