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
"""
module TwoTypeDLModel

using Parameters, FFTW, DifferentialEquations
using NewickTree, StatsFuns, Distributions
using DataFrames, Random, StatsBase

export TwoTypeDL, TwoTypeRootPrior, simulate

"""
    TwoTypeDL{T}

Two-type branching process model for gene family evolution by duplication and
loss. `η` is the parameter for the geometric prior on the number of lineages at
the root. See module docs for more info.
"""
@with_kw struct TwoTypeDL{T}
    λ ::T
    μ₁::T
    ν ::T
    μ₂::T
    η ::T
end

"""
    pgf_ode!(dϕ, ϕ, θ, t)

The ODE system for the two elementary probability generating functions of the
two-type DL model.
"""
function pgf_ode!(dϕ, ϕ, θ::TwoTypeDL, t)
    @unpack λ, ν, μ₁, μ₂ = θ
    dϕ[1] = μ₁ + λ*ϕ[1]*ϕ[2] - (λ + μ₁)*ϕ[1]
    dϕ[2] = μ₂ + ν*ϕ[1] + λ*ϕ[2]^2 - (λ + ν + μ₂)ϕ[2]
end

"""
    ϕ1ϕ2(θ::TwoTypeDL, s1, s2, t; kwargs_for_ODE_solver...)

Computes the probability generating functions ϕ₁(s1, s2, t) and ϕ₂(s1, s2, t)
(coefficients are P(X(t)=(j,k)|X(0)=(1,0)) and P(X(t)=(j,k)|X(0)=(0,1))).
Involves solving a system of two ODEs.
"""
function ϕ1ϕ2(θ::TwoTypeDL, s1, s2, t; kwargs...)
    ϕ0 = [s1; s2]
    ts = (0., t)
    pr = ODEProblem(pgf_ode!, ϕ0, ts, θ)
    sl = OrdinaryDiffEq.solve(pr, Tsit5(); dense=false, kwargs...)[:,end]
    (ϕ1=sl[1], ϕ2=sl[2])
end

"""
    ϕ_fft_grid(θ, t, N)

Evaluate the probability generating functions along the complex unit circle on
a N × N grid, to serve as input for a discrete fourier transform. 

!!! note:
    We use the ensemble solver from DifferentialEquations to solve the ODEs
    along the grid of initial conditions in parallel.
"""
function ϕ_fft_grid(θ, t, N)
    prob = ODEProblem(pgf_ode!, [0., 0.], (0., t), θ)
    init = [[exp(2π*im*u/N), exp(2π*im*v/N)] for u=0:N-1, v=0:N-1]
    function ensemble_probfun(prob, i, repeat)
        remake(prob, u0=init[i])
    end
    ensemble_prob = EnsembleProblem(prob, prob_func=ensemble_probfun)
    sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=N*N)
    extract_ϕ1ϕ2_solution(sim, N)
end

"""
    extract_ϕ1ϕ2_solution(sln::EnsembleSolution, N)

Extract the ODE solver results along the NxN grid.
"""
function extract_ϕ1ϕ2_solution(sln::EnsembleSolution, N)
    U  = last.(sln.u)
    ϕ1 = reshape(first.(U), N, N)
    ϕ2 = reshape(last.(U), N, N)
    (ϕ1, ϕ2)
end

"""
    transitionp_fft(ϕ1, ϕ2, j, k)

Obtain the matrix of transition probabilities P{X(t)=(l,m)|X(0)=(j,k)} using
the fast Fourier transform for the grid computed for the probability generating
functions.
"""
function transitionp_fft(ϕ1, ϕ2, j, k)
    A = (ϕ1 .^ j) .* (ϕ2 .^ k)
    dft = fft(A)
    P = real(dft) ./ size(ϕ1, 1)^2
    P[P .< 0.] .= 0.
    return log.(P)
end

"""
    log_antidiagsum(A)

Perform a `logsumexp` on the antidiagonals of a matrix.
"""
function log_antidiagsum(A)
    n = size(A, 1)
    x = fill(-Inf, 2n)
    for i=0:n-1, j=0:i
        x[i + 1] = logaddexp(x[i + 1], A[i - j + 1, j + 1])
        x[2n - i] = logaddexp(x[2n - i], A[n-i+j, n-j])
    end
    return x
end

"""
    integrate_prior(L, d::Distribution)

Integrate a prior distribution on a discrete likelihood vector/matrix.

!!! note
    Assumes the prior is defined over a domain 0:n, but we shift it by 1
    (i.e. P(X=1) = pdf(d, 0) is assumed).
"""
integrate_prior(L::Matrix, d) = integrate_prior(log_antidiagsum(L), d)
integrate_prior(L::Vector, d) = logsumexp(L[1:end] .+ logpdf(d, 0:length(L)-1))

"""
    prune_edge(Lvs, θ, t, N, n)

Do the pruning step along an edge [u → v].
"""
function prune_edge(Lvs, θ, t, N, n, ndata)
    Lus = fill(-Inf, n, n, ndata)
    ϕ1, ϕ2 = ϕ_fft_grid(θ, t, N)
    Threads.@threads for j=0:n-1
        #@info "thread $(Threads.threadid())"
        for k=0:n-1  
            P = transitionp_fft(ϕ1, ϕ2, j, k)
            _prune_edge!(Lus, Lvs, P, j, k, n) 
        end
    end
    return Lus
end

# inner loop for pruning along internal edge
function _prune_edge!(Lus, Lvs::Array{T,3}, P, j, k, n) where T
    Pjk = (@view P[1:n, 1:n]) .+ Lvs  # (j, k) -> (l, m) matrix
    Lus[j+1, k+1, :] .= vec(mapslices(logsumexp, Pjk, dims=[1,2]))
    # lots of allocations in this last step...
end

# inner loop for pruning along edge leading to leaf
function _prune_edge!(Lus, Lvs::Vector, P, j, k, n) 
    for i=1:length(Lvs)
        x = Lvs[i]
        @assert n > x "Observation $x exceeds bound $n, do increase `n`."
        for l=0:Lvs[i]
            Lus[j+1, k+1, i] = logaddexp(Lus[j+1, k+1, i], P[l+1, x-l+1])
        end
    end
end

"""
    loglikelihood(θ, data, tree; n=8, N=16)

Compute the loglikelihood P(data|θ,tree) for model θ, data and a tree. Assumes
`n` as bound on the state space and `N` as the number of discretization points
for the FFT to compute the transition probabilities. Does a recursion along the
tree. Data should be accessible like a DataFrame, i.e. counts for species `A`
should be accessible through `data[:,"A"]`.

# Example

```julia-repl
julia> θ = TwoTypeDL(rand(5)...);

julia> tree = readnw("((A:1,B:1):0.5,C:1.5);");

julia> data = DataFrame(:A=>rand(1:5, 5), :B=>rand(1:5, 5), :C=>rand(1:5, 5))
5×3 DataFrame
 Row │ A      B      C     
     │ Int64  Int64  Int64 
─────┼─────────────────────
   1 │     2      4      1
   2 │     5      5      2
   3 │     5      3      2
   4 │     5      5      1
   5 │     2      4      3

julia> l, L = loglikelihood(θ, data, tree);

julia> l
-31.53046866294061
```
"""
function loglikelihood(θ::TwoTypeDL, X, tree; n=8, N=16)
    @assert N > n "N (FFT discretization) must be larger than n (bound)"
    ndata = size(X, 1)
    function prune(node)
        isleaf(node) && return X[:,name(node)] 
        L = map(children(node)) do child
            Lchild = prune(child)
            Ledge = prune_edge(Lchild, θ, distance(child), N, n, ndata)
        end
        return L[1] .+ L[2]
    end
    Ls = prune(getroot(tree))
    ℓ  = sum(mapslices(x->integrate_prior(x, Geometric(θ.η)), Ls, dims=[1,2]))
    ℓ, Ls
end

"""
    getrates(θ::TwoTypeDL, X)

Get the duplication/loss/... rates when in state X = (X₁, X₂).
"""
getrates(p::TwoTypeDL, X) = [p.λ*X[1], p.μ₁*X[1], p.λ*X[2], p.μ₂*X[2], p.ν*X[2]]

"""
    TwoTypeRootPrior

A simple prior distribution for the number of genes of each type at the root of
the gene tree, assuming a geometric distribution for the total number Z, and 
assuming X₁ ~ Binomial(Z, p), X₂ = Z - X₁.
"""
struct TwoTypeRootPrior{T}
    η::T
    p::T
end

function Base.rand(p::TwoTypeRootPrior) 
    Z = rand(Geometric(p.η)) + 1
    X1 = rand(Binomial(Z, p.p))
    X2 = Z - X1
    (X1, X2)
end

"""
    simulate(θ::TwoTypeDL, rootprior, tree, n)

Simulate `n` family profiles from the two-type branching process model with
parameters `θ` along a tree, assuming the `rootprior` for the number of genes
at the root.

# Example

```julia-repl
julia> θ = TwoTypeDL(rand(5)...);

julia> prior = TwoTypeRootPrior(0.8, 0.9);

julia> tree = readnw("((A:1,B:1):0.5,C:1.5);");

julia> simulate(θ, prior, tree, 3)
(3×3 DataFrame
 Row │ A      B      C
     │ Int64  Int64  Int64
─────┼─────────────────────
   1 │     3      2      0
   2 │     2      1      7
   3 │     0      0      1, 3×3 DataFrame
 Row │ A       B       C
     │ Tuple…  Tuple…  Tuple…
─────┼────────────────────────
   1 │ (2, 1)  (2, 0)  (0, 0)
   2 │ (2, 0)  (0, 1)  (1, 6)
   3 │ (0, 0)  (0, 0)  (1, 0))
```
"""
function simulate(θ::TwoTypeDL, rootprior, tree, n)
    df = map(i->simulate(θ, rand(rootprior), tree), 1:n) |> DataFrame
    ddf = select(df, names(df) .=> x->first.(x) .+ last.(x))
    rename!(ddf, names(df))
    ddf, df
end

"""
    simulate(θ::TwoTypeDL, X, tree)

Simulate the two-type branching process model with parameters `θ` along a tree
`tree` with root state `X`.
"""
function simulate(p::TwoTypeDL, X, tree)
    result = Dict{Symbol,Tuple{Int,Int}}()
    function simwalk(node, X)
        _X = simulate(p, X, distance(node))
        isleaf(node) && return result[Symbol(name(node))] = _X
        for c in children(node)
            simwalk(c, _X)
        end
    end
    simwalk(tree, X)
    return (; result...)
end

"""
    simulate(θ::TwoTypeDL, X, t)

Simulate the two-tye branching process model with parameters `θ` starting from
state `X` (a tuple (X₁, X₂)) for a time period `t`.
"""
function simulate(p::TwoTypeDL, X, t::Real)
    rates = getrates(p, X)
    t -= randexp()/sum(rates)
    while t > 0.
        i = sample(1:5, Weights(rates))
        if i == 1
            X = (X[1], X[2]+1)
        elseif i == 2
            X = (X[1]-1, X[2])
        elseif i == 3
            X = (X[1], X[2]+1)
        elseif i == 4
            X = (X[1], X[2]-1)
        else
            X = (X[1]+1, X[2]-1)
        end
        rates = getrates(p, X)
        t -= randexp()/sum(rates)
    end
    return X
end

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

