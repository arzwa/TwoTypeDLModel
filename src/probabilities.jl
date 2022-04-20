"""
    PSettings

Store settings for computing approximate transition probabilities.
"""
@with_kw struct PSettings
    """Grid size"""
    N::Int = 24
    """State truncation bound (1D)"""
    n::Int = 12
    """`abstol` setting for ODE solver"""
    abstol::Float64 = 1e-8
    """`reltol` setting for ODE solver"""
    reltol::Float64 = 1e-6
    @assert N > n 
end

"""
    pgf_ode!(dϕ, ϕ, θ, t)

The ODE system for the two elementary probability generating functions of the
two-type DL model.
"""
function pgf_ode!(dϕ, ϕ, θ, t)
    @unpack λ, ν, μ₁, μ₂ = θ
    dϕ[1] = μ₁ + λ*ϕ[1]*ϕ[2] - (λ + μ₁)*ϕ[1]
    dϕ[2] = μ₂ + ν*ϕ[1] + λ*ϕ[2]^2 - (λ + ν + μ₂)ϕ[2]
end

"""
    ϕ1ϕ2(θ::TwoTypeDL, s1, s2, t; kwargs_for_ODE_solver...)

Computes the probability generating functions ϕ₁(s1, s2, t) and ϕ₂(s1, s2, t)
(coefficients are P(X(t)=(j,k)|X(0)=(1,0)) and P(X(t)=(j,k)|X(0)=(0,1))).
Involves solving a system of two ODEs.

!!! note: 
    Including a callback to ensure positivity is possible, but doesn't work
    with complex input.
"""
function ϕ1ϕ2(θ, s1::T, s2::T, t; kwargs...) where T
    ϕ0 = [s1; s2]
    ts = (0., t)
    pr = ODEProblem(pgf_ode!, ϕ0, ts, θ)
    sl = OrdinaryDiffEq.solve(pr, Tsit5(); dense=false, kwargs...)[:,end]
    (ϕ1=sl[1]::T, ϕ2=sl[2]::T)
end

"""
    ϕlogeny!(θ::TwoTypeDL, s, tree; kwargs...)

Compute the probability generating function along the phylogeny. `s` should be
a `num_nodes × 4` matrix with the desired power series arguments initialized at
the leaves of the tree. This matrix will be filled with the `i`th row containing
the computed values for the tree node with id `i`, with the first two columns
the values for the generating functions for the gene counts of the two types at
the end of the branch leading to node `i` and the last two columns the values
for the generating functions for the gene counts of two types at the beginning
of the branch (is that the right way to express this?). 
"""
function ϕlogeny!(θ, s, tree; kwargs...)
    function walk(n)
        if isleaf(n) 
            s1, s2 = s[id(n),1], s[id(n),2]
            n1, n2 = ϕ1ϕ2(θ, s1, s2, distance(n); kwargs...)
            s[id(n),3] = n1
            s[id(n),4] = n2
            return n1, n2
        end
        childs = map(walk, children(n))
        if iswgm(n)
            s1, s2 = wgmpgf(θ, n, childs[1]...) 
        else
            s1 = prod(first.(childs))
            s2 = prod(last.(childs))
        end
        s[id(n),1] = s1
        s[id(n),2] = s2         
        isroot(n) && return s1, s2
        n1, n2 = ϕ1ϕ2(θ, s1, s2, distance(n); kwargs...)
        s[id(n),3] = n1
        s[id(n),4] = n2         
        return n1, n2
    end
    walk(tree)
    return s
end

"""
    p_nonextinct_bothclades(θ, tree, settings)

Compute the probability of non extinction in both clades stemming from the 
root of the phylogeny using the probability generating functions.
"""
function p_nonextinct_bothclades(m::TwoTypeTree, settings)
    @unpack tree, params, prior = m
    @unpack N, n, abstol, reltol = settings
    s = zeros(length(postwalk(tree)), 4)
    s = ϕlogeny!(params, s, tree, abstol=abstol, reltol=reltol)
    a = id(tree[1])
    b = id(tree[2])
    p = -Inf
    for k=1:n-1
        for i=1:k
            # root state (i, k-i)
            left = log(1 - s[a,3]^i * s[a,4]^(k-i))
            rght = log(1 - s[b,3]^i * s[b,4]^(k-i))
            p = logaddexp(p, left + rght + logpdf(prior, i, k-i)) 
        end
    end
    return p
end
# Note that this is stuff which becomes more tricky when not having pgf's
# available as when we would using more general bivariate markov chains with a
# matrix exponential approach

"""
    ϕ_fft_grid(θ, t, N)

Evaluate the probability generating functions along the complex unit circle on
a N × N grid, to serve as input for a discrete fourier transform. 

!!! note:
    We use the ensemble solver from DifferentialEquations to solve the ODEs
    along the grid of initial conditions in parallel.

!!! note:
    kwargs go to the `solve` function in DiffEq.  The tolerance settings in the
    ODE solver seem to be the main factor affecting accuracies of small
    transition probabilities.
"""
function ϕ_fft_grid(θ, t, settings::PSettings)
    @unpack N, abstol, reltol = settings
    prob = ODEProblem(pgf_ode!, [0., 0.], (0., t), θ)
    init = [[exp(2π*im*u/N), exp(2π*im*v/N)] for u=0:N-1, v=0:N-1]
    function ensemble_probfun(prob, i, repeat)
        remake(prob, u0=init[i])
    end
    ensemble_prob = EnsembleProblem(prob, prob_func=ensemble_probfun)
    sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), 
                trajectories=N*N, abstol=abstol, reltol=reltol, dense=false)
    extract_ϕ1ϕ2_solution(sim, N, Matrix{Complex{Float64}})
end
# note on the output of the ensemble solver: sim.u is a vector of length N x N
# with the solutions along the N x N grid. `last.(sim.u)` will give the solutions
# at the end of the time interval for all points on the grid. sol.u[i](0.2) will
# work to get the solution at time 0.2 for the ith grid point.

# note on interpolating: solving the system seems to be cheaper than extracting
# the solutions across the grid for intermediate times... So it is better to
# recompute the solutions with different end times for different branches than
# to solve it once for the longest branch and extract intermediate solutions?

"""
    extract_ϕ1ϕ2_solution(sln::EnsembleSolution, N)

Extract the ODE solver results along the NxN grid.
"""
function extract_ϕ1ϕ2_solution(sln::EnsembleSolution, N, T=Any)
    U  = last.(sln.u)
    ϕ1 = reshape(first.(U), N, N)
    ϕ2 = reshape(last.(U), N, N)
    (ϕ1::T, ϕ2::T)
end

"""
    transitionp_fft(ϕ1, ϕ2, j, k, [nonext=true])

Obtain the matrix of transition probabilities P{X(t)=(l,m)|X(0)=(j,k)} using
the fast Fourier transform for the grid computed for the probability generating
functions. 

!!! warn
    Numerical error in the transition probability approximations comes mainly
    from the ODE solver. These numerical errors lead to some transition
    probabilities turning out to be ≈ 0 but < 0. Currently we truncate the
    computed transition probabilities such that all `p` for which `abs(p) <
    abs(minimum(P))` are set to `-Inf`. 
"""
function transitionp_fft(ϕ1, ϕ2, j, k, nonext=false)
    A = (ϕ1 .^ j) .* (ϕ2 .^ k)
    fft!(A)
    P = real(A) ./ size(ϕ1, 1)^2
    trunc = abs(minimum(P))
    P[P .< trunc] .= 0.
    nonext && (P[1,:] .= 0.)
    return log.(P)
end

"""
    transitionp(j, k, t, N)

Obtain the matrix of transition probabilities P{X(t)=(l,m)|X(0)=(j,k)} using
the fast Fourier transform of the fourier series representation of the probability
generating functions.
"""
function transitionp(θ, j, k, t, settings::PSettings)
    ϕ1, ϕ2 = ϕ_fft_grid(θ, t, settings)
    transitionp_fft(ϕ1, ϕ2, j, k, θ.μ₁ == 0.)
end
