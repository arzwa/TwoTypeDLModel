# This was the first implementation, but it has no advantage over the 
# CountDAG for our current applications.
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
    loglikelihood(m::TwoTypeTree, data, settings)

Compute the loglikelihood P(data|θ,tree) for model θ, data and a tree,
marginalized over the root prior distribution.

!!! note 
    This function is implemented so that the output is, as expected, a negative
    real number being the loglikelihood. The function `loglhood` which takes
    the same arguments returns a tuple with both the marginal loglikelihood and
    the dynamic programming matrix, which may be useful for recycling precious
    computations when only the root state/prior changes.

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

julia> model = TwoTypeTree(tree, θ, GeometricPrior(0.8, 0.5));

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
    prune(θ::TwoTypeDL, tree, data, settings)

Compute the matrix of 'partial' loglikelihood values alon the tree using
Felsenstein's pruning algorithm (variable elimination).
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

