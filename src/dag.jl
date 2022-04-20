# we will need this
iswgm(node) = NewickTree.degree(node) == 1

# Adapted from DeadBird.
# NOTE: currently the dag struct is bigger than it should, as we take up
# unnecessary space for each leaf observation in the last dimension (we
# don't need to compute two matrices for leaf branches...)
"""
    CountDAG(df::DataFrame, tree::Node)

Get a `CountDAG` from a count matrix The directed acyclic graph (DAG)
representation of a phylogenetic profile for an (assumed known) species tree.  
This is a [multitree](https://en.wikipedia.org/wiki/Multitree)
"""
struct CountDAG{T,V,G}  
    graph::SimpleDiGraph{G}  # the DAG, with vertices ordered in a post-order
    nodes::Vector{UnitRange{G}}
    ndata::Vector{Tuple{V,Int}}
    parts::Array{T,4}
    nfam ::Int
end

function CountDAG(df, tree, n)
    @assert maximum(map(sum, Matrix(df))) < n
    dag = SimpleDiGraph()
    ndata = Tuple{eltype(df[:,1]),Int}[]
    nodes = Vector{UnitRange{Int}}(undef, length(postwalk(tree)))
    function walk(n, l)
        if isleaf(n)
            y = add_leaves!(dag, ndata, df[:,name(n)], id(n))
            nodes[id(n)] = UnitRange(extrema(y)...)
        else
            x = zip([walk(c, l+1) for c in children(n)]...)
            y = add_internal!(dag, ndata, x, id(n))
            nodes[id(n)] = UnitRange(extrema(y)...)
            isroot(n) && add_root!(dag, ndata, y, id(n))
        end
        return y
    end
    walk(tree, 1)
    # our innermost loops are across families, so that should be the first index
    parts = fill(-Inf, nv(dag), n, n, 2)
    return CountDAG(dag, nodes, ndata, parts, nrow(df))
end

Base.show(io::IO, dag::CountDAG) = write(io, "CountDAG($(dag.graph))")
Base.length(dag::CountDAG) = dag.nfam

"""
    add_leaves!(dag, ndata, x, n)

For a species tree leaf node `n`, this adds the vector of (gene) counts `x` for
that species to the graph.  This returns for each gene family the corresponding
node that was added to the graph
"""
function add_leaves!(dag, ndata, x, n)
    idmap = Dict()
    for (k,v) in countmap(x)
        push!(ndata, (k, v))
        add_vertex!(dag)
        idmap[k] = nv(dag)
    end
    [idmap[xᵢ] for xᵢ in x]
end

"""
    add_internal!(dag, ndata, x, n)

For a species tree internal node `n`, this adds the gene family nodes
associated with `n` to the graph and provides the bound on the number of
lineages that survive to the present below `n` for each gene family.  Note that
`x` is a vector of tuples of DAG nodes that each will be joined into a newly
added node.  The resulting nodes are returned.

!!! note: I believe this also works for multifurcating species trees (like the
Csuros Miklos algorithm does too)
"""
function add_internal!(dag, ndata, x, n)
    idmap = Dict()
    for (k,v) in countmap(collect(x))
        # ugly hack to make it work with completely observed data as well
        eltype(ndata) == Tuple{Int,Int} ? 
            push!(ndata, (-1, v)) : push!(ndata, ((-1,-1),v))
        add_vertex!(dag); i = nv(dag)
        for j in k add_edge!(dag, i, j) end
        idmap[k] = i
    end
    [idmap[xᵢ] for xᵢ in x]
end

"""
    add_root!(dag, ndata, x, n)
"""
function add_root!(dag, ndata, x, n)
    add_vertex!(dag); i = nv(dag)
    for j in unique(x) add_edge!(dag, i, j) end
end

# likelhood algorithm
# marginalize over root prior for the DAG
"""
    integrate_prior(dag::CountDAG, d)

Marginalize the likelihood over the root prior for the DAG data structure.
This outputs the complete marginal loglikelihood for the DAG.
"""
function integrate_prior(dag::CountDAG{T}, d) where T
    ℓ = zero(T)
    for n in dag.nodes[1]
        k = dag.ndata[n][2]
        ℓ += k*integrate_prior(dag.parts[n,:,:,1], d)
    end
    return ℓ
end

# for the MCMC algorithm
function integrate_prior(L, dag::CountDAG{T}, d) where T
    ℓ = zero(T)
    for (i, n) in enumerate(dag.nodes[1])
        k = dag.ndata[n][2]
        ℓ += k*integrate_prior(L[i,:,:], d)
    end
    return ℓ
end

"""
    loglikelihood(model, dag, [settings])
"""
function Distributions.loglikelihood(
        model::TwoTypeTree, dag::CountDAG, settings=PSettings())
    prune!(dag, model.tree, model.params, settings)
    p = p_nonextinct_bothclades(model, settings)
    return integrate_prior(dag, model.prior) - dag.nfam * p
end

# For the MCMC algorithm
function _loglhoodroot(model, dag, L, settings=PSettings())
    p = p_nonextinct_bothclades(model, settings)
    return integrate_prior(L, dag, model.prior) - dag.nfam * p
end

"""
    prune!(dag, tree, θ, settings)

Execute the pruning algorithm, i.e. compute the probabilistic graphical model
using variable elimination.
"""
function prune!(dag, tree, θ, settings) where T
    for node in postwalk(tree)
        if !isleaf(node) 
            iswgm(node) ? wgm!(dag, node, θ) : combine!(dag, node)
        end
        !isroot(node) && prune_edge!(dag, node, θ, settings)
    end
end

"""
    combine!(dag, node)

Compute the partial likelihoods for the end of the branch leading to node 
`node` by taking the product of the partial likelihoods at the beginning 
of the conditionally independent daughter branches.
"""
function combine!(dag, node)
    @unpack nodes, graph, parts = dag
    for n in nodes[id(node)]
        u, v = outneighbors(graph, n)
        @inbounds parts[n, :, :, 1] .= parts[u, :, :, 2] .+ parts[v, :, :, 2]
    end
end

"""
    prune_edge!(dag, node, θ, settings)

Compute the partial likelihood at the beginning of the branch leading to node
`node` given the matrix of partial likelihoods at the end of that same branch.
"""
function prune_edge!(dag, node, θ, settings)
    @unpack n = settings
    ϕ1, ϕ2 = ϕ_fft_grid(θ, distance(node), settings)
    leaf = isleaf(node)
    Threads.@threads for k=0:n-1
        for j=0:n-1  
            # (j, k) -> (l, m)
            P = transitionp_fft(ϕ1, ϕ2, j, k, θ.μ₁ == 0.)
            leaf ? _prune_leaf!(dag, node, j, k, P) : 
                   _prune_edge!(dag, node, j, k, P, n)
        end
    end
end

# inner loop for edge leading to leaf, where observations are Z
function _prune_leaf!(dag::CountDAG{T,Int}, node, j, k, P) where T
    for i in dag.nodes[id(node)]
        Z = dag.ndata[i][1] 
        @inbounds dag.parts[i, j+1, k+1, 2] = -Inf
        for l=0:Z
            @inbounds dag.parts[i, j+1, k+1, 2] = 
                logaddexp(dag.parts[i, j+1, k+1, 2], P[l+1, Z-l+1])  # P[(j,k)->(l,Z-l)]
        end
    end
end

# inner loop for edge leading to leaf, where observations are (X₁, X₂)
function _prune_leaf!(dag::CountDAG{T,<:Tuple}, node, j, k, P) where T
    for i in dag.nodes[id(node)]
        X₁, X₂ = dag.ndata[i][1] 
        @inbounds dag.parts[i, j+1, k+1, 2] = -Inf
        @inbounds dag.parts[i, j+1, k+1, 2] = 
            logaddexp(dag.parts[i, j+1, k+1, 2], P[X₁+1, X₂+1])
    end
end

# inner loop for pruning along internal edge
function _prune_edge!(dag, node, j, k, P, n)
    @inbounds Pjk = (@view P[1:n, 1:n])
    for i in dag.nodes[id(node)]
        @inbounds dag.parts[i, j+1, k+1, 2] = 
            logsumexp(dag.parts[i, :, :, 1] .+ Pjk)
    end
end

