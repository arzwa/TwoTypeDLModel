"""
    getrates(θ::TwoTypeDL, X)

Get the duplication/loss/... rates when in state X = (X₁, X₂).
"""
getrates(p::TwoTypeDL, X) = [p.λ*X[1], p.μ₁*X[1], p.λ*X[2], p.μ₂*X[2], p.ν*X[2]]

"""
    simulate(θ, rootprior, tree, n)
    simulate(θ, tree, n; p=0.5) 

Simulate `n` family profiles from the two-type branching process model with
parameters `θ` along a tree, assuming the `rootprior` for the number of genes
at the root.

# Example

```julia-repl
julia> θ = TwoTypeDL(rand(5)...);

julia> prior = RootPrior(0.8, 0.9);

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
function simulate(m::TwoTypePhyloModel, n)
    df = map(i->simulate(m.params, rand(m.prior), m.tree), 1:n) |> DataFrame
    ddf = select(df, names(df) .=> x->first.(x) .+ last.(x))
    rename!(ddf, names(df))
    ddf, df
end

"""
    simulate(θ::TwoTypeDL, X, tree)

Simulate the two-type branching process model with parameters `θ` along a tree
`tree` with root state `X`.
"""
function simulate(p::TwoTypeDL, X::Tuple, tree)
    result = Dict{Symbol,Tuple{Int,Int}}()
    function simwalk(node, X)
        _X = _simulate(p, X, distance(node))
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
function _simulate(p::TwoTypeDL, X, t)
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
