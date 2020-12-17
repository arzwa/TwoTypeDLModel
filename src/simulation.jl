"""
    getrates(θ::TwoTypeDL, X)

Get the duplication/loss/... rates when in state X = (X₁, X₂).
"""
getrates(p::TwoTypeDL, X) = [p.λ*X[1], p.μ₁*X[1], p.λ*X[2], p.μ₂*X[2], p.ν*X[2]]

"""
    simulate(model::TwoTypeTree, n, [condition::Function])

Simulate `n` family profiles from the two-type branching process model, subject
to the condition. `condition` should be a function `f(x)` checking whether a named
tuple `x` of the form (leafname=(X1, X2),...) satisfies the condition.

# Example

```julia-repl
julia> tree = readnw("((A:1.2,B:1.2):0.8,C:2.0);");

julia> model = TwoTypeTree(tree, TwoTypeDL(rand(4)...), GeometricPrior(eta, 0.5));

julia> TwoTypeDLModel.simulate(model, 3)
(3×3 DataFrame
 Row │ A      B      C
     │ Int64  Int64  Int64
─────┼─────────────────────
   1 │     4      2      1
   2 │     1      7      2
   3 │     1      4      3, 3×3 DataFrame
 Row │ A       B       C
     │ Tuple…  Tuple…  Tuple…
─────┼────────────────────────
   1 │ (1, 3)  (2, 0)  (0, 1)
   2 │ (1, 0)  (2, 5)  (1, 1)
   3 │ (1, 0)  (2, 2)  (2, 1))
```
"""
function simulate(m::TwoTypeTree, n, condition=default_condition(m.tree))
    df = map(i->simulate(m.params, rand(m.prior), m.tree, condition), 1:n) |> DataFrame
    ddf = select(df, names(df) .=> x->first.(x) .+ last.(x))
    rename!(ddf, names(df))
    ddf, df
end

function default_condition(tree)
    left = Symbol.(name.(getleaves(tree[1])))
    rght = Symbol.(name.(getleaves(tree[2])))
    x -> any(y->sum(getfield(x, y)) .> 0, left) && 
         any(y->sum(getfield(x, y)) .> 0, rght) 
end

"""
    simulate(θ::TwoTypeDL, X, tree, [condition::Function])

Simulate the two-type branching process model with parameters `θ` along a tree
`tree` with root state `X`.
"""
function simulate(p, X, tree, condition=x->true)
    x = simulate(p, X, tree)
    while !condition(x)
        x = simulate(p, X, tree)
    end
    return x
end

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
