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
    ddf = select(df, names(df) .=> x->sum.(x))
    rename!(ddf, names(df))
    ddf, df
end

function simulate(p, X, tree, condition=x->true)
    x = simulate(p, X, tree)
    while !condition(x)
        x = simulate(p, X, tree)
    end
    return x
end

"""
    default_condition(tree, maxn=Inf)

Get a closure for the Boolean function that checks whether a (simulated) gene
family profile satisfies the default condition of being non-extinct in both
lineages stemming from the root. Optionally this takes a maximum count if we
have an upper bound condition.
"""
function default_condition(tree, maxn=Inf)
    left = Symbol.(name.(getleaves(tree[1])))
    rght = Symbol.(name.(getleaves(tree[2])))
    function cond(y)
        y = map(x->sum(x), y)
        a = any(x->getfield(y, x) > 0, left)
        b = any(x->getfield(y, x) > 0, rght)
        c = all(x->x < maxn, y)
        a && b && c
    end
end

"""
    simulate(θ, X, tree)

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


# Ideal two-type model
# ====================
# In the ideal two-type model we model the evolution of functionally redundant
# groups. The state is now represented by a vector of integers, recording the
# number of gene copies for each functional group. *functionalization events
# give rise to a new group of size 1.
"""
    getrates(θ::IdealTwoTypeDL, x::Vector)

Get the total event rate, the total event rate for each functional group, and
the individual within group event rates.
"""
function getrates(θ::IdealTwoTypeDL, x)
	@unpack λ, μ₁, μ₂, ν = θ
	rates = map(x) do n
		n == 1 ? [λ, μ₁, 0.] : [n*λ, n*μ₂, n*ν]
	end
	grouprates = sum.(rates)
	return sum(grouprates), grouprates, rates
end

"""
    simulate(θ::IdealTwoTypeDL, x, t)

Simulate the ideal two-type DL model for a time interval `t`. The state `x` is
a vector of integers, representing the number of genes in each redundant group.
"""
function simulate(θ::IdealTwoTypeDL, x, t::Real)
	x = copy(x)
	r, gr, rs = getrates(θ, x)
	t -= randexp() / r
	while t > 0.
		group = sample(1:length(gr), Weights(gr))
		event = sample(1:3, Weights(rs[group]))
		if event == 1
			x[group] += 1
		elseif event == 2
			x[group] -= 1
		elseif event == 3
			x[group] -= 1
			push!(x, 1)
		end
		r, gr, rs = getrates(θ, x)
		t -= randexp() / r
	end
	return x
end

function simulate(θ::IdealTwoTypeDL, x::T, tree::Node) where T
    result = Dict{Symbol,T}()
    function simwalk(node, X)
        _X = simulate(θ, X, distance(node))
        isleaf(node) && return result[Symbol(name(node))] = _X
        for c in children(node)
            simwalk(c, _X)
        end
    end
    simwalk(tree, x)
    return (; result...)
end
