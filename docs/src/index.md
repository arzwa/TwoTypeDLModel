
# A two-type continuous time branching process model of gene family evolution

This is the documentation for `TwoTypeDLModel.jl`, a julia package for
performing statistical infernce for a two-type branching process model of
gene family evolution along a phylogeny.

## Phylogenetic modeling of gene family evolution using gene counts

## The two-type branching process model

## *Drosophila* example

Load the required modules

```@example index
using TwoTypeDLModel, CSV, DataFrames, NewickTree, Distributions, StatsBase, Turing
using Plots, StatsPlots
```

and read in the data

```@example index
rdata = CSV.read("../data/drosophila-8taxa-max10-oib.csv", DataFrame)
tree  = readnw(readline("../data/drosophila-8taxa.nw"));
nothing #hide
```

### Fitting a stationary model to the non-extinct families

If we assume $\mu_1 = \nu = 0$, the two-type model reduces to a linear
birth-death process with immigration at rate $\lambda$. Since $\mu_1 = 0$, a
family cannot go extinct under the model, so this model is only applicable to
gene families which are non-extinct in all species under consideration.  We
may think of this subset of families more or less as those families which are
'essential' in some way.

Interestingly, this model has a stationary distribution, in contrast with the
two-type DL model and single-type linear BDP. This stationary distribution is
a Geometric distribution with parameter $p = 1 - \lambda/\mu_2$. Under the
assumption of stationarity, non-zero gene counts should be iid distributed
according to a geometric distribution under this model.

We can fit a geometric distribution to non-extinct gene counts to obtain
information about $\lambda/\mu_2$. First we obtain the number of observations
for each non-zero gene count in the data:

```@example index
X = counts(Matrix(rdata))[2:end]
```

We define a Turing model and do Bayesian inference

```@example index
@model geometric(X) = begin
    η ~ Uniform(0.001, 0.999)
    for (k, count) in enumerate(X)
        Turing.@addlogprob! count*logpdf(Geometric(η), k-1)
    end
end

chain_geom = sample(geometric(X), NUTS(), 1000)
```

Now we sample from the posterior predictive distribution. First some settings

```@example index
N = sum(X)  # number of data points in observed data
m = length(X)  # maximum count in observed data
```

some utility functions

```@example index
geom_simfun(x, N, m) = proportions(rand(Geometric(get(x, :η).η[1]), N) .+ 1, 1:m)
ppsim(chain, simfun, args...) =
    mapreduce(i->simfun(chain[i], args...), hcat, 1:length(chain)) |> permutedims
```

Now the posterior predictive simulations

```@example index
ppx_geom = ppsim(chain_geom, geom_simfun, N, m)
```

and some plots

```@example index
kwargs = (xlabel="\$k\$", ylabel="\$f_k\$", title_loc=:left,
          grid=false, legend=false, yscale=:log10, titlefont=10)
p1 = violin(ppx_geom, color=:white, size=(400,300), title="geometric"; kwargs...)
scatter!(X ./ sum(X), color=:black)
```

We see that the geometric distribution fit is quite bad, it seriously
underestimates the frequency of higher gene counts. The size distribution
of gene families is well-known to follow approximately a power-law in its
tail, and is universally (i.e. across the tree of life) overdispersed with
respect to the geometric expectation.

One biologically plausible reason for this is heterogeneity in gene
duplication and loss rates across gene families. If we assume for instance
rates vary across families in such a way that the ratio $\lambda/\mu_2$ is
distributed according to a $\mathrm{\Beta}(\alpha,\beta)$, the stationary
distribution under the model we are considering becomes a **beta-geometric**
(BG) distribution. We can specify a BG stationary model as follows

```@example index
using TwoTypeDLModel: BetaGeometric

@model betageometric(X) = begin
    η ~ Uniform(0.001, 0.999)
    ζ ~ Turing.FlatPos(0.)
    X ~ BetaGeometric(η, ζ)
end
```

Note that we use the 'dispersion $\zeta = \alpha + \beta$ and mean $\eta =
\alpha / \zeta$' parameterization of the Beta distribution in this
specification. We can sample from the posterior again

```@example index
chain_bgeom = sample(betageometric(X), NUTS(), 1000)
```

here $\eta$ can still be interpreted as $1 - \lambda/\mu_2$, but we have
the $\zeta$ parameter which governs the variation in $\lambda/\mu_2$ across
families.

Now to show this gives a wonderful fit compared to the geometric model, we'll
do posterior predictive simulations again:

```@example index
function bgeom_simfun(x, N, m)
    xs = get(x, [:η, :ζ])
    proportions(rand(BetaGeometric(xs.η[1], xs.ζ[1]), N) .+ 1, 1:m)
end

ppx_bgeom = ppsim(chain_bgeom, bgeom_simfun, N, m)

p2 = violin(ppx_bgeom, color=:white, title="beta-geometric"; kwargs...)
scatter!(X ./ sum(X), color=:black)
plot(p1, p2, size=(600,200))
```

While the stationary distribution fit does not give us very detailed
information about gene family evolution, and does not take into account the
phylogeny, this is nevertheless more than a simple exercise in distribution
fitting. These results are useful because

1. We can use the fitted stationary distribution (say, the BG distribution
with posterior mean values for $\eta$ and $\zeta$) as a reasonable
data-informed **prior distribution** on the number of lineages at the root of
the species tree for each family.

2. The $\zeta$ parameter gives us an idea of the degree of rate heterogeneity
across families. For $\zeta$ large, the BG distribution approaches the
geometric distribution, entailing a more or less constant $\lambda/\mu_2$
ratio, whereas small values of $\zeta$ suggest $\lambda/\mu_2$ varies
strongly across families.

```@example index
### Inference for the two-type model
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

