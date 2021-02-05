# # A two-type continuous time branching process model of gene family evolution

#=
This is the documentation for `TwoTypeDLModel.jl`, a julia package for
performing statistical inference for a two-type branching process model of gene
family evolution along a phylogeny.  

## Phylogenetic modeling of gene family evolution using gene counts

Given **gene count data** for a set of species related by a **known
phylogeny**, we seek to model the evolution of gene copy number along the
phylogeny. Some popular tools for this purpose include CAFE, BadiRate, DupliPhy
or Count, which all use phylogenetic birth-death processes (BDPs) to model gene
family counts. These phylogenetic BDPs do not however provide a good fit to
observed gene family data. We propose to model gene family evolution by
duplication and loss using a two-type branching process model to better capture
some of the evolutionary dynamics of these processes.

Specifically, the methods implemented in this package currently allow to:

1) Compute the likelihood of a phylogenetic profile (a set of gene count 
   for a family for the leaves of the species tree) under the two-type
   duplication loss (DL) model.
2) Perform Bayesian inference for the two-type DL model using MCMC
3) Simulate gene count data under the two-type DL model and the
   duplication-loss-functionalization (DLF) model along the phylogeny

!!! note
    Gene families can be inferred using a software like
    [OrthoFinder](https://github.com/davidemms/OrthoFinder). The branch
    lengths of the species tree define the relevant time units of the
    evolutionary rates.

!!! warning
    All methods in this package assume that for each family in the data set,
    there is at least one gene in each clade stemming from the root.
    Specifically, this means the likelihood is calculated conditional on this
    assumption. It is therefore necessary to filter the data accordingly,
    otherwise the likelihood is incorrect.

## The two-type branching process model

The model for which this package implements inference and simulation methods is
a [two-type continuous-time branching
process](https://en.wikipedia.org/wiki/Branching_process#Multitype_branching_processes)
which obeys the following stochastic dynamics

$$\Pr\{X(t+\Delta t) = (k,l) | X(t)=(i,j)\} = \begin{cases}
   (i+j)\lambda\Delta t + o(\Delta t) & k+l = i+j+1   \\
   i\mu_1 \Delta t + o(\Delta t) & k = i-1,\ l=j \\
   j\mu_2 \Delta t + o(\Delta t) & k = i,\ l = j-1 \\
   j\nu \Delta t + o(\Delta t)   & k = i+1,\ l = j-1 \\
   1 - ((i+j)\lambda + i\mu_1 + j(\mu_2 + \nu))\Delta t + o(\Delta t) & k=i, l=j \\
   o(\Delta t) & \text{else}
  \end{cases}$$

Or schematically (kind of like a chemical reaction network) 

```
1 → 12   with rate λ
1 →  ∅   with rate μ₁
2 → 22   with rate λ
2 →  ∅   with rate μ₂ > μ₁ and μ₂ > λ + ν
2 →  1   with rate ν
```

where `1` denotes a type 1 particle and `2` a type 2 particle, while `∅`
denotes nothing. The basic model is represented in the package as follows:
=#
using TwoTypeDLModel
m = TwoTypeDL(λ=0.2, μ₁=0.2, ν=0.1, μ₂=1.5)

#=
The package implements the method of Xu *et al.* (2015) to compute transition
probabilities for two-type branching processes using a differential equation
representation of the associated probability generating function (pgf), and
numerical inversion of the pgf using FFT to *approximate* the relevant
probabilities. Important algorithm settings that determine the accuracy of
the numerical approximations of transition probabilities are represented by
the following object:
=#
s = PSettings(n=12, N=16, abstol=1e-6, reltol=1e-6)

#=
Transition probabilities are computed up to abound of $n-1$ (so that the
maximum family size is equal to $2(n-1)$, consisting in that case of $n-1$ type
1 genes and $n-1$ type 2 genes; i.e. all admissable states are $(x_1,x_2)$
where $x_1 \le n-1$ and $x_2 \le n-1$). The $N$ settings determines the
discretization used in computing the Fourier transform of the probability
generating functions, and must be $\ge n$. The `abstol` and `reltol` are the
tolerance settings for the ODE solver used to evaluate the probability
generating functions solver used to evaluate the probability generationg
functions.

With these available we can compute transition probabilities, for instance the
transition probabilities for the process defined above, starting from a state
(2, 3) (i.e. two type 1 genes, 3 type 2 genes) over a time period of 1.2 can be
computed using the above define settings as
=#
P = TwoTypeDLModel.transitionp(m, 2, 3, 1.2, s)

#=
where `P[i,j]` is the log-probability to end up in state `(i+1, j+1)`, starting
from state (2,3) over a time of 1.2 time units. The rows correspond to the
number of type 1 genes, while the columns refer to the number of type 2 genes,
and the first column and row reflect the zero state. We can visualize this
matrix
=#
using Plots, StatsPlots
heatmap(P, size=(300,300), grid=false)

# or several of them over different time points:
plot([heatmap(TwoTypeDLModel.transitionp(m, 2, 3, t, s)) for t in 0.01:1:6]..., 
     layout=(2,3), size=(650,300), grid=false)

# ## *Drosophila* example

# Load the required modules
using CSV, DataFrames, NewickTree, Distributions, StatsBase, Turing
using TwoTypeDLModel, Plots, StatsPlots

# and read in the data
rdata = CSV.read("../data/drosophila-8taxa-max10-oib.csv", DataFrame)
tree  = readnw(readline("../data/drosophila-8taxa.nw"));

# ### Fitting a stationary model to the non-extinct families

# If we assume $\mu_1 = \nu = 0$, the two-type model reduces to a linear
# birth-death process with immigration at rate $\lambda$. Since $\mu_1 = 0$, a
# family cannot go extinct under the model, so this model is only applicable to
# gene families which are non-extinct in all species under consideration.  We
# may think of this subset of families more or less as those families which are
# 'essential' in some way.
#
# Interestingly, this model has a stationary distribution, in contrast with the
# two-type DL model and single-type linear BDP. This stationary distribution is
# a Geometric distribution with parameter $p = 1 - \lambda/\mu_2$. Under the
# assumption of stationarity, non-zero gene counts should be iid distributed
# according to a geometric distribution under this model. 
#
# We can fit a geometric distribution to non-extinct gene counts to obtain
# information about $\lambda/\mu_2$. First we obtain the number of observations
# for each non-zero gene count in the data:

X = counts(Matrix(rdata))[2:end] 

# We define a Turing model and do Bayesian inference 
@model geometric(X) = begin
    η ~ Uniform(0.001, 0.999)
    for (k, count) in enumerate(X)
        Turing.@addlogprob! count*logpdf(Geometric(η), k-1)
    end
end

chain_geom = sample(geometric(X), NUTS(), 1000, progress=false)

# Now we sample from the posterior predictive distribution. First some settings
N = sum(X)  # number of data points in observed data
m = length(X)  # maximum count in observed data

# some utility functions
geom_simfun(x, N, m) = proportions(rand(Geometric(get(x, :η).η[1]), N) .+ 1, 1:m) 
ppsim(chain, simfun, args...) = 
    mapreduce(i->simfun(chain[i], args...), hcat, 1:length(chain)) |> permutedims

# Now the posterior predictive simulations
ppx_geom = ppsim(chain_geom, geom_simfun, N, m)

# and some plots
kwargs = (xlabel="\$k\$", ylabel="\$f_k\$", title_loc=:left, 
          grid=false, legend=false, yscale=:log10, titlefont=10)
p1 = violin(ppx_geom, color=:white, size=(400,300), title="Geometric"; kwargs...)
scatter!(X ./ sum(X), color=:black)

# The dots (scatter plot) show the observed frequencies of families of size $k$
# in the *Drosophila* data, while the violin plot shows the posterior
# predictive distribution.
# We see that the geometric distribution fit is quite bad, it seriously 
# underestimates the frequency of higher gene counts. The size distribution
# of gene families is well-known to follow approximately a power-law in its
# tail, and is universally (i.e. across the tree of life) overdispersed with
# respect to the geometric expectation. 

# One biologically plausible reason for this is heterogeneity in gene
# duplication and loss rates across gene families. If we assume for instance
# rates vary across families in such a way that the ratio $\lambda/\mu_2$ is
# distributed according to a $\mathrm{Beta}(\alpha,\beta)$, the stationary
# distribution under the model we are considering becomes a **beta-geometric**
# (BG) distribution. We can specify a BG stationary model as follows
using TwoTypeDLModel: BetaGeometric

@model betageometric(X) = begin
    η ~ Uniform(0.001, 0.999)
    ζ ~ Turing.FlatPos(0.)
    X ~ BetaGeometric(η, ζ)
end

# Note that we use the dispersion ($\zeta = \alpha + \beta$) and mean ($\eta =
# \alpha / \zeta$) parameterization of the Beta distribution in this
# specification. We can sample from the posterior again

chain_bgeom = sample(betageometric(X), NUTS(), 1000, progress=false)

# here $\eta$ can still be interpreted as $1 - \lambda/\mu_2$, but we have
# the $\zeta$ parameter which governs the variation in $\lambda/\mu_2$ across
# families.

# Now to show this gives a wonderful fit compared to the geometric model, we'll
# do posterior predictive simulations again:
function bgeom_simfun(x, N, m) 
    xs = get(x, [:η, :ζ])
    proportions(rand(BetaGeometric(xs.η[1], xs.ζ[1]), N) .+ 1, 1:m)
end

ppx_bgeom = ppsim(chain_bgeom, bgeom_simfun, N, m)

p2 = violin(ppx_bgeom, color=:white, title="Beta-Geometric"; kwargs...)
scatter!(X ./ sum(X), color=:black)
plot(p1, p2, size=(600,200))

# While the stationary distribution fit does not give us very detailed
# information about gene family evolution, and does not take into account the
# phylogeny, this is nevertheless more than a simple exercise in distribution
# fitting. These results are useful because
#
# 1) We can use the fitted stationary distribution (say, the BG distribution
#    with posterior mean values for $\eta$ and $\zeta$) as a reasonable
#    data-informed **prior distribution** on the number of lineages at the root of
#    the species tree for each family. 
#
# 2) The $\zeta$ parameter gives us an idea of the degree of rate heterogeneity
#    across families. For $\zeta$ large, the BG distribution approaches the
#    geometric distribution, entailing a more or less constant $\lambda/\mu_2$
#    ratio, whereas small values of $\zeta$ suggest $\lambda/\mu_2$ varies
#    strongly across families.

# I'll save the posterior mean values for later
zeta, eta = mean(chain_bgeom).nt[2]

# Note that a $\zeta < 1$ value may be problematic.

# ## Inference for the two-type model
#
# !!! note 
#     You will probably want to run the code below multi-threaded. Don't forget
#     to start your julia session (or run your script) with multiple threads, 
#     e.g. `julia -t 10`.
#
# ### Setting up the model and data
#
# We first specify some settings for computing the transition probability
settings = PSettings(n=12, N=16, abstol=1e-6, reltol=1e-6)

# Next we load the data in a directed acyclic data structure (DAG). The DAG
# structure allows to organize the computations in an efficient manner when
# model parameters are shared across families. For the example here I'll use
# a subset of the data of 100 families
idx  = sample(1:nrow(rdata), 100, replace=false) 
data = TwoTypeDLModel.CountDAG(rdata[idx,:], tree, settings.n)

# Now we set up the prior on the number of lineages at the root. We will use
# a bounded Beta-Geometric prior, with bound $2n$ and the parameters estimated
# under the stationary distribution model (see above). Note that the root prior
# models offset $\zeta$ by 1 to ensure the distribution is proper (so we should
# provide `zeta - 1`)
rprior = TwoTypeDLModel.BBGPrior(eta, zeta - 1, 0.5, 1:(settings.n-1)*2);

# Now we specify the actual model. We initialize the model with random
# parameter values.
model  = TwoTypeTree(tree, TwoTypeDL(rand(4)...), rprior)

# The model object behaves more or less as you'd expect it, e.g.
TwoTypeDLModel.loglikelihood(model, data, settings)

# ### Setting up the MCMC algorithm and doing inference
#
# The package implements a dedicated adaptive metropolis-within-gibbs MCMC
# algorithm. It currently does not make use of any of the more flexible 
# packages for Bayesian inference (e.g. Turing.jl) available in the julia
# ecosystem. 
#
# We do not perform inference of $\lambda, \mu_1, \mu_2$ and $\nu$ directly, but
# instead assume $\lambda, \mu_1, \nu$ are all $< \mu_2$, and express the
# former parameters as fractions of the latter, i.e. we assume $\lambda =
# a\mu_2$, $\mu_1 = b\mu_2$ and $\nu = c\mu_2$, where $a, b, c$ are all $\in
# (0,1)$. We specify independent priors for $a, b, c, \mu_2$ and $r$ (the
# probability that an extra gene at the root is an excess (type 2) gene) in a
# tuple as follows.
priors = (Beta(), Beta(), Beta(), Exponential(10.), Beta())

# Optionally, we may add a prior for $\zeta$ at the end of this tuple, in which
# case $\zeta$ will also be treated as random.

# We can sample from the posterior distribution using the following code
chn = TwoTypeDLModel.Chain(model, priors, data, settings)
spl = sample(chn, 100);

# Of course we should sample much longer (about 11000, discarding 1000 as
# burn-in should do usually). To obtain the posterior distribution and model
# objects for posterior predictions, use the following
p, m = TwoTypeDLModel.post(chn, spl);
last(p, 10)

# ### Posterior predictive simulation for the two-type model

# To perform posterior predictive simulations, simply run
ppx = TwoTypeDLModel.ppsim(m, 100);

# `ppx` contains the simualted size distributions. Something like the following
# line of code could be used to obtain for instance the posterior predictive 
# 95% intervals for the size distribution for each taxon
combine(groupby(ppx, :k), names(ppx) .=> x->Tuple(quantile(x, [0.025, 0.975])))
