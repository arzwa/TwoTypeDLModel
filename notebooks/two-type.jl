### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ e52945b8-33d2-11eb-27e3-890af106f8dc
using DifferentialEquations, Plots, Parameters, StatsFuns

# ╔═╡ f649189e-356b-11eb-32cc-7da3efe6ba92
using FFTW

# ╔═╡ 6f4f6fc6-386d-11eb-080e-9309615f7cf3
using LinearAlgebra

# ╔═╡ d6091f72-389b-11eb-28a5-9f0c7b631f67
using ForwardDiff

# ╔═╡ 7381da4a-38b1-11eb-09ca-9f2e657dfd9a
using NewickTree, CSV, DataFrames

# ╔═╡ a1517d28-392d-11eb-3846-5f3fd9bf6514
using Random, StatsBase, Distributions

# ╔═╡ 86f6c9c2-3952-11eb-118b-53328311f861
using Optim

# ╔═╡ a426a434-33b9-11eb-2c0a-81b18487b3cb
md""" 
# Two-type branching process model of gene family evolution

**Remark:** another field where cognate models that may be appliccable to gene family evolution could be found is the theory of phase-type distributions and QBDs. Howvere, here I will first consider the problem from the branching-process viewpoint.
"""

# ╔═╡ e69c43b2-33cf-11eb-15aa-4f0ba0fbb8e8
md"""
We use the approach of Xu *et al.* (which is based on the wonderful book by N.T. Bailey) to derive a system of ODEs for the probability generating functions of the two-type continuous-time branching process. We assume the process has instantaneous rates 

$a_1(k,l) = P\{X(dt) = (k,l)| X(0)=(1,0)\}$ 
$a_2(k,l) = P\{X(dt) = (k,l)| X(0)=(0,1)\}$

genes evolve independently (i.e. the branching property holds) and have exponentially distributed life times (i.e. the Markov property holds, here the life time is interpreted as the interevent time).

In the standard duplicate gene model we assume duplicate genes are incorporated at rate $\nu$ and incorporated genes have loss rate $\mu_1< \mu_2$ where $\mu_2$ is the loss rate for a duplicated gene. In full, we assume that $a_1(0,0) = \mu_1$, $a_1(1,1) = \lambda$ and $a_1(1,0) = -(\lambda + \mu_1)$, and furthermore that $a_2(0,0) = \mu_2$, $a_2(1,0) = \nu$, $a_2(0,2) = \lambda$ and $a_2(0,1) = (\lambda+\nu +\mu_2)$.

We denote by 

$u_1(s_1, s_2) = \sum_{k=0}^\infty \sum_{l=0}^\infty a_1(k,l)s_1^k s_2^k$

the bivariate generating function for the $a_1(k,l)$. An analogous definition holds for $u_2(s_1, s_2)$. We write the probability generating function for the transient distribution conditional on starting in state $(1,0)$ as 

$\phi_1(t, s_1, s_2) = \sum_{k=0}^\infty \sum_{l=0}^\infty P\{X(t) = (k,l)|X(0)=(1,0)\}s_1^ks_2^l$

with $\phi_2(t, s_1, s_2)$ defined analogously be conditional on staring in $X(0)=(0,1)$.
"""

# ╔═╡ d57f28e0-33d1-11eb-253b-47d7444fd676
md"""
Using the approach of Xu *et al.* we arrive at the system of ODEs

$\frac{d}{dt} \phi_1 = \mu_1 + \lambda \phi_1 \phi_2 - (\lambda + \mu_1)\phi_1$

$\frac{d}{dt} \phi_2 = \mu_2 + \nu \phi_1 + \lambda \phi_2^2 - (\lambda + \nu + \mu_2)\phi_2$

**Remark:** these seem more tricky than in the BDS model of Xu *et al.*, where the second DE in the system is a function of $\phi_2$ only.
"""

# ╔═╡ 6b2fe6be-388b-11eb-2e4d-e5ce95c938f2
@with_kw struct TwoTypeDL{T}
	λ ::T
	ν ::T
	μ₁::T
	μ₂::T
end

# ╔═╡ 0d462160-389c-11eb-2c96-e3e74916de80
asvector(p::TwoTypeDL) = [p.λ, p.ν, p.μ₁, p.μ₂]

# ╔═╡ ee062fea-33d3-11eb-1dd4-a7842503d0f7
function system1!(dϕ, ϕ, p::TwoTypeDL, t)
	@unpack λ, ν, μ₁, μ₂ = p
	dϕ[1] = μ₁ + λ*ϕ[1]*ϕ[2] - (λ + μ₁)*ϕ[1]
	dϕ[2] = μ₂ + ν*ϕ[1] + λ*ϕ[2]^2 - (λ + ν + μ₂)ϕ[2]
end

# ╔═╡ b27f9208-33d6-11eb-1e6f-49732a81a577
function pgf1(ps::TwoTypeDL, s1, s2, t)
	ϕ0 = [s1; s2]
	ts = (0., t)
	pr = ODEProblem(system1!, ϕ0, ts, ps)
	sl = DifferentialEquations.solve(pr)[:,end]
end

# ╔═╡ ead9751c-33d9-11eb-19b2-d1c4562f2091
pgf1(TwoTypeDL(0.1, 1., 1., 2.), 0., 1., 1e8)

# ╔═╡ 55563f6c-33d8-11eb-3a90-07dbe909ab01
contourf(1.:0.5:50., 1.:0.1:10., 
	(μ2, ν)->pgf1(TwoTypeDL(λ=1., ν=ν, μ₁=0.1, μ₂=μ2), 0., 1., 1e8)[1], 
	xlabel="\$\\mu_2\$", ylabel="\$\\nu\$", 
	title="Extinction probability (1)")

# ╔═╡ cac5fb6a-33d9-11eb-3593-25fe89289b88
md"""
This is the extinction probability (assuming $t=10^8$ is long enough, but it seems so) for type 1 genes. Since it is assumed $\mu_2 + \nu > \lambda$, this is also the extinction probability for the system as a whole I guess. This seems to give expected results, suggesting the ODEs are properly derived. This does not give much insights though, as we would like to know whether there is a stationary distribution, not only whether there is a possibility of non-extinction (which is quite obvious, since if $\nu \gg \mu_2$, we effectively have a linear BDP with $\lambda > \mu$, i.e. a supercritical continuous time branching process, because any type 2 gene will rapidly become an incoroporated type 1 gene).
"""

# ╔═╡ 0fbdbaae-356b-11eb-1957-55551088614f
md"""
## General transition probabilities

We can probably use the approach using FFT as in Xu et al. To compute $P\{X(t)=(l,m)| X(0) = (j,k)\}$ we need to compute the DFT of the generating function $\phi_{jk} = \phi_1^j \phi_2^k$ and get the $[l,m]$ coefficient as far as I understand.
"""

# ╔═╡ 66222e64-356d-11eb-3af4-216f3a7f0f38
fft([0 1; 1 1; 2 1; 1 0])

# ╔═╡ a10130c8-358d-11eb-1ab7-7deb044245e4
# compute probability generating function for starting state (j,k)
function ϕjk(ps::TwoTypeDL, s1, s2, j, k, t)
	ϕ0 = [s1; s2]
	ts = (0., t)
	pr = ODEProblem(system1!, ϕ0, ts, ps)
	sl = DifferentialEquations.solve(pr)[:,end]
	sl[1]^j * sl[2]^k
end

# ╔═╡ f75f3bca-358f-11eb-0d4e-47c7860c38ef
function Pfft(ps::TwoTypeDL, j, k, t, N=16)
	A = [ϕjk(ps, exp(2π*im*u/N), exp(2π*im*v/N), j, k, t) for u=0:N-1, v=0:N-1]
	dft = fft(A) 
	P = real(dft) ./ N^2
	P[P .< zero(t)] .= 1e-16
	return P
end

# ╔═╡ f391c27e-358f-11eb-1428-9f492b868981
θ = TwoTypeDL(μ₁=0.01, λ=0.1, ν=0.5, μ₂=1.0);

# ╔═╡ cbb0a286-358d-11eb-0735-fbeb520f630c
ϕjk(θ, exp(2π*im/16), exp(2π*im/16), 4, 2, 1.1)

# ╔═╡ 76412cb6-358e-11eb-3ce3-29ce0f4a8b78
P = Pfft(θ, 1, 0, 1., 10.)

# ╔═╡ 775723f2-358f-11eb-10fe-eb18980e4dd9
map(1e-6:0.05:0.55+1e-6) do t 
	heatmap(log10.(Pfft(θ, 4, 3, t, 16.)), size=(200,200), grid=false, colorbar=false)
end |> ps->plot(ps..., size=(600,450))

# ╔═╡ 35003c06-3868-11eb-1c0c-9769f841c0c8
map(0.1:0.2:2.3) do t 
	heatmap(log10.(Pfft(θ, 4, 3, 10^t, 16.)), size=(200,200), 
		grid=false, colorbar=false)
end |> ps->plot(ps..., size=(600,450))

# ╔═╡ 67b4f18c-3596-11eb-29ac-09bd132ecba3
md"""
The transition probabilities look as expected. The process still lacks a stationary distribution, with the only equilibrium solution being $(0,0)$, but I guess it may approach this equilibrium situation quite slowly, making it a fairly reasonable model of gene family evolution I guess. At least its parameters are much more biologically meaningful than for the single-type process. 

Note that we could recover a process with a stationary distribution if we were to focus on families that do not go extinct. In this process $(0,0) \rightarrow (0,1)$ transitions are allowed and happen with rate $\lambda$. Clearly this should involve a $\phi_{00}$ pgf (which in the single-type case is a geometric), but haven't figure it out yet.
"""

# ╔═╡ 578cc30a-359a-11eb-2309-e125ae2a8e28
md"""
**Remark**: for the numerical evaluation, we can do some savings on computations. If we wish to obtain transition probabilities for different *starting states*, we only have to solve the ODEs once for $\phi_1(s_1, s_2)$ and $\phi_2(s_1, s_2)$ for every $(s_1, s_2)$ pair. So depending on our goals we should organize the calculations in other ways than the most intuitive.
"""

# ╔═╡ 77e7ab30-386d-11eb-21d1-cb3d7a0c1158
function antidiagsum(A)
	n = size(A, 1)
	x = zeros(2n)
	for i=0:n-1, j=0:i
		x[i + 1] += A[i - j + 1, j + 1]
		x[2n - i] += A[n-i+j, n-j]
	end
	return x
end

# ╔═╡ 84869a44-386e-11eb-2522-1b4e4cce2e65
begin
	p = plot()
	map(1e-6:0.5:10+1e-6) do t 
		plot!(p, antidiagsum(Pfft(θ, 4, 3, t, 16.)), color=:black)
	end 
	plot(p, grid=false, legend=false, xlim=(0,20))
end

# ╔═╡ 22fd6b60-388a-11eb-2b6e-b53cf2428ac9
md"""
## Use in a pruning algorithm

How to use the above numerical approach for computing transition probabilities efficiently in a phylogenetic context? Denote by $Z(t) = X(t)[1] + X(t)[2]$ the total family size at time $t$, our observed data will consist of the $Z(t)$ at the leaves of a species tree, we do not have access to the full information contained in $X(t)$. We also do not have access to the transition probabilities for the $Z$ process under the two type model (at least, I currently do not see how to figure that out...).

We should be able to devise a rather 'brute force' approach, computing the transition probability matrix and using a pruning algorithm (but not as 'brutal' as computing the transition probability matrix by matrix exponentiation, or is it similar?).
"""

# ╔═╡ 5249d270-388d-11eb-3d44-29dc1f6f980b
# assume we have two leaf nodes, v and w, with observed states 3 and 5
Xv = 3; Xw = 5;

# ╔═╡ b42e6d5c-388d-11eb-284e-d71aa40c2d12
# assume we bound the state space to 10 × 10, so we have at most 20 genes in a family
begin
	xmax = 10
	Lv = log.(zeros(xmax))
	Lw = log.(zeros(xmax))
	Lv[Xv+1] = 0.
	Lw[Xw+1] = 0.
end;

# ╔═╡ 1d580928-388e-11eb-2cb1-63cd2d219c0c
# now we wish to compute the likelihood of these observations by marginalizing
# over the ancestral state at node u at distance t=1, assuming parameters θ
t = 1.; θ

# ╔═╡ ca3748e0-3891-11eb-133f-238ead39b64f
function ϕ1ϕ2(θ::TwoTypeDL, s1, s2, t; kwargs...)
	ϕ0 = [s1; s2]
	ts = (0., t)
	pr = ODEProblem(system1!, ϕ0, ts, θ)
	sl = DifferentialEquations.solve(pr; dense=false, kwargs...)[:,end]
	(ϕ1=sl[1], ϕ2=sl[2])
end

# ╔═╡ 39e4bc10-38b3-11eb-18b7-77c73b161f0c
@timed ϕ1ϕ2(θ, exp(2π*im*1/16), exp(2π*im*1/16), 0.1; dense=false)

# ╔═╡ 874ee00e-388e-11eb-200e-6db912d21b80
function thewholedeal(Lv, θ, t; n=size(Lv, 1), N=2n)
	Lu = fill(-Inf, n, n)
	# we compute ϕ1 and ϕ2 for these values of s1 and s2
	ϕs = [ϕ1ϕ2(θ, exp(2π*im*u/N), exp(2π*im*v/N), t) for u=0:N-1, v=0:N-1]
	ϕ1 = first.(ϕs)
	ϕ2 = last.(ϕs)
	for j=0:n-1, k=0:n-1
		# compute transition probabilities from state (j, k)
		A = (ϕ1 .^ j) .* (ϕ2 .^ k)
		dft = fft(A)
		P = real(dft) ./ N^2
		P[P .< 0.] .= 0.
		P = log.(P)
		# for internal nodes
		if Lv isa Matrix
			for l=1:n, m=1:n
				Lu[j+1, k+1] = logaddexp(Lu[j+1, k+1], P[l, m] + Lv[l, m])
			end
		else
			# for leaf nodes, do a convolution for Z
			xobs = findfirst(x->x==0., Lv) - 1
			for l=0:xobs
				Lu[j+1, k+1] = logaddexp(Lu[j+1, k+1], P[l+1, xobs-l+1])
			end
		end
	end
	Lu
end

# ╔═╡ 5660edce-388f-11eb-3504-ad9cf63f750a
Xu1 = thewholedeal(Lv, θ, 0.05, N=25)

# ╔═╡ 34459de2-3890-11eb-0cd4-09651764f642
heatmap(Xu1, size=(220,200))

# ╔═╡ 60953b94-3897-11eb-2821-bb168cd07f46
Xu2 = thewholedeal(Lw, θ, 0.05, N=25)

# ╔═╡ 9df7c880-3897-11eb-0adf-c7f27ebac766
heatmap(Xu2, size=(220,200))

# ╔═╡ 421df4e6-389a-11eb-0a49-1dece8a538c1
heatmap(Xu1 .+ Xu2, size=(220,200))

# ╔═╡ ac07aff4-3896-11eb-3998-f51016c81714
md"""
It seems to be quite important to choose the `N` for the FFT large enough, otherwise we get artifacts... Note that not only does the size of `N` required to get a nice convex likelihood surface depends on the chosen bound on the state space, it also seems to depend on `t`. However, the larger `N`, the higher the computational burden of course.

While this looks promising, it seems likely that this will turn out to be computationally too intensive to apply? Of course transition probability matrices should be shared across the likelihood computations for iid observations.
"""

# ╔═╡ 6b649576-389a-11eb-20df-a931bf486c7e
# toy function for testing
function computePs(n, θ::TwoTypeDL{T}, t; N=2n) where T
	# we compute ϕ1 and ϕ2 for these values of s1 and s2
	ϕs = [ϕ1ϕ2(θ, exp(2π*im*u/N), exp(2π*im*v/N), t) for u=0:N-1, v=0:N-1]
	ϕ1 = first.(ϕs)
	ϕ2 = last.(ϕs)
	P = nothing
	for j=0:n-1, k=0:n-1
		# compute transition probabilities from state (j, k)
		A = (ϕ1 .^ j) .* (ϕ2 .^ k)
		dft = fft(A)
		P = real(dft) ./ N^2
		P[P .< 0.] .= 0.
		P = log.(P)
	end
	logsumexp(P)
end

# ╔═╡ 13fc4486-389b-11eb-257f-fd2c4940aa2e
@timed computePs(10, θ, 0.5, N=20)

# ╔═╡ 3e4215e0-389b-11eb-2af3-5bc6e02c9e4a
md"""
In smallish problems, we might be doing this about 10 to 20 times, so we'd have about 1 to 10 seconds for this per iteration depending on our choice of `N`, and that's without doing the actual pruning or AD... In other words, this will likely be prohibitively slow. Would AD actually work (would it get through the FFT and the ODE solver for complex input)? If we don't have efficient AD, we'd need an MWG sampler to do Bayesian inference (but that would be straightforward to implement I guess). Alternatively, we might resort to ML using Nelder-Mead or so, but that case would not be so nice, as we'd really want informative priors and posterior distributions, not point estimates, since we will have identifiability issues anyway.
"""

# ╔═╡ 82206f7c-38a1-11eb-195f-eb30f71ac7ea
for i=1:10 computePs(8, θ, 0.5, N=20) end

# ╔═╡ e67cd1a0-389b-11eb-3e14-8dc5971d85d2
# ForwardDiff.gradient(x->sum(computePs(10, TwoTypeDL(x...), 0.1)), asvector(θ));

# ╔═╡ d2618bb4-38c0-11eb-1a4d-5f0cea0d0466
md"""
See [DiffEq docs on sensitivity analysis](https://diffeq.sciml.ai/stable/analysis/sensitivity/#sensitivity) but I'm not sure about complex numbers, I don't even know how derivatives work for complex numbers...
"""

# ╔═╡ fa7d4492-38ae-11eb-119e-95d1ceb32793
md"""
## A complete likelihood function

For completeness, a reference/toy implementation for computing the loglikelihood of a family profile given a phylogeny the model.
"""

# ╔═╡ a1c0275e-38b1-11eb-08ea-95a1525fe6e6
data = (tree = readnw(readline("../data/oryzinae.6taxa.nw")), 
	    df = CSV.read("../data/nlr-N0-oib.6taxa-counts.csv", DataFrame));

# ╔═╡ f35933ca-38bc-11eb-3182-4358581d7b79
data.df[1:2,:]

# ╔═╡ 6c52ff16-38d3-11eb-1b95-13356412c853
dat = Dict(k=>v for (k,v) in zip(names(data.df), data.df[1,:]))

# ╔═╡ 7cf02308-38d5-11eb-3ee6-0152aded51b7
function leafL(x, n)
	L = fill(-Inf, n)
	L[x+1] = 0.
	return L
end

# ╔═╡ 9e9a80fc-38d3-11eb-2ad2-7589964058c1
function loglikelihood(θ, d, tree; n=8, N=16)
	function prune(node)
		isleaf(node) && return leafL(d[name(node)], n)
		L = map(children(node)) do child
			Lchild = prune(child)
			Ledge = thewholedeal(Lchild, θ, distance(child); N=N)
		end
		return L[1] .+ L[2]
	end
	prune(getroot(tree))
end

# ╔═╡ 12a88a4a-3927-11eb-2fb8-5fef760935c0
md"""
We seem to get about two seconds for a single likelihood evaluation with this unoptimized version. We will probably get some nudges here and there but it will remain in this range, especially since we might need to increase `N`.
"""

# ╔═╡ 9c63060c-393b-11eb-0f56-2d12e5c72cf4
function log_antidiagsum(A)
	n = size(A, 1)
	x = fill(-Inf, 2n)
	for i=0:n-1, j=0:i
		x[i + 1] = logaddexp(x[i + 1], A[i - j + 1, j + 1])
		x[2n - i] = logaddexp(x[2n - i], A[n-i+j, n-j])
	end
	return x
end

# ╔═╡ a1083632-393b-11eb-2a29-0b4f29561204
function integrate_rootprior(L, d::Distribution)  # integrate over a prior on the root
	l = log_antidiagsum(L)
	logsumexp(l[1:end] .+ logpdf(d, 0:length(l)-1))
	# exclude zero state and assume we need to shift the discrete prior
end

# ╔═╡ dcbfa474-3929-11eb-3e0a-9d5dfc5ca686
md"""
## Simulation

I guess the way to move forward is to test on simulated data whether we have any hope of recovering the parameters accurately without observing the two types at the leaves. If so, I believe it is worthwhile to investigate the model in some real applications and perhaps write a paper about it. If not, I guess it is a worthwhile exercise and addition to my thesis.
"""

# ╔═╡ c022d962-3933-11eb-243a-bd178a9b4eeb
import Distributions: Geometric

# ╔═╡ 6bc7e6ec-392d-11eb-1f73-89c5b519389d
getrates(p::TwoTypeDL, X) = [p.λ*X[1], p.μ₁*X[1], p.λ*X[2], p.μ₂*X[2], p.ν*X[2]]

# ╔═╡ 4fcc865a-392d-11eb-300e-d7a3b807a3b7
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

# ╔═╡ 63ba81aa-3930-11eb-05e7-11c9a2adc987
getparams(p::TwoTypeDL, id) = p  # no branch rates

# ╔═╡ 01652b68-3930-11eb-3c81-fde40926ed0c
function simulate(p::TwoTypeDL, X, tree)
	result = Dict{Symbol,Tuple{Int,Int}}()
	function simwalk(node, X)
		_X = simulate(getparams(p, id(node)), X, distance(node))
		isleaf(node) && return result[Symbol(name(node))] = _X
		for c in children(node)
			simwalk(c, _X)
		end
	end
	simwalk(tree, X)
	return (; result...)
end

# ╔═╡ 0c1a2c2c-3933-11eb-0811-f7ca048a380f
struct TwoTypeRootPrior{T}
	η1::T
	η2::T
end

# ╔═╡ 32daf81e-3933-11eb-0e82-3da097f6871e
function Base.rand(p::TwoTypeRootPrior) 
	Z = rand(Geometric(p.η1)) + 1
	X1 = rand(Binomial(Z, p.η2))
	X2 = Z - X1
	(X1, X2)
end

# ╔═╡ 863e606e-3932-11eb-24a1-e7f298459491
function simulate(θ::TwoTypeDL, root, tree, n)
	df = map(i->simulate(θ, rand(root), data.tree), 1:n) |> DataFrame
	ddf = select(df, names(df) .=> x->first.(x) .+ last.(x))
	rename!(ddf, names(df))
	ddf, df
end

# ╔═╡ 3442bf50-392e-11eb-2794-07eca5574f7f
simulate(θ, (2, 2), 1.10)

# ╔═╡ 8dbd474e-3930-11eb-36e2-7bc3290bf629
df, _ = simulate(θ, TwoTypeRootPrior(0.8, 0.9), data.tree, 10000);

# ╔═╡ 66e4a8a4-3937-11eb-3d90-bfda39fb4bc0
begin
	pm = proportionmap(df[:,1])
	xx = log10.([haskey(pm, i) ? pm[i] : 0. for i=0:15])
	scatter(xx, xscale=:log10, grid=false, color=:black, size=(300,200), legend=false)
end

# ╔═╡ 6a50a5c4-393a-11eb-3b3f-ab0950aa8bd2
md"""
We get the typical geometrically declining tail as expected.
"""

# ╔═╡ bdec013e-393b-11eb-372a-1d130efe833b
md"""
## Likelihood for a complete data set

Above we had a toy implementation of the likelihood as a proof of principle, however we need to share the transition probabilities we compute across different iid data vectors.
"""

# ╔═╡ ef3b00cc-3944-11eb-39f0-6fe9eb1ee877
function _prune_edge!(Lu, Lv::Matrix, P, j, k)
	n = size(Lu, 1)
	for l=1:n, m=1:n
		Lu[j+1, k+1] = logaddexp(Lu[j+1, k+1], P[l, m] + Lv[l, m])
	end
end

# ╔═╡ 32303f3e-3945-11eb-1455-b7e28ff33ac5
function _prune_edge!(Lu, Lv::Vector, P, j, k)
	n = size(Lu, 1)
	xobs = findfirst(x->x==0., Lv) - 1
	for l=0:xobs
		Lu[j+1, k+1] = logaddexp(Lu[j+1, k+1], P[l+1, xobs-l+1])
	end
end

# ╔═╡ 26c48faa-393c-11eb-188b-4d9989536028
function prune_edge(Lvs, θ, t, N, n)
	Lus = map(i->fill(-Inf, n, n), 1:length(Lvs))
	# we compute ϕ1 and ϕ2 for these values of s1 and s2
	ϕs = [ϕ1ϕ2(θ, exp(2π*im*u/N), exp(2π*im*v/N), t) for u=0:N-1, v=0:N-1]
	ϕ1 = first.(ϕs)
	ϕ2 = last.(ϕs)
	for j=0:n-1, k=0:n-1  # this loop could in principle be parallelized?
		# compute transition probabilities from state (j, k)
		A = (ϕ1 .^ j) .* (ϕ2 .^ k)
		dft = fft(A)
		P = real(dft) ./ N^2
		P[P .< 0.] .= 0.
		P = log.(P)
		# for internal nodes
		for (Lu, Lv) in zip(Lus, Lvs)
			_prune_edge!(Lu, Lv, P, j, k)
		end
	end
	Lus
end

# ╔═╡ c2ac309e-393c-11eb-3459-5d55fdd39c87
exdata = NamedTuple.(eachrow(df[1:100,:]))

# ╔═╡ b40b3242-393c-11eb-298c-69eee93a5e88
function loglikelihood(θ, rootprior, d, tree; n=8, N=16)
	function prune(node)
		isleaf(node) && return map(x->leafL(getfield(x, Symbol(name(node))), n), d)
		L = map(children(node)) do child
			Lchild = prune(child)
			Ledge = prune_edge(Lchild, θ, distance(child), N, n)
		end
		return map(i->L[1][i] .+ L[2][i], 1:length(d))
	end
	Ls = prune(getroot(tree))
	mapreduce(x->integrate_rootprior(x, rootprior), +, Ls), Ls
end

# ╔═╡ eab458fe-38d4-11eb-2183-074c62fc80f8
L = loglikelihood(θ, dat, data.tree, N=20)

# ╔═╡ b296c596-38d5-11eb-3adf-a3ba2c2a5262
heatmap(L, size=(220,200))

# ╔═╡ cb140c5a-38d5-11eb-2c3b-49a5d05b9f98
begin
	Proot = antidiagsum(exp.(L))
	plot(
		size=(600,200),
		plot(0:15, Proot, legend=false, color=:black, xticks=0:3:15),
		plot(0:15, log.(Proot), legend=false, color=:black, xticks=0:3:15)
	)
end

# ╔═╡ 54704e2e-3943-11eb-25f4-d937f788e42a
integrate_rootprior(L, Geometric(0.8))

# ╔═╡ 4adc64ca-393d-11eb-15ce-d7e259d64890
l, Ls = loglikelihood(θ, Geometric(0.66), exdata, data.tree)

# ╔═╡ 2c24ef38-3952-11eb-2320-d7aaa1f2c6a9
md"""
We have a different situation then we're used to, most of the time is spent in computing the relevant transition probabilities, not in the pruning algorithm, so using 1000 vs. 100 gene families does not lead to so much differences in terms of speed.
"""

# ╔═╡ 8c2933ee-3952-11eb-29c3-cba4fe5d6537
function getobjective(data, rootprior, tree; N=16, n=8)
	function objective(x)   # simple MAP objective function
		p = (λ=1., μ₁=1., ν=1., μ₂=1.)  # prior means...
		model = TwoTypeDL(exp.(x)...)
		!(model.μ₁ < model.λ < model.μ₂) && return Inf
		prior = 
			logpdf(Exponential(p.λ), model.λ) + 
			logpdf(Exponential(p.ν), model.ν) +
			logpdf(Exponential(p.μ₁), model.μ₁) + 
			logpdf(Exponential(p.μ₂), model.μ₂)
		-(loglikelihood(model, rootprior, data, tree, N=N, n=n)[1] + prior)
	end
end

# ╔═╡ d344b4d0-3952-11eb-082b-5339c22b5074
obj = getobjective(exdata, Geometric(0.8), data.tree)

# ╔═╡ 8661f968-3960-11eb-1081-9596232ca5f7
obj([0., 0.2, -1., 1.])

# ╔═╡ f61df29e-3952-11eb-00f4-119894907ca7
result = optimize(obj, [0., 0.2, -1., 1.], 
	method=NelderMead(), 
	iterations=5, 
	show_trace=true)

# ╔═╡ 567748d4-3953-11eb-031e-03b6425598a9
θhat = TwoTypeDL(exp.(result.minimizer)...)

# ╔═╡ 1bd03bb8-3954-11eb-3407-b1d4e6ce3cf5
θ

# ╔═╡ c3736598-3954-11eb-1d2a-913ec7e90c0d
loglikelihood(θ, Geometric(0.8), exdata, data.tree)[1]

# ╔═╡ 032bc8c4-3955-11eb-332c-b98df73b902e
loglikelihood(θhat, Geometric(0.8), exdata, data.tree)[1]

# ╔═╡ d458299c-395a-11eb-3072-e9c2961d8f4e
md"""
It doesn't work too well, but we'd need an MCMC sampler, because without estimates of uncertainty we're really nowhere in this case where the model is barely identifiable. Still, even though the parameters are hard to identify, we should be able to learn more than in the single-type linear BDP...
"""

# ╔═╡ a0894a14-3960-11eb-20ac-83582cc16a8d
# begin
# 	X = NamedTuple.(eachrow(data.df))
# 	res = optimize(
# 		getobjective(X, Geometric(0.8), data.tree, n=12, N=16), 
# 		[0., 0.2, -1., 1.], 
# 		method=NelderMead(), 
# 		iterations=50, 
# 		show_trace=true)
# end

# ╔═╡ Cell order:
# ╟─a426a434-33b9-11eb-2c0a-81b18487b3cb
# ╟─e69c43b2-33cf-11eb-15aa-4f0ba0fbb8e8
# ╟─d57f28e0-33d1-11eb-253b-47d7444fd676
# ╠═e52945b8-33d2-11eb-27e3-890af106f8dc
# ╠═6b2fe6be-388b-11eb-2e4d-e5ce95c938f2
# ╠═0d462160-389c-11eb-2c96-e3e74916de80
# ╠═ee062fea-33d3-11eb-1dd4-a7842503d0f7
# ╠═b27f9208-33d6-11eb-1e6f-49732a81a577
# ╠═ead9751c-33d9-11eb-19b2-d1c4562f2091
# ╠═55563f6c-33d8-11eb-3a90-07dbe909ab01
# ╟─cac5fb6a-33d9-11eb-3593-25fe89289b88
# ╟─0fbdbaae-356b-11eb-1957-55551088614f
# ╠═f649189e-356b-11eb-32cc-7da3efe6ba92
# ╠═66222e64-356d-11eb-3af4-216f3a7f0f38
# ╠═a10130c8-358d-11eb-1ab7-7deb044245e4
# ╠═f75f3bca-358f-11eb-0d4e-47c7860c38ef
# ╠═f391c27e-358f-11eb-1428-9f492b868981
# ╠═cbb0a286-358d-11eb-0735-fbeb520f630c
# ╠═76412cb6-358e-11eb-3ce3-29ce0f4a8b78
# ╠═775723f2-358f-11eb-10fe-eb18980e4dd9
# ╠═35003c06-3868-11eb-1c0c-9769f841c0c8
# ╟─67b4f18c-3596-11eb-29ac-09bd132ecba3
# ╟─578cc30a-359a-11eb-2309-e125ae2a8e28
# ╠═6f4f6fc6-386d-11eb-080e-9309615f7cf3
# ╠═77e7ab30-386d-11eb-21d1-cb3d7a0c1158
# ╠═84869a44-386e-11eb-2522-1b4e4cce2e65
# ╟─22fd6b60-388a-11eb-2b6e-b53cf2428ac9
# ╠═5249d270-388d-11eb-3d44-29dc1f6f980b
# ╠═b42e6d5c-388d-11eb-284e-d71aa40c2d12
# ╠═1d580928-388e-11eb-2cb1-63cd2d219c0c
# ╠═ca3748e0-3891-11eb-133f-238ead39b64f
# ╠═39e4bc10-38b3-11eb-18b7-77c73b161f0c
# ╠═874ee00e-388e-11eb-200e-6db912d21b80
# ╠═5660edce-388f-11eb-3504-ad9cf63f750a
# ╠═34459de2-3890-11eb-0cd4-09651764f642
# ╠═60953b94-3897-11eb-2821-bb168cd07f46
# ╠═9df7c880-3897-11eb-0adf-c7f27ebac766
# ╠═421df4e6-389a-11eb-0a49-1dece8a538c1
# ╟─ac07aff4-3896-11eb-3998-f51016c81714
# ╠═6b649576-389a-11eb-20df-a931bf486c7e
# ╠═13fc4486-389b-11eb-257f-fd2c4940aa2e
# ╟─3e4215e0-389b-11eb-2af3-5bc6e02c9e4a
# ╠═82206f7c-38a1-11eb-195f-eb30f71ac7ea
# ╠═d6091f72-389b-11eb-28a5-9f0c7b631f67
# ╠═e67cd1a0-389b-11eb-3e14-8dc5971d85d2
# ╟─d2618bb4-38c0-11eb-1a4d-5f0cea0d0466
# ╟─fa7d4492-38ae-11eb-119e-95d1ceb32793
# ╠═7381da4a-38b1-11eb-09ca-9f2e657dfd9a
# ╠═a1c0275e-38b1-11eb-08ea-95a1525fe6e6
# ╠═f35933ca-38bc-11eb-3182-4358581d7b79
# ╠═6c52ff16-38d3-11eb-1b95-13356412c853
# ╠═7cf02308-38d5-11eb-3ee6-0152aded51b7
# ╠═9e9a80fc-38d3-11eb-2ad2-7589964058c1
# ╠═eab458fe-38d4-11eb-2183-074c62fc80f8
# ╟─12a88a4a-3927-11eb-2fb8-5fef760935c0
# ╠═b296c596-38d5-11eb-3adf-a3ba2c2a5262
# ╠═cb140c5a-38d5-11eb-2c3b-49a5d05b9f98
# ╠═9c63060c-393b-11eb-0f56-2d12e5c72cf4
# ╠═a1083632-393b-11eb-2a29-0b4f29561204
# ╠═54704e2e-3943-11eb-25f4-d937f788e42a
# ╟─dcbfa474-3929-11eb-3e0a-9d5dfc5ca686
# ╠═a1517d28-392d-11eb-3846-5f3fd9bf6514
# ╠═c022d962-3933-11eb-243a-bd178a9b4eeb
# ╠═6bc7e6ec-392d-11eb-1f73-89c5b519389d
# ╠═4fcc865a-392d-11eb-300e-d7a3b807a3b7
# ╠═3442bf50-392e-11eb-2794-07eca5574f7f
# ╠═63ba81aa-3930-11eb-05e7-11c9a2adc987
# ╠═01652b68-3930-11eb-3c81-fde40926ed0c
# ╠═0c1a2c2c-3933-11eb-0811-f7ca048a380f
# ╠═32daf81e-3933-11eb-0e82-3da097f6871e
# ╠═863e606e-3932-11eb-24a1-e7f298459491
# ╠═8dbd474e-3930-11eb-36e2-7bc3290bf629
# ╠═66e4a8a4-3937-11eb-3d90-bfda39fb4bc0
# ╟─6a50a5c4-393a-11eb-3b3f-ab0950aa8bd2
# ╟─bdec013e-393b-11eb-372a-1d130efe833b
# ╠═26c48faa-393c-11eb-188b-4d9989536028
# ╠═ef3b00cc-3944-11eb-39f0-6fe9eb1ee877
# ╠═32303f3e-3945-11eb-1455-b7e28ff33ac5
# ╠═c2ac309e-393c-11eb-3459-5d55fdd39c87
# ╠═b40b3242-393c-11eb-298c-69eee93a5e88
# ╠═4adc64ca-393d-11eb-15ce-d7e259d64890
# ╟─2c24ef38-3952-11eb-2320-d7aaa1f2c6a9
# ╠═86f6c9c2-3952-11eb-118b-53328311f861
# ╠═8c2933ee-3952-11eb-29c3-cba4fe5d6537
# ╠═d344b4d0-3952-11eb-082b-5339c22b5074
# ╠═8661f968-3960-11eb-1081-9596232ca5f7
# ╠═f61df29e-3952-11eb-00f4-119894907ca7
# ╠═567748d4-3953-11eb-031e-03b6425598a9
# ╠═1bd03bb8-3954-11eb-3407-b1d4e6ce3cf5
# ╠═c3736598-3954-11eb-1d2a-913ec7e90c0d
# ╠═032bc8c4-3955-11eb-332c-b98df73b902e
# ╟─d458299c-395a-11eb-3072-e9c2961d8f4e
# ╠═a0894a14-3960-11eb-20ac-83582cc16a8d
