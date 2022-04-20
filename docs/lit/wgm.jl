# ## WGM model

# Not yet implemented with hyperpriors!
# i.e. the parameter vector is currently assumed to be (λ, μ₁, ν, μ₂, qs...)
using Pkg; Pkg.activate(@__DIR__)
using TwoTypeDLModel, CSV, DataFrames, NewickTree, Distributions
using TwoTypeDLModel: TwoTypeDLWGM


# simulation
tr = nw"(((A:0.2,B:0.2):0.1):0.2,C:0.5);"
i = id(filter(x->length(children(x)) == 1, postwalk(tr))[1])
p = TwoTypeDLModel.BBGPrior(0.92, 10., 0.5, 1:10)
m = TwoTypeTree(tr, TwoTypeDLWGM(0.5, 0.2, 0.3, 4., [0.5], [2], Dict(i=>1)), p)

X, Y = simulate(m, 500) 

s = PSettings(n=12, N=24)
d = CountDAG(X, tr, s.n)
priors = (Beta(), Beta(), Beta(), Exponential(5.), Beta())
c = Chain(m, priors, d, s)
xs = sample(c, 1250, root=false)

y = permutedims(hcat(map(x->TwoTypeDLModel.transform(c, x.θ), xs)...))
for i=1:3; y[:,i] .*= y[:,4]; end

sims = (X, Y, c, xs, y)


# Poplar
df = CSV.read("scripts/data/9dicots-f01-1000.csv", DataFrame)
tr0 = nw"((cpa:0.106,ptr:0.106):0.011,bvu:0.117);"
sp = name.(getleaves(tr0))
df = df[:,name.(getleaves(tr0))]
df = filter(x->sum(x[1:3]) >0 && x[4] > 0 && sum(x) ≤ 10, df)

# no WGD
p = TwoTypeDLModel.BBGPrior(0.92, 10., 0.5, 1:10)
s = PSettings(n=14, N=36, abstol=1e-8, reltol=1e-8)
m = TwoTypeTree(tr, TwoTypeDL(rand(4)...), p)
d = CountDAG(df, tr, s.n)
TwoTypeDLModel.loglikelihood(m, d, s)

priors = (Beta(), Beta(), Beta(), Exponential(5.))
c = Chain(m, priors, d, s)
xs = sample(c, 1250, root=false)

y = permutedims(hcat(map(x->TwoTypeDLModel.transform(c, x.θ), xs)...))
for i=1:3; y[:,i] .*= y[:,4]; end

results1 = (c, xs, y)

# with WGD
tr = nw"((cpa:0.106,(ptr:0.065):0.041):0.011,bvu:0.117);"
i = id(filter(x->length(children(x)) == 1, postwalk(tr))[1])
θ = TwoTypeDLWGM(2.5, 0.75, 1., 5.7, [0.5], [2], Dict(i=>1))  
p = TwoTypeDLModel.BBGPrior(0.92, 10., 0.5, 1:10)
s = PSettings(n=14, N=36, abstol=1e-8, reltol=1e-8)
m = TwoTypeTree(tr, θ, p)
d = CountDAG(df, tr, s.n)
TwoTypeDLModel.loglikelihood(m, d, s)

X, Y = simulate(m, 500) 
df = df[:,names(X)]
plot(heatmap(Matrix(df)), heatmap(Matrix(X)))

d = CountDAG(X, tr, s.n)
priors = (Beta(1,5), Beta(1,5), Beta(1,5), Exponential(5.), Beta())
c = Chain(m, priors, d, s)
xs = sample(c, 1250, root=false)
ys = permutedims(hcat(getfield.(xs, :θ)...))
y = permutedims(hcat(map(x->TwoTypeDLModel.transform(c, x.θ), xs)...))
for i=1:3; y[:,i] .*= y[:,4]; end

d = CountDAG(df, tr, s.n)
priors = (Beta(1,5), Beta(1,5), Beta(1,100), Exponential(5.), Beta())
c = Chain(m, priors, d, s)
xs = sample(c, 1250, root=false)
y = permutedims(hcat(map(x->TwoTypeDLModel.transform(c, x.θ), xs)...))
for i=1:3; y[:,i] .*= y[:,4]; end

results2 = (c, xs, y)

θ = TwoTypeDLWGM(1.8, 0.9, 0.00, 5.7, [0.00], [2], Dict(i=>1))  
#θ = TwoTypeDL(1.8, 0.9, 0.8, 13.7)  
p = TwoTypeDLModel.BBGPrior(0.92, 10., 0.5, 1:10)
m = TwoTypeTree(tr, θ, p)
X, Y = simulate(m, 500) 
plot(heatmap(Matrix(df)), heatmap(Matrix(X)))

# Yeast
df = CSV.read("scripts/data/ygob-10taxa.csv", DataFrame)
tr = readnw(readline("scripts/data/ygob-10taxa.nw"))

sp = name.(getleaves(tr))
#heatmap(Matrix(df[:,sp]), xticks=(1:length(sp), sp), color=:binary)
        
i = id(filter(x->length(children(x)) == 1, postwalk(tr))[1])
θ = TwoTypeDLWGM(0.2, 0.1, 0.2, 5., [0.2], [2], Dict(i=>1))  
p = TwoTypeDLModel.BBGPrior(0.98, 4.06, 0.03, 1:10)
s = PSettings(n=12, N=24)
m = TwoTypeTree(tr, θ, p)
d = CountDAG(df, tr, s.n)
priors = (Beta(), Beta(), Beta(), Exponential(5.), Beta())
c = Chain(m, priors, d, s)
xs = sample(c, 1250, root=false)

y = permutedims(hcat(map(x->TwoTypeDLModel.transform(c, x.θ), xs)...))
for i=1:3; y[:,i] .*= y[:,4]; end

# We get q ~ 4%, which seems quite low, but we have a seemingly
# inflated ν, which could indicate that the WGD is differently
# retained in (tph,vpo) than the others (as suggested by heatmap) and
# hence an artefact of the model that WGD retention is instantaneous.
