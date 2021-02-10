# Code for producing figure 2
# Author: Arthur Zwaenepoel
using Turing, Distributions, StatsBase, CSV, DataFrames, TwoTypeDLModel
using TwoTypeDLModel: quantiles, BetaGeometric
using Plots, StatPlots, Printf, Parameters
using DeadBird: stepplot, stepplot!

# Models
# ======
# 1. Geometric model (simple linear BDP)
# --------------------------------------
@model geometric(Y) = begin
    p ~ Beta()
    if !(zero(p) < p < one(p))
        p = abs(p - 1e-9)
    end
    Y ~ Geometric(p)
end

# Efficiently compute loglikelihood for Geometric distribution when data is a
# vector of counts.
function Distributions.loglikelihood(d::Geometric, x::Vector)
    ℓ = 0.
    for (i, k) in enumerate(x)
        ℓ += k * logpdf(d, i-1)
    end
    return ℓ
end

function sim_out(Ys, Y)
    Ys = mapreduce(x->proportions(x, 1:maximum(Y)+1), hcat, Ys)
    return quantiles(Ys)
end

function simulate_geometric(chain, Y)
    fn = i->rand(Geometric(get(chain[i], :p).p[1]), length(Y)) .+ 1
    Ys = map(fn, 1:length(chain))
    sim_out(Ys, Y)
end

# 2. Beta-geometric model
# -----------------------
@model betageometric(Y) = begin
    η ~ Beta()
    ζ ~ Turing.FlatPos(0.)
    Y ~ BetaGeometric(η, ζ)
end

function simulate_betageometric(chain, Y)
    fn(x) = BetaGeometric(get(x, :η).η[1], get(x, :ζ).ζ[1])
    Ys = map(i->rand(fn(chain[i]), length(Y)), 1:length(chain))
    sim_out(Ys, Y)
end

# Analysis
# ========
sumry(x) = @printf "  %.3f (%.3f, %.3f)\n" mean(x) quantile(x, 0.025) quantile(x, 0.975)

# All three data sets
X = map([
       CSV.read("data/drosophila-8taxa-max10-oib.csv", DataFrame),
       CSV.read("data/ygob-8taxa-max10-oib.csv", DataFrame),
       CSV.read("data/primates-GO:0002376-oib-max10.csv", DataFrame)
      ]) do df
    Xall = filter(x->x>0, vec(Matrix(df)))
    Xcount = counts(Xall)
    c1 = sample(betageometric(Xcount), NUTS(), 1000)
    X1 = simulate_betageometric(c1, Xall)
    c2 = sample(geometric(Xcount), NUTS(), 1000)
    X2 = simulate_geometric(c2, Xall)
    y  = log10.(proportions(Xall) .+ 1e-7)
    (c1=c1, c2=c2, X1=X1, X2=X2, y=y)
end

ns = ["Drosophila", "Yeasts", "Primates (GO:0002376)"]
map(enumerate(X)) do (i,x)
    @unpack X1, X2, y, c1, c2 = x
    println(ns[i])
    sumry(DataFrame(c1)[:,:η])
    sumry(DataFrame(c1)[:,:ζ])
    sumry(DataFrame(c2)[:,:p])
    p = stepplot(X1[:,1], ribbon=(X1[:,2], X1[:,3]), 
                 xscale=:log10, vert=false, fillalpha=0.2,
                 title=ns[i], title_loc=:left, titlefont=7, 
                 titlefontfamily=i==1 ? "helvetica oblique" : "helvetica")
    stepplot!(X2[:,1], ribbon=(X2[:,2], X2[:,3]), 
              color=:firebrick, xscale=:log10, 
              vert=false, fillalpha=0.2)
    xticks = (1.5:1:10.5, ["1", "2", "3", "4", "5", "", "", "", "", "10"])
    scatter!(1.5:1:length(y)+0.5, y, xticks=xticks, 
             color=:black, grid=false, legend=false)
    ylims!(-5, 0.5)
    xlabel!("\$n\$", guidefont=9)
    ylabel!("\$\\log_{10} f\$")
    i==1 && (annotate!(7, -1.8, "Beta-Geometric", 6))
    i==1 && (annotate!(1.8, -3.5, text("Geometric", 6, :left, :top, :firebrick)))
    p
end |> ps->plot(ps..., layout=(1,3), size=(600,150))

pth = mkpath("output/fig02")
savefig(joinpath(pth, "fig02.pdf")
