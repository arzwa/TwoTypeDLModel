using Pkg; Pkg.activate("..")
using TwoTypeDLModel
Pkg.activate(".")
using CSV, DataFrames, Distributions, NewickTree, ThreadTools, StatsBase

pdf = CSV.read("output/table01-a/drosophila/0026/chain.csv", DataFrame)
pth = mkpath("output/fig05") 
data = CSV.read("data/drosophila.csv", DataFrame)
tree = readnw(readline("data/drosophila.nw"))
nmap = Dict(x[1]=>x[2] for x in eachrow(CSV.read("data/species.csv", DataFrame)))
settings = PSettings(n=12, N=16, abstol=1e-6, reltol=1e-6)

# 1. Posterior predictive simulations
# ===================================
function simfunrh(row, tree, settings, N, cond)
    rates = zeros(N,2)
    res = map(1:N) do j
        η = row[:λ] / row[:μ₂]
        α = η * (row[:ζ] + 1)  
        β = (1 - η) * (row[:ζ] + 1)
        ξ = rand(Beta(α, β))  # λ/μ
        λ = row[:μ₂] * ξ 
        model = TwoTypeTree(tree, 
            TwoTypeDL(λ, row[2:4]...),   
            TwoTypeDLModel.BBGPrior(row[:η], row[:ζ], row[:r], 1:(settings.n-1)*2))
        X, _ = TwoTypeDLModel.simulate(model, 1)
        rates[j,1] = λ; rates[j,2] = row[:μ₂]
        X
    end 
    X = vcat(res...)
    y = mapreduce(x->proportions(x, 0:9), hcat, eachcol(X))
    z = proportions(Matrix(X), 0:9)
    Y = DataFrame(hcat(0:9, z, y), vcat(:k, :all, Symbol.(names(X))))
    Y, rates
end

N = nrow(data)
ys = tmap(i->simfunrh(pdf[i,:], tree, settings, N, x->cond(x)), 1000:10:nrow(pdf)) 

CSV.write(joinpath(pth, "ppsim-rhlambda.csv"), vcat(first.(ys)...))
CSV.write(joinpath(pth, "rates-rhlambda.csv"), DataFrame(vcat(last.(ys)...), ["λ", "μ₂"]))


# 2. Plots
# ========
using Plots, StatsPlots, Measures, KernelDensity

X = (norh  = CSV.read("output/table01-a/drosophila/0026/ppsim.csv", DataFrame),
     duprh = CSV.read(joinpath(pth, "ppsim-rhlambda.csv"), DataFrame),
     rates = CSV.read(joinpath(pth, "rates-rhlambda.csv"), DataFrame)[1:10:end,:])

function addpp!(p, sims, n; rnge=0:9, kwargs...)
    for k=rnge
        xs = filter(x->x[:k] == k, sims)[:,n] .+ 1e-6
        q1, q2 = quantile(xs, [0.025, 0.975])
        m = mean(xs)
        ys = [m, m]
        plot!(p, [k-0.5, k+0.5], ys, ribbon=(ys .- q1, q2 .- ys); kwargs...)
    end
end

function observedps(data)
    observed = map(names(data)) do n
        n=>proportions(data[:,n], 0:9)
    end |> Dict
    observed["all"] = proportions(Matrix(data), 0:9)
    return observed
end

# contour plot of rates
kdep = let x = log10.(Matrix(X.rates)[:,[2,1]])
    p = contourf(kde(x, bandwidth=(0.005,0.1)), color=:cividis,
        ylim=(-9, 1.), size=(300,300),
        grid=false, levels=0:0.5:6, colorbar=false)
    xs = round.(exp10.(0.78:0.02:0.84), digits=1)
    xticks!(p, log10.(xs), string.(xs))
    xlabel!(p, "\$\\mu_2\$")
    ylabel!(p, "\$\\log_{10}\\lambda\$")
end

# posterior predictive densities
eightps = let X = X, observed=observedps(data)
    ps = map(filter(x->x[1]!="all", collect(sort(observed)))) do (n, x)
        p = plot(grid=false, legend=false,
            title=nmap[n], title_loc=:left,
            titlefont=("helvetica oblique", 8))
        # posterior predictive distribution
        addpp!(p, X.norh, n, color=:black, fillalpha=0.2)
        # addpp!(p, X.loss, n, color=:navy, fillalpha=0.2)
        addpp!(p, X.duprh, n, color=:firebrick, fillalpha=0.2)
        # observed data
        scatter!(p, 0:9, x .+ 1e-6, yscale=:log10,
            color=:black, markersize=3, ylim=(5e-6, 2.))
        p
    end
    xlabel!.(ps[end-3:end], "\$k\$")
    ylabel!.(ps[[1,5]], "\$p_k\$")
    yticks!.(ps[[2,3,4,6,7,8]], [Float64[] for i=1:6])
    yticks!.(ps[[1,5]], [[10^-4, 10^-2, 10^-0] for i=1:2])
    xticks!.(ps[[1,2,3,4]], [Float64[] for i=1:4])
    ps
end;

# join the plots
plot(kdep, eightps..., layout=(@layout [a{0.25w} grid(2,4)]),
     size=(700,200), bottom_margin=3mm)

savefig(joinpath(pth, "fig05.pdf"))
