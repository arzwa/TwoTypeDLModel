# Code for producing figure 4 and related supplementary material
# Author: Arthur Zwaenepoel

# Comparing posterior predictive distribution of the two-type model to the
# single-type model
using DataFrames, CSV, StatsBase, Plots, StatsPlots, Measures

# taxon names
nmap = Dict(x[1]=>x[2] for x in eachrow(CSV.read("data/species.csv", DataFrame)))

# output path
pth  = mkpath("output/fig04")

# read in data
input = [("drosophila", "0026"),
         ("ygob", "2947"),
         ("primates-GO:0002376", "6419")]

alldata = map(input) do (prefix, suffix)
    (data = CSV.read("data/$prefix.csv", DataFrame),
     sims = Dict("two-type" => CSV.read("output/table01-a/$prefix/$suffix/ppsim.csv", DataFrame),
                 "single-type" => CSV.read("output/table01-b/$prefix/ppsim2.csv", DataFrame),
                 "single-type-ne" => CSV.read("output/table01-b/$prefix/ppsim3.csv", DataFrame)))
end

# 1. Plot of deviations
# =====================
function obsproportions(obs, rnge, ϵ=1e-6)
	x = proportions(Matrix(obs), rnge) .+ ϵ
	y = mapreduce(x->proportions(x, rnge) .+ ϵ, hcat, eachcol(obs))
	return DataFrame(hcat(rnge, x, y), Symbol.(vcat(:k, :all, names(obs))))
end

function devplot(alldata, i)
    data = alldata[i]
    yall = obsproportions(data.data, 0:9);
    ps = map(names(data.data)) do sp
        p = plot(grid=false, legend=false,
            title=nmap[sp], title_loc=:left, titlefont=8,
            titlefontfamily="helvetica oblique", fontfamily="sans-serif");
        colors = [:firebrick, :grey]
        map(enumerate(["two-type", "single-type"])) do (i, key)
            df = filter(x->x[:k] < 5, data.sims[key])
            map(groupby(df, :k)) do gdf
                k = Int(gdf[1,:k])
                yobs = filter(x->x[:k] == k, yall)[:,sp]
                devs = yobs .- gdf[:,sp]
                violin!(repeat([k], 1000), devs, color=colors[i], alpha=0.5)
            end
            hline!([0], color=:firebrick, linestyle=:dash)
        end
        plot(p)
    end
    if i == 3
        ylims!.(ps, -0.12, 0.22)
        annotate!(ps[end], 6.5, 0.12, text("two-type", 8, :left, :firebrick))
        annotate!(ps[end], 6.5, 0.08, text("single-type", 8, :left, :black))
        ylabel!.(ps[[1,5,9]], "\$p_k - \\tilde{p}_k\$", guidefont=9)
        xlabel!.(ps[9:11], "\$k\$", guidefont=9)
        plot(ps..., size=(600,400))
    else
        ylims!.(ps, -0.12, 0.12)
        annotate!(ps[end], 4.5, 0.11, text("two-type", 8, :right, :firebrick))
        annotate!(ps[end], 4.5, 0.08, text("single-type", 8, :right, :black))
        ylabel!.(ps[[1,5]], "\$p_k - \\tilde{p}_k\$", guidefont=9)
        xlabel!.(ps[5:8], "\$k\$", guidefont=9)
        plot(ps..., size=(600,300), layout=(2,4))
    end
end

for i in 1:length(alldata)
    plot(devplot(alldata, i))
    savefig(joinpath(pth, "$i.pdf"))
end

# 2. KL divergence between posterior predictive and observed
# ==========================================================
function kldivergence(p, q)
	d = 0.
	for i=1:length(p)
		d += p[i]*(log(p[i]) - log(q[i]))
	end
	return d
end

kldivs = map(alldata) do data
    yall = obsproportions(data.data, 0:9)
    p = yall[:,:all]
    kldiv = map(["two-type", "single-type"]) do key
        map(1:10:nrow(data.sims[key])) do k
            q = data.sims[key][k:k+9,:all] .+ 1e-5
            kldivergence(p, q)
        end
    end
    m1, q1 = mean(kldiv[1]), quantile(kldiv[1], [0.025, 0.975])
    m2, q2 = mean(kldiv[2]), quantile(kldiv[2], [0.025, 0.975])
    @info "two-type" m1 q1 
    @info "single-type" m2 q2
    kldiv
end

titles = ["Drosophila", "Yeast", "Primates (GO:0002376)"]
ps = map(zip(kldivs, titles)) do (kldiv, title)
    stephist(kldiv, fill=true, fillalpha=0.2, color=[:firebrick :grey],
            grid=false, normalize=true, legend=false, size=(300,200),
            xlabel="\$D_{\\mathrm{KL}}(p\\ ||\\tilde{p})\$", guidefont=8,
            title= title, titlefont=7, title_loc=:left,
            ylabel="density", fontfamily="helvetica", xrotation=45, tickfont=7)
end
plot(ps..., layout=(1,3), size=(550,150), margin=5mm)
savefig(joinpath(pth, "kldiv.pdf"))


# 3. complete posterior predictive distributions
# ==============================================
labels = ["two-type", "single-type", "single-type-ne"]
ps = map(alldata) do data
    yall = obsproportions(data.data, 0:9)
    ycor = obsproportions(data.data, 1:9)
    map(labels) do key
        sp = :all
        p = plot(grid=false, legend=false);
        map(groupby(data.sims[key], :k)) do gdf
            violin!(p, gdf[:,sp],
                yscale=:log10, xlabel="\$k\$", ylabel="\$p_k\$",
                title=key, title_loc=:left, titlefont=8,
                color=:white, ylim=(5e-6, 2))
        end
        key == "single-type-ne" ? 
            scatter!(ycor[:,sp], color=:black) : 
            scatter!(yall[:,sp], color=:black)
        plot(p)
    end 
end 
plot(vcat(ps...)..., layout=(3,3))
savefig(joinpath(pth, "allpps-violin.pdf"))
