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
input = [("drosophila", "7540"),
         ("ygob", "0536"),
         ("primates-GO:0002376", "4663")]

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
"""
    kldivergence(p, q, eps)

Divergence of q from p, with smoothing `eps`
"""
function kldivergence(p, q, eps)
    pp = smooth(p, eps)
    qq = smooth(q, eps)
	d = 0.
	for i=eachindex(pp)
		d += pp[i]*(log(pp[i]) - log(qq[i]))
	end
	return d
end

"""
    smooth(x, eps)

Smooth a probability vector by setting every entry < eps to eps and correcting
the pther values accordingly.
"""
function smooth(x, eps) 
    nzero = 0
    nonzero = Int[]
    y = similar(x)
    for i=eachindex(x)
        if x[i] < eps 
            y[i] = eps 
            nzero += 1
        else
            y[i] = x[i]
            push!(nonzero, i)
        end
    end
    if nzero != 0 
        y[nonzero] .-= eps/nzero 
    end
    return y
end
            
kldivs = map(alldata) do data
    yall = obsproportions(data.data, 0:9)
    col = :all
    eps = 1/(nrow(data.data) * ncol(data.data))
    p = yall[:,col] #yall[:,:all]
    kldiv = map(["two-type", "single-type"]) do key
        map(1:10:nrow(data.sims[key])) do k
            q = data.sims[key][k:k+9,col]
            kldivergence(p, q, eps)
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
            xlabel="\$D(p,\\tilde{p})\$", guidefont=8,
            title= title, titlefont=7, title_loc=:left,
            ylabel="density", fontfamily="helvetica", 
            xrotation=45, tickfont=7)
end
plot(ps..., layout=(1,3), size=(550,150), margin=5mm)

savefig(joinpath(pth, "kldiv.pdf"))


# KL divergence for each  species
# -------------------------------
function kldivplot(data, kmax=9)
    yall = obsproportions(data.data, 0:kmax);
    eps = 1/(nrow(data.data))
    kldivs = map(names(yall)[3:end]) do sp
        @info "" sp
        p = yall[:,sp] 
        kldiv = map(["two-type", "single-type"]) do key
            map(1:10:nrow(data.sims[key])) do k
                q = data.sims[key][k:k+kmax,sp]
                kldivergence(p, q, eps)
            end
        end
        m1, q1 = mean(kldiv[1]), quantile(kldiv[1], [0.025, 0.975])
        m2, q2 = mean(kldiv[2]), quantile(kldiv[2], [0.025, 0.975])
        @info "two-type" m1 q1 
        @info "single-type" m2 q2
        sp, kldiv
    end
    kwargs = (title_loc=:left, titlefont=8, fill=true, fillalpha=0.2, 
              normalize=true, color=[:firebrick :black], xrotation=0, 
              legend=false, grid=false, bins=25, tickfont=6)
    [stephist(x; title=sp, kwargs...) for (sp, x) in kldivs]
end

p1 = plot(kldivplot(alldata[1])..., layout=(2,4), size=(800,200))
p2 = plot(kldivplot(alldata[2])..., layout=(2,4), size=(800,200))
p3 = plot(kldivplot(alldata[3])..., size=(800,300))
plot(p1, p2, p3, layout= grid(3, 1, heights=[0.28, 0.28, 0.44]), size=(800,800))
savefig(joinpath(pth, "kldiv-sp.pdf"))


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
