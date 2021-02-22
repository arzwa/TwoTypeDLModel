using Documenter, TwoTypeDLModel, Literate
outdir = joinpath(@__DIR__, "src")
srcdir = joinpath(@__DIR__, "lit")
mkpath(outdir)

fnames = ["index.jl"]
output = String[]

for f in fnames
    target = string(split(f, ".")[1])
    outpath = joinpath(outdir, target*".md")
    push!(output, relpath(outpath, joinpath(@__DIR__, "src")))
    @info "Literating $f"
    Literate.markdown(joinpath(srcdir, f), outdir, documenter=true)
    x = read(`tail -n +4 $outpath`)
    write(outpath, x)
end

makedocs(
    modules = [TwoTypeDLModel],
    sitename = "TwoTypeDLModel.jl",
    authors = "Arthur Zwaenepoel",
    doctest = false,
    pages = ["Index" => "index.md"])

deploydocs(repo = "https://github.com/arzwa/TwoTypeDLModel", devbranch = "main")
