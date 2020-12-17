using TwoTypeDLModel
using TwoTypeDLModel: loglikelihood, ϕ1ϕ2, ϕ_fft_grid, transitionp_fft
using TwoTypeDLModel: Chain, mwg_sweep!
using AdvancedMH: AdaptiveProposal
using Test, DataFrames, NewickTree, Distributions

@testset "TwoTypeDLModel" begin
    tree1 = readnw("((A:1.2,B:1.2):0.8,C:2.0);")
    tree2 = readnw("((og:0.19,ob:0.19):0.62,((on:0.39,osi:0.39)"*
                   ":0.16,(or:0.27,osj:0.27):0.28):0.26);")
    
    # extinction probability for linear BDP
    p10_lbdp(λ, μ, t) = μ*(1 - exp((λ - μ)*t))/(μ - λ*exp((λ - μ)*t))

    @testset "Probability generating functions" begin
        # the extinction probability given starting state (0,1) and ν=0
        # corresponds to the extinction probability of a linear BDP.
        test = [(0.1, 0.2, 1.), (1.5, 0.8, 0.1)]
        for (λ, μ, t) in test 
            θ = TwoTypeDL(λ, 0., 0., μ)  # ν = 0., stay in type 2
            f1, f2 = ϕ1ϕ2(θ, 1., 1., t)
            @test f1 ≈ one(f1)  # test ϕ(1,1) = 1. 
            @test f2 ≈ one(f1)
            f1, f2 = ϕ1ϕ2(θ, 0., 0., t)  
            @test f2 ≈ p10_lbdp(λ, μ, t)  # test against linear BDP
            settings = PSettings()
            f1, f2 = ϕ_fft_grid(θ, t, settings)
            lP = transitionp_fft(f1, f2, 0, 1)
            @test exp(lP[1,1]) ≈ p10_lbdp(λ, μ, t) atol=1e-5
        end
    end
    
    @testset "Likelihood and N" begin
        θ = TwoTypeDL(0.1, 0.01, 0.1, 0.5)  
        X = Profiles(DataFrame(:A=>[1,2], :B=>[0,2], :C=>[4,6]))
        p = GeometricPrior(0.8, 0.5)
        m = TwoTypeTree(tree1, θ, p)
        l = map(N->loglikelihood(m, X, PSettings(n=N÷2, N=N)), 15:5:50)
        @test all(isapprox.(l, -20.050456, atol=1e-3)) 
    end

    @testset "Simulation and larger data set" begin
        θ = TwoTypeDL(0.1, 0.01, 0.1, 0.5)  
        m = TwoTypeTree(tree2, θ, GeometricPrior(0.8, 0.5))
        X, Y = TwoTypeDLModel.simulate(m, 100)
        @time lx = loglikelihood(m, Profiles(X))
        @time ly = loglikelihood(m, Profiles(Y))
        @test ly < lx  # should always be, state space much larger for Y
    end

    @testset "Extinction probabilities Monte Carlo test" begin
        for i=1:10
            μ = exp(randn())
            x = μ .* rand(3) ./ 10
            η = 0.5 + rand()/2
            q = rand()
            θ = TwoTypeDL(x..., μ)  
            R = GeometricPrior(η, q)
            m = TwoTypeTree(tree2, θ, R)
            X, Y = TwoTypeDLModel.simulate(m, 50000)
            X_ = filter(x->any(Array(x[[:og, :ob]]) .> 0) && 
                      any(Array(x[[:on,:osi,:osj,:or]]) .> 0), X)        
            p̂ = nrow(X_)/nrow(X)
            p = exp(TwoTypeDLModel.p_nonextinct_bothclades(m, PSettings(N=32, n=20)))
            #@info "probability of non-extinction in both clades" θ R p p̂
            @test p ≈ p̂ atol=1e-2
        end
    end

    @testset "Custom MWG algorithm, prior sample" begin
        θ = TwoTypeDL(0.3, 0.1, 0.3, 5.5)  
        model = TwoTypeTree(tree2, θ, GeometricPrior(0.8, 0.5))
        prior = (Beta(), Beta(), Beta(), Exponential(5), Beta()) 
        props = [AdaptiveProposal(Normal(0, 0.1)) for i=1:5]
        chain = Chain(model, prior)
        smple = map(1:10000) do i
            mwg_sweep!(chain)
            chain.state.θ
        end |> x->permutedims(reduce(hcat, x)) |> 
               X->TwoTypeDLModel.transform(chain, X)
        ms = mean(smple, dims=1)
        for (m, p) in zip(ms, priors)
            @test m ≈ mean(p) rtol=1e-1
        end
    end

    @testset "Custom MWG algorithm" begin
        θ = TwoTypeDL(0.3, 0.1, 0.3, 5.5)  
        model = TwoTypeTree(tree2, θ, GeometricPrior(0.8, 0.5))
        D, _  = TwoTypeDLModel.simulate(model, 500)
        data  = Profiles(D)
        prior = (Beta(), Beta(), Beta(), Exponential(5), Beta()) 
        props = [AdaptiveProposal(Normal(0, 0.1)) for i=1:5]
        chain = Chain(model, prior, data)
        smple = map(1:10) do i
            @printf "%10d " i
            println(join([(@sprintf "%3.5f" x) for x in chain.state.θ], " "))
            mwg_sweep!(chain)
            chain.state.θ
        end |> x->permutedims(reduce(hcat, x)) |>
               X->TwoTypeDLModel.transform(chain, X)
    end

end


