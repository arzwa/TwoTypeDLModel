using TwoTypeDLModel
using TwoTypeDLModel: loglikelihood, ϕ1ϕ2, ϕ_fft_grid, transitionp_fft
using Test, DataFrames, NewickTree

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
        p = RootPrior(0.8, 0.5)
        m = TwoTypePhyloModel(tree1, θ, p)
        l = map(N->loglikelihood(m, X, PSettings(n=N÷2, N=N)), 15:5:50)
        @test all(isapprox.(l, -20.94242, atol=1e-3)) 
    end

    @testset "Simulation and larger data set" begin
        θ = TwoTypeDL(0.1, 0.01, 0.1, 0.5)  
        m = TwoTypePhyloModel(tree2, θ, RootPrior(0.8, 0.5))
        X, Y = simulate(m, 100)
        @time lx = loglikelihood(m, Profiles(X))
        @time ly = loglikelihood(m, Profiles(Y))
        @test ly < lx  # should always be, state space much larger for Y
    end

    @testset "Extinction probabilities Monte Carlo test" begin
        θ = TwoTypeDL(0.2, 0.2, 0.4, 1.5)  
        m = TwoTypePhyloModel(tree2, θ, RootPrior(0.8, 0.5))
        X, Y = simulate(m, 10000)
        X_ = filter(x->any(Array(x[[:og, :ob]]) .> 0) && 
                  any(Array(x[[:on,:osi,:osj,:or]]) .> 0), X)        
        p = nrow(X_)/nrow(X)
        @info "probability of non-extinction in both clades" p
    end

end


