@testset "PotentialDistributions" begin
    d = Potential{Float32}(m²=0.5, λ=0.0, xmin=-4, xmax=4) # should be similar to Distributions.Normal(0, 1)
    rng = MersenneTwister(0)
    samples = rand(rng, d, 10^6)
    @test abs(mean(samples)) < 0.0012
    @test abs(mean(samples .^ 2)) - 1.0f0 < 1e-4
end
